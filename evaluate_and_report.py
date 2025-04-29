import itertools
import json

import fire
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from data.dataset_metadata import metadata_by_dataset

ANNOTATED_LOCATIONS = [
    "Table Column",
    "Table Cell",
    "Table Header",
    "Table Caption",
    "Footnote",
    "Page Text",
    "Not on page",
    "Not Present",
    "generic",  # fallback case
]

NUMERICAL_THRESHOLD = 5


def get_columns_to_predict(column_config: dict) -> list[str]:
    return (
        column_config["numerical"]
        + column_config["textual"]
        # + list(itertools.chain(*[group["columns"] for group in column_config["aligned_lists"]]))
    )


## only makes sense for fully location-annotated data
# def get_comparison_columns(data_df):
#     comparison_columns = []
#     for column in NUMERICAL_COLUMNS + TEXT_COLUMNS:
#         if data_df[column + "_location"].iloc[0] not in ["Not Present", "Not on page"]:
#             comparison_columns.append(column)
#     return comparison_columns


def only_numeric(data, column_config):
    """Get only numeric "columns" from either a row or series"""
    return data[
        [
            column
            for column in (data.index if isinstance(data, pd.Series) else data)
            if column in column_config["numerical"]
        ]
    ]


def only_textual(data, column_config):
    """Get only textual "columns" from  either a row or series, and normalize the text."""
    textual_columns = column_config["textual"]
    if data.empty:
        return data
    if isinstance(data, pd.Series):
        filtered = data[[column for column in data.index if column in textual_columns]]
        normalized = filtered.str.lower().str.replace("\W+", "_", regex=True)
    else:
        filtered = data[[column for column in data if column in textual_columns]]
        normalized = filtered.apply(
            lambda x: (
                x.str.lower().str.replace("\W+", "_", regex=True) if not x.isnull().any() else x
            )
        )
    return normalized


def get_row_match_score(gt_row, pred_row, numerical_threshold, column_config):
    """Compute a match score between a ground truth row and prediction row"""
    data_numerical: np.ndarray = np.abs(
        (only_numeric(gt_row, column_config).values - only_numeric(pred_row, column_config).values)
    )  # / only_numeric(data_row).values
    numerical_score = (data_numerical < numerical_threshold).sum()

    text_score = (
        only_textual(gt_row, column_config) == only_textual(pred_row, column_config)
    ).sum()

    total_score = numerical_score + text_score

    return total_score


def get_alignment_scores(gt_df, pred_df, comparison_columns, column_config):
    """Get a matrix of alignment scores between ground truth and predicted rows."""
    alignment_matrix = np.zeros((len(gt_df), len(pred_df)))
    for i, (_, data_row) in enumerate(gt_df[comparison_columns].iterrows()):
        for j, (_, pred_row) in enumerate(pred_df[comparison_columns].iterrows()):
            alignment_matrix[i, j] = get_row_match_score(
                data_row, pred_row, NUMERICAL_THRESHOLD, column_config
            )

    return alignment_matrix


def align_predictions(pred_df, alignment_matrix):
    """Compute a heuristic alignment between ground truth and predicted rows.

    This function computes a heuristic match score between a predicted row and all ground truth rows
    and greedily assigns each prediction to a ground truth row based on highest score. In case of
    ties, it defaults to sequential alignment.
    """
    candidate_columns = list(range(len(pred_df)))
    assigned_order = []
    for row in alignment_matrix:
        if len(candidate_columns) == 0:
            break

        # sorting trick taken from https://stackoverflow.com/questions/64238462/numpy-descending-stable-arg-sort-of-arrays-of-any-dtype
        # this produces a descending sort that's still stable; otherwise, the initial element order is reversed. This makes the
        # default heuristic for assigning data alignment in the case of a tie the next data.
        ranked_columns = len(row) - 1 - np.argsort(row[::-1], kind="stable")[::-1]

        i = 0
        top_candidate_column = ranked_columns[0]

        while top_candidate_column not in candidate_columns:
            i += 1
            top_candidate_column = ranked_columns[i]
        assigned_order.append(top_candidate_column)

        candidate_columns.remove(top_candidate_column)

    unaligned_rows = pred_df.iloc[candidate_columns] if len(candidate_columns) != 0 else None
    aligned_df = pred_df.iloc[assigned_order]

    if (rows_to_add := (alignment_matrix.shape[0] - len(aligned_df))) > 0:
        none_df = pd.DataFrame({col: [None] * rows_to_add for col in aligned_df})
        aligned_df = pd.concat((aligned_df, none_df))

    return aligned_df, unaligned_rows


def compute_aligned_df_f1(gt_df, aligned_rows, unaligned_rows, present_columns, column_config):
    """Compute F1 score for a single paper.

    gt_df and aligned_rows should be the same shape, and unaligned rows are additional rows that
    have been predicted that have no corresponding ground truth row. Present columns are columns
    that have a value in ground truth, i.e. can be found within the page context.
    """
    numerical_columns = column_config["numerical"]
    textual_columns = column_config["textual"]

    # temporarily commenting out absent columns - it really only makes sense with fully
    # location-annotated data.
    absent_columns = [
        # column
        # for column in gt_df
        # if column not in present_columns and column in numerical_columns + textual_columns
    ]
    tp_numeric, fp_numeric, tn_numeric, fn_numeric = 0, 0, 0, 0
    tp_text, fp_text, tn_text, fn_text = 0, 0, 0, 0
    fp_extra_rows = 0

    source_metrics = {
        (mtype, loc): 0 for mtype in ["tp", "fp", "tn", "fn"] for loc in ANNOTATED_LOCATIONS
    }

    ## METRICS FOR NUMERIC DATA

    # For all present data, compute true and false positives, and false negatives
    for column in set(numerical_columns).intersection(set(present_columns)):
        location = (
            gt_df[column + "_location"].values[0]
            if (column + "_location" in gt_df.columns)
            else "generic"
        )

        num_divergence_pos = np.abs(
            gt_df[column].values - aligned_rows[column].values
        )  # / only_numeric(gt_df[present_columns]).values

        new_tp = (num_divergence_pos < NUMERICAL_THRESHOLD).sum()
        tp_numeric += new_tp
        source_metrics[("tp", location)] += new_tp

        new_fn = np.isnan(num_divergence_pos).sum()
        fn_numeric += new_fn
        source_metrics[("fn", location)] += new_fn

        fp_numeric += num_divergence_pos.shape[0] - (new_tp + new_fn)
        source_metrics[("fp", location)] += num_divergence_pos.shape[0] - (new_tp + new_fn)

    # for absent data, compute true and false negatives, as well as false positives.
    for column in set(numerical_columns).intersection(set(absent_columns)):
        location = (
            gt_df[column + "_location"].values[0]
            if (column + "_location" in gt_df.columns)
            else "generic"
        )  # all absent values should be calculated against a value of 0.
        absent_value = np.zeros_like(gt_df[column].values) - 1
        num_divergence_neg = np.abs(
            absent_value - aligned_rows[column].fillna(-1).fillna(-2)
        ).values

        new_tn = (num_divergence_neg == 0).sum(axis=None)
        tn_numeric += new_tn
        source_metrics[("tn", location)] += new_tn

        fp_numeric += num_divergence_neg.shape[0] - new_tn
        source_metrics[("fp", location)] += num_divergence_neg.shape[0] - new_tn

    ## METRICS FOR TEXT DATA

    # For all present data, compute true and false positives, and false negatives
    for column in set(textual_columns).intersection(set(present_columns)):
        location = (
            gt_df[column + "_location"].values[0]
            if (column + "_location" in gt_df.columns)
            else "generic"
        )

        new_tp = (gt_df[column].values == aligned_rows[column].values).sum(axis=None)
        tp_text += new_tp
        source_metrics[("tp", location)] += new_tp

        new_fn = aligned_rows[column].isnull().sum().sum()
        fn_text += new_fn
        source_metrics[("fn", location)] += new_fn

        fp_text += gt_df[column].shape[0] - (new_tp + new_fn)
        source_metrics[("fp", location)] += gt_df[column].shape[0] - (new_tp + new_fn)

    # for absent data, compute true and false negatives, as well as false positives.
    for column in set(textual_columns).intersection(set(absent_columns)):
        location = (
            gt_df[column + "_location"].values[0]
            if (column + "_location" in gt_df.columns)
            else "generic"
        )
        # all absent textual values should be empty string
        absent_text = np.empty_like(only_textual(gt_df[column], column_config).values, dtype=object)
        absent_text[:] = "-1"

        new_tn = (
            absent_text == only_textual(aligned_rows[column], column_config).fillna("-1").values
        ).sum(axis=None)
        tn_text += new_tn
        source_metrics[("tn", location)] += new_tn

        fp_text += only_textual(gt_df[column], column_config).shape[0] - new_tn
        source_metrics[("fp", location)] += (
            np.prod(only_textual(gt_df[column], column_config).shape) - new_tn
        )

    ## ADJUSTMENTS FOR EXTRA DATA

    if unaligned_rows is not None:
        fp_extra_rows = unaligned_rows[numerical_columns + textual_columns].notnull().sum().sum()
        for col in numerical_columns + textual_columns:
            location = (
                gt_df[column + "_location"].values[0]
                if (column + "_location" in gt_df.columns)
                else "generic"
            )
            source_metrics[("fp", location)] += len(unaligned_rows)

    ## CALCULATING P, R, F1

    tp = tp_numeric + tp_text
    fn = fn_numeric + fn_text
    fp = fp_numeric + fp_text + fp_extra_rows
    tn = tn_numeric + tn_text

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return {
        "tp_numeric": tp_numeric,
        "tp_text": tp_text,
        "tp": tp,
        "fp_numeric": fp_numeric,
        "fp_text": fp_text,
        "fp": fp,
        "fn_numeric": fn_numeric,
        "fn_text": fn_text,
        "fn": fn,
        "tn_numeric": tn_numeric,
        "tn_text": tn_text,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "location_metrics": source_metrics,
    }


def get_results_by_location(location_totals: dict) -> dict[str, dict[str, float]]:
    """Given data per-paper computed per-location, aggregate per-location metrics across papers."""
    location_results = {}
    for location in ANNOTATED_LOCATIONS:
        if location in ["Not on page", "Not Present"]:
            continue
        location_precision = location_totals["tp"][location] / (
            location_totals["tp"][location] + location_totals["fp"][location]
        )

        location_recall = location_totals["tp"][location] / (
            location_totals["tp"][location] + location_totals["fn"][location]
        )

        location_f1 = (2 * location_precision * location_recall) / (
            location_precision + location_recall
        )
        location_results[location] = {
            "precision": location_precision,
            "recall": location_recall,
            "f1": location_f1,
        }
    return location_results


def evaluate_predictions(gt_df, pred_df, column_config):

    if "doi" not in pred_df.columns:
        pred_df["doi"] = pred_df["source"].str.replace(".png|.xml|.html", "").str.replace("_", "/")

    missing_columns = [
        column for column in get_columns_to_predict(column_config) if column not in pred_df.columns
    ]
    if missing_columns:
        raise AssertionError(f"Predictions dataframe missing required columns: {missing_columns}")

    numerical_columns = column_config["numerical"]
    textual_columns = column_config["textual"]

    for column in numerical_columns:
        pred_df[column] = pd.to_numeric(pred_df[column], errors="coerce")

    print(f"Predicted papers: {len(pred_df['doi'].unique())}")

    results_dict = {}

    for doi in tqdm(gt_df["doi"].unique()):
        try:
            ddf = gt_df[gt_df["doi"] == doi]
            pdf = pred_df[pred_df["doi"] == doi][numerical_columns + textual_columns]

            if pdf.empty:
                raise AssertionError(f"DOI {doi} not found in predictions. Skipping.")

            comparison_columns = numerical_columns + textual_columns  # get_comparison_columns(ddf)
            alignment_matrix = get_alignment_scores(ddf, pdf, comparison_columns, column_config)
            aligned_df, unaligned_df = align_predictions(pdf, alignment_matrix)
            result = compute_aligned_df_f1(
                ddf, aligned_df, unaligned_df, comparison_columns, column_config
            )
            results_dict[doi] = result
        except Exception as e:
            print(doi, e)
            continue

    results_df = pd.DataFrame(results_dict).T
    totals = results_df[["tp", "fp", "tn", "fn"]].sum()
    precision = totals["tp"] / (totals["tp"] + totals["fp"])
    recall = totals["tp"] / (totals["tp"] + totals["fn"])
    f1 = (2 * precision * recall) / (precision + recall)

    location_metrics = {k: v["location_metrics"] for k, v in results_dict.items()}

    location_totals = pd.DataFrame.from_dict(location_metrics, orient="index").sum()
    location_results = get_results_by_location(location_totals)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "location_results": location_results,
    }


def evaluate_predictions_wrapper(
    dataset: str,
    predictions_path: str,
) -> None:
    """Evaluate predictions against annotations, and provide precision, recall and f1.

    This script calculates and reports both overall metrics and metrics per-location of the data.

    :param dataset: the name of the dataset to evaluate.
    :param predictions_path: Path to CSV containing predictions. Script will error if there are
    expected columns missing. Expected columns are all columns found in NUMERIC_COLUMNS and
    TEXT_COLUMNS in this file.
    :return:
    """
    if dataset not in metadata_by_dataset:
        raise AssertionError(
            f"Unrecognized dataset {dataset}. Allowable options are: "
            f"{metadata_by_dataset.keys()}"
        )

    gt_df = pd.read_csv(metadata_by_dataset[dataset]["ground_truth_csv"])
    pred_df = pd.read_csv(predictions_path)

    results = evaluate_predictions(
        gt_df, pred_df, metadata_by_dataset[dataset]["evaluation_config"]
    )

    json_results = json.dumps(results, indent=4)

    with open(predictions_path.replace(".csv", "_results.json"), "w") as f:
        f.write(json_results)

    print(json_results)


if __name__ == "__main__":
    fire.Fire(evaluate_predictions_wrapper)
