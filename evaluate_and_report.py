import json

import fire
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from data.dataset_metadata import metadata_by_dataset
from evaluation import (
    align_predictions,
    evaluate_numerical_columns,
    evaluate_textual_columns,
    get_alignment_scores,
    only_textual,
    only_numeric,
)


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

    fp_extra_rows = 0

    source_metrics = {
        (mtype, loc): 0 for mtype in ["tp", "fp", "tn", "fn"] for loc in ANNOTATED_LOCATIONS
    }

    ## METRICS FOR NUMERIC DATA
    tp_numeric, fp_numeric, tn_numeric, fn_numeric = evaluate_numerical_columns(
        gt_df, aligned_rows, numerical_columns, present_columns, absent_columns, source_metrics
    )
    ## METRICS FOR TEXT DATA
    tp_text, fp_text, tn_text, fn_text = evaluate_textual_columns(
        gt_df, aligned_rows, textual_columns, present_columns, absent_columns, source_metrics
    )

    ## ADJUSTMENTS FOR EXTRA DATA

    if unaligned_rows is not None:
        fp_extra_rows = unaligned_rows[numerical_columns + textual_columns].notnull().sum().sum()
        for column in numerical_columns + textual_columns:
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
