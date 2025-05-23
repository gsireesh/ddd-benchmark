import json

import fire
import pandas as pd
from tqdm.auto import tqdm

from data.dataset_metadata import metadata_by_dataset
from evaluation import (
    StatsContainer,
    align_predictions,
    evaluate_list_columns,
    evaluate_numerical_columns,
    evaluate_textual_columns,
    get_alignment_scores,
    get_all_list_columns,
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
        column_config["numerical"] + column_config["textual"] + get_all_list_columns(column_config)
    )


# only makes sense for fully location-annotated data
def get_comparison_columns(data_df, columns_to_predict):
    comparison_columns = []
    if (
        columns_to_predict[0] + "_location" not in columns_to_predict
        or columns_to_predict[columns_to_predict[0]].isnull().any()
    ):
        return columns_to_predict
    for column in columns_to_predict:
        if data_df[column + "_location"].iloc[0] not in ["Not Present", "Not on page"]:
            comparison_columns.append(column)
    return comparison_columns


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

    paper_stats = StatsContainer()

    ## METRICS FOR NUMERIC DATA
    numerical_stats = evaluate_numerical_columns(
        gt_df, aligned_rows, numerical_columns, present_columns, absent_columns
    )
    ## METRICS FOR TEXT DATA
    textual_stats = evaluate_textual_columns(
        gt_df, aligned_rows, textual_columns, present_columns, absent_columns
    )

    ## METRICS FOR LIST FIELDS
    # list_stats = evaluate_list_columns(
    #     gt_df,
    #     aligned_rows,
    #     column_config,
    #     present_columns,
    #     absent_columns,
    # )

    ## ADJUSTMENTS FOR EXTRA DATA

    if unaligned_rows is not None:
        fp_extra_rows = unaligned_rows[numerical_columns + textual_columns].notnull().sum().sum()
        for column in numerical_columns + textual_columns:
            location = (
                gt_df[column + "_location"].values[0]
                if (column + "_location" in gt_df.columns)
                else "generic"
            )
            paper_stats.record("fp", fp_extra_rows, location)

    ## CALCULATING P, R, F1

    paper_stats += numerical_stats + textual_stats  # + list_stats

    return paper_stats


def get_results_by_location(dataset_stats: StatsContainer) -> dict[str, dict[str, float]]:
    """Given data per-paper computed per-location, aggregate per-location metrics across papers."""
    scores_by_location = {}
    result_df = dataset_stats.to_dataframe()
    by_location = result_df.groupby("location")
    for location, index in by_location.groups.items():
        location_df = result_df.loc[index]
        scores_by_location[location] = calculate_prf_from_df(location_df)
    return scores_by_location


def calculate_prf_from_df(df: pd.DataFrame) -> dict[str, float]:
    totals = df.groupby("stat_type")["number"].sum()
    precision = totals.loc["tp"] / (totals.loc["tp"] + totals.loc["fp"])
    recall = totals.loc["tp"] / (totals.loc["tp"] + totals.loc["fn"])
    f1 = (2 * precision * recall) / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_predictions(gt_df, pred_df, column_config):

    if "doi" not in pred_df.columns:
        pred_df["doi"] = pred_df["source"].str.replace(".png|.xml|.html", "").str.replace("_", "/")

    columns_to_predict = get_columns_to_predict(column_config)
    missing_columns = [column for column in columns_to_predict if column not in pred_df.columns]
    if missing_columns:
        raise AssertionError(f"Predictions dataframe missing required columns: {missing_columns}")

    for column in column_config["numerical"]:
        pred_df[column] = pd.to_numeric(pred_df[column], errors="coerce")

    dataset_stats = StatsContainer()

    for doi in tqdm(gt_df["doi"].unique()):
        try:
            ddf = gt_df[gt_df["doi"] == doi]
            pdf = pred_df[pred_df["doi"] == doi][columns_to_predict]

            if pdf.empty:
                raise AssertionError(f"DOI {doi} not found in predictions. Skipping.")

            comparison_columns = get_comparison_columns(ddf, columns_to_predict)
            alignment_matrix = get_alignment_scores(ddf, pdf, comparison_columns, column_config)
            aligned_df, unaligned_df = align_predictions(pdf, alignment_matrix)
            paper_stats = compute_aligned_df_f1(
                ddf, aligned_df, unaligned_df, comparison_columns, column_config
            )
            paper_stats.broadcast_doi(doi)
            dataset_stats += paper_stats
        except Exception as e:
            print(doi, e)
            # raise e
            continue

    results_df = dataset_stats.to_dataframe()
    global_results = calculate_prf_from_df(results_df)
    location_metrics = get_results_by_location(dataset_stats)

    return {**global_results, "scores_by_location": location_metrics}


def evaluate_predictions_wrapper(
    dataset: str,
    modality: str,
    predictions_path: str,
) -> None:
    """Evaluate predictions against annotations, and provide precision, recall and f1.

    This script calculates and reports both overall metrics and metrics per-location of the data.

    :param dataset: the name of the dataset to evaluate.
    :param modality: The modality in which to evaluate predictions (XML or PDF)
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

    metadata = metadata_by_dataset[dataset]

    if modality.lower() not in {"xml", "pdf"}:
        raise AssertionError("Unrecognized modality {modality}. Expected one of xml, pdf")

    publisher_meta = pd.read_csv(metadata["metadata_csv"])

    relevant_dois = set(
        publisher_meta[publisher_meta["included_in_dataset"] & publisher_meta[modality.lower()]][
            "doi"
        ].tolist()
    )

    print(f"Evaluating against {len(relevant_dois)} papers.")

    gt_df = pd.read_csv(metadata["ground_truth_csv"])
    gt_df = gt_df[gt_df["doi"].isin(relevant_dois)]

    pred_df = pd.read_csv(predictions_path)
    all_pred_dois = pred_df["doi"].unique()
    predicted_relevant_dois = pred_df[pred_df["doi"].isin(relevant_dois)]["doi"].unique()
    print(f"Predictions from {len(all_pred_dois)} papers; ")

    results = evaluate_predictions(gt_df, pred_df, metadata["evaluation_config"])

    json_results = json.dumps(results, indent=4)

    with open(predictions_path.replace(".csv", "_results.json"), "w") as f:
        f.write(json_results)

    print(json_results)


if __name__ == "__main__":
    fire.Fire(evaluate_predictions_wrapper)
