import pandas as pd

from evaluation.numerical import NUMERICAL_THRESHOLD
from evaluation.textual import string_equals
from evaluation.utils import StatsContainer


def compute_row_alignment(gt_row, aligned_row):
    alignment = []
    for element in gt_row:
        try:
            alignment.append(aligned_row.tolist().index(element))
        except ValueError:
            alignment.append(None)
    return alignment


def compute_row_score(gt_row, aligned_row, alignment, row_type) -> StatsContainer:
    stats = StatsContainer()

    for i, j in enumerate(alignment):
        # if we don't have an alignment, it's a false negative.
        if j is None:
            stats.fn += 1
            continue

        label = gt_row.iloc[i]
        pred = aligned_row.iloc[j]

        # if the label is null
        if pd.isnull(label):
            if pd.isnull(pred):
                stats.tn += 1
            else:
                stats.fp += 1
            continue

        # label not null
        if pd.isnull(pred):
            stats.fn += 1
            continue

        assert not pd.isnull(label) and not pd.isnull(pred)

        if row_type == "textual":
            if string_equals(label, pred):
                stats.tp += 1
            else:
                stats.fp += 1

        elif row_type == "numerical":
            if label - pred <= NUMERICAL_THRESHOLD:
                stats.tp += 1
            else:
                stats.fp += 1

    return stats


def evaluate_list_columns(
    gt_df, aligned_rows, column_config, present_columns, absent_columns, source_metrics
):
    stats = StatsContainer()

    for list_config in column_config["aligned_lists"]:
        list_columns = list_config["columns"]
        list_type = list_config["type"]

        if isinstance(list_columns[0], str):
            focus_columns = list_columns
            focus_type = list_type
        elif isinstance(list_columns[0], tuple):
            focus_columns = [tup[0] for tup in list_columns]
            focus_type = list_type[0]
        else:
            raise AssertionError(f"Disallowed type of columns type: {type(list_columns[0])}")

        for (gt_row_index, gt_row), (aligned_row_index, aligned_row) in zip(
            gt_df[focus_columns].iterrows(), aligned_rows[focus_columns].iterrows()
        ):
            row_alignment = compute_row_alignment(gt_row, aligned_row)
            focus_row_stats = compute_row_score(gt_row, aligned_row, row_alignment, focus_type)
            stats += focus_row_stats

            # compute stats for parallel columns (e.g. OSDA quantity being parallel aligned to
            # OSDA name)
            if isinstance(list_columns[0], tuple):
                for i in range(1, len(list_columns[0])):
                    parallel_columns = [tup[i] for tup in list_columns]
                    parallel_gt_row = gt_df[parallel_columns].loc[gt_row_index]
                    parallel_pred_row = aligned_rows[parallel_columns].loc[aligned_row_index]
                    aligned_row_stats = compute_row_score(
                        parallel_gt_row, parallel_pred_row, row_alignment, list_type[i]
                    )
                    stats += aligned_row_stats

    return stats.tp, stats.fp, stats.tn, stats.fn
