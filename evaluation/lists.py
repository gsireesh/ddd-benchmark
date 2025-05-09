import difflib

import numpy as np
import pandas as pd

from evaluation.numerical import NUMERICAL_THRESHOLD
from evaluation.textual import string_equals
from evaluation.utils import StatsContainer


def string_similarity(a, b):
    if pd.isnull(a) or pd.isnull(b):
        return -1
    return difflib.SequenceMatcher(None, a, b).ratio()


def numeric_similarity(a, b):
    if pd.isnull(a) or pd.isnull(b):
        return -1
    return a - b


def compute_row_alignment(gt_row, aligned_row, row_type):
    alignment_matrix = np.zeros((len(gt_row), len(gt_row)))
    sim_fn = string_similarity if row_type == "textual" else numeric_similarity
    for i, gt in enumerate(gt_row):
        for j, aligned in enumerate(aligned_row):
            alignment_matrix[i][j] = sim_fn(gt, aligned)

    alignment = []
    for i, element in enumerate(gt_row):
        sim_scores = alignment_matrix[i]
        top_assigned_columns = (
            len(sim_scores) - 1 - np.argsort(sim_scores[::-1], kind="stable")[::-1]
        )
        for potential_alignment, value in zip(
            top_assigned_columns, sim_scores[top_assigned_columns]
        ):
            if value == -1:
                alignment.append(None)
                break
            elif potential_alignment in alignment:
                continue
            else:
                alignment.append(potential_alignment)
                break
        else:
            alignment.append(None)

    return alignment


def compute_row_score(gt_row, aligned_row, alignment, row_type, location) -> StatsContainer:
    stats = StatsContainer()

    for i, j in enumerate(alignment):
        # if we don't have an alignment, it's a false negative.
        if j is None:
            stats.record("fn", 1, location)
            continue

        label = gt_row.iloc[i]
        pred = aligned_row.iloc[j]

        # if the label is null
        if pd.isnull(label):
            if pd.isnull(pred):
                stats.record("tn", 1, location)
            else:
                stats.record("fp", 1, location)
            continue

        # label not null
        if pd.isnull(pred):
            stats.record("fn", 1, location)
            continue

        assert not pd.isnull(label) and not pd.isnull(pred)

        if row_type == "textual":
            if string_equals(label, pred):
                stats.record("tp", 1, location)
            else:
                stats.record("fp", 1, location)

        elif row_type == "numerical":
            if label - pred <= NUMERICAL_THRESHOLD:
                stats.record("tp", 1, location)
            else:
                stats.record("fp", 1, location)

    return stats


def evaluate_list_columns(gt_df, aligned_rows, column_config, present_columns, absent_columns):
    stats = StatsContainer()

    for list_config in column_config["aligned_lists"]:
        list_columns = list_config["columns"]
        list_type = list_config["type"]

        if isinstance(list_columns[0], str):
            focus_columns = list_columns
            focus_type = list_type
            focus_location = "generic"
        elif isinstance(list_columns[0], tuple):
            focus_columns = [tup[0] for tup in list_columns]
            focus_type = list_type[0]
            focus_location = "generic"
        else:
            raise AssertionError(f"Disallowed type of columns type: {type(list_columns[0])}")

        for (gt_row_index, gt_row), (aligned_row_index, aligned_row) in zip(
            gt_df[focus_columns].iterrows(), aligned_rows[focus_columns].iterrows()
        ):

            row_alignment = compute_row_alignment(gt_row, aligned_row, focus_type)
            focus_row_stats = compute_row_score(
                gt_row, aligned_row, row_alignment, focus_type, location=focus_location
            )
            stats += focus_row_stats

            # compute stats for parallel columns (e.g. OSDA quantity being parallel aligned to
            # OSDA name)
            if isinstance(list_columns[0], tuple):
                for i in range(1, len(list_columns[0])):
                    parallel_columns = [tup[i] for tup in list_columns]
                    parallel_gt_row = gt_df[parallel_columns].loc[gt_row_index]
                    parallel_pred_row = aligned_rows[parallel_columns].loc[aligned_row_index]
                    aligned_row_stats = compute_row_score(
                        parallel_gt_row,
                        parallel_pred_row,
                        row_alignment,
                        list_type[i],
                        focus_location,
                    )
                    stats += aligned_row_stats

    return stats
