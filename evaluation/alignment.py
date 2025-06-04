import numpy as np
import pandas as pd
from typing import Any

from evaluation.numerical import NUMERICAL_THRESHOLD
from evaluation.utils import only_numeric, only_textual


def get_row_match_score(
        gt_row: pd.Series, 
        pred_row: pd.Series, 
        numerical_threshold:float, 
        column_config: dict[str, list[str]]
    ) -> int:
    """Compute a match score between a ground truth row and prediction row"""
    gt_numerical = only_numeric(gt_row, column_config).fillna(0.0)
    pred_numerical = only_numeric(pred_row, column_config).fillna(0.0)
    data_numerical: np.ndarray = np.abs(gt_numerical - pred_numerical)  # / only_numeric(data_row).values
    numerical_score = (data_numerical < numerical_threshold).sum()

    text_score = (
        only_textual(gt_row, column_config) == only_textual(pred_row, column_config)
    ).sum()

    total_score = numerical_score + text_score

    return total_score


def get_alignment_scores(
        gt_df: pd.DataFrame, 
        pred_df: pd.DataFrame, 
        comparison_columns: list[str], 
        column_config: dict[str,list[str]]
    ) -> np.ndarray:
    """Get a matrix of alignment scores between ground truth and predicted rows."""
    alignment_matrix = np.zeros((len(gt_df), len(pred_df)))
    for i, (_, data_row) in enumerate(gt_df[comparison_columns].iterrows()):
        for j, (_, pred_row) in enumerate(pred_df[comparison_columns].iterrows()):
            alignment_matrix[i, j] = get_row_match_score(
                data_row, pred_row, NUMERICAL_THRESHOLD, column_config
            )

    return alignment_matrix


def align_predictions(pred_df: pd.DataFrame, alignment_matrix: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Compute a heuristic alignment between ground truth and predicted rows.

    This function computes a heuristic match score between a predicted row and all ground truth rows
    and greedily assigns each prediction to a ground truth row based on highest score. In case of
    ties, it defaults to sequential alignment.
    """

    # candidate columns from the alignment matrix
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
