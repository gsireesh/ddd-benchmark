import numpy as np
import pandas as pd

from evaluation.utils import StatsContainer

NUMERICAL_THRESHOLD = 0.1


def evaluate_numerical_columns(
    gt_df: pd.DataFrame,
    aligned_rows: pd.DataFrame,
    numerical_columns: list[str], 
    present_columns: list[str], 
    absent_columns: list[str]
) -> StatsContainer:
    # For all present data, compute true and false positives, and false negatives
    stats = StatsContainer()

    for column in set(numerical_columns).intersection(set(present_columns)):
        location = (
            gt_df[column + "_location"].values[0]
            if (column + "_location" in gt_df.columns) and len(gt_df) > 0
            else "generic"
        )

        nonnull_filter = gt_df[column].notnull().to_numpy()

        num_divergence_pos = np.abs(
            gt_df[column].values[nonnull_filter] - aligned_rows[column].values[nonnull_filter]
        )  # / only_numeric(gt_df[present_columns]).values

        new_tp = (num_divergence_pos <= NUMERICAL_THRESHOLD).sum()
        stats.record("tp", new_tp, location)

        new_fn = np.isnan(num_divergence_pos).sum()
        stats.record("fn", new_fn, location)

        should_be_null_fp = (aligned_rows[column][~nonnull_filter].notnull()).sum()
        wrong_value_fp = (num_divergence_pos > NUMERICAL_THRESHOLD).sum()

        stats.record("tn", aligned_rows[column][~nonnull_filter].isnull().sum(), location)
        stats.record("fp", should_be_null_fp + wrong_value_fp, location)

    # for absent data, compute true and false negatives, as well as false positives.
    for column in set(numerical_columns).intersection(set(absent_columns)):
        location = (
            gt_df[column + "_location"].values[0]
            if (column + "_location" in gt_df.columns)
            else "generic"
        )  # all absent values should be calculated against a value of 0.
        absent_value = np.zeros_like(gt_df[column].values) - 1
        num_divergence_neg = np.abs(absent_value - aligned_rows[column].fillna(-1.0).values)

        new_tn = (num_divergence_neg == 0).sum(axis=None)
        stats.record("tn", new_tn, location)

        stats.record("fp", num_divergence_neg.shape[0] - new_tn, location)

    return stats
