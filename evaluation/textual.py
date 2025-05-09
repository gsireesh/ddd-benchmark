import numpy as np

from evaluation.utils import StatsContainer


def string_equals(string_a, string_b):
    return string_a == string_b


def evaluate_textual_columns(gt_df, aligned_rows, textual_columns, present_columns, absent_columns):
    stats = StatsContainer()

    # For all present data, compute true and false positives, and false negatives
    for column in set(textual_columns).intersection(set(present_columns)):
        location = (
            gt_df[column + "_location"].values[0]
            if (column + "_location" in gt_df.columns)
            else "generic"
        )

        nonnull_filter = gt_df[column].notnull().values

        new_tp = (
            gt_df[column][nonnull_filter].values == aligned_rows[column][nonnull_filter].values
        ).sum(axis=None)
        stats.record("tp", new_tp, location)

        new_fn = aligned_rows[column][nonnull_filter].isnull().sum().sum()
        stats.record("fn", new_fn, location)

        should_be_null_fp = (aligned_rows[column][~nonnull_filter].notnull()).sum()
        wrong_value_fp = (
            gt_df[column][nonnull_filter].values != aligned_rows[column][nonnull_filter].values
        ).sum(axis=None)
        stats.record("fp", should_be_null_fp + wrong_value_fp, location)

    # for absent data, compute true and false negatives, as well as false positives.
    for column in set(textual_columns).intersection(set(absent_columns)):
        location = (
            gt_df[column + "_location"].values[0]
            if (column + "_location" in gt_df.columns)
            else "generic"
        )

        # all absent textual values should be empty string
        absent_text = np.empty_like(gt_df[column].values, dtype=object)
        absent_text[:] = "-1"

        new_tn = (absent_text == aligned_rows[column].fillna("-1").values).sum(axis=None)
        stats.record("tn", new_tn, location)

        stats.record("fp", "gt_df[column].shape[0] - new_tn")

    return stats
