import numpy as np
from evaluation.utils import only_textual


def evaluate_textual_columns(
    gt_df, aligned_rows, textual_columns, present_columns, absent_columns, source_metrics
):
    tp_text, fp_text, tn_text, fn_text = 0, 0, 0, 0

    # For all present data, compute true and false positives, and false negatives
    for column in set(textual_columns).intersection(set(present_columns)):
        location = (
            gt_df[column + "_location"].values[0]
            if (column + "_location" in gt_df.columns)
            else "generic"
        )

        nonnull_filter = gt_df[column].notnull()

        new_tp = (
            gt_df[column][nonnull_filter].values == aligned_rows[column][nonnull_filter].values
        ).sum(axis=None)
        tp_text += new_tp
        source_metrics[("tp", location)] += new_tp

        new_fn = aligned_rows[column][nonnull_filter].isnull().sum().sum()
        fn_text += new_fn
        source_metrics[("fn", location)] += new_fn

        should_be_null_fp = (aligned_rows[column][~nonnull_filter].notnull()).sum()
        wrong_value_fp = (
            gt_df[column][nonnull_filter].values != aligned_rows[column][nonnull_filter].values
        ).sum(axis=None)
        fp_text += should_be_null_fp + wrong_value_fp
        source_metrics[("fp", location)] += should_be_null_fp + wrong_value_fp

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
        tn_text += new_tn
        source_metrics[("tn", location)] += new_tn

        fp_text += gt_df[column].shape[0] - new_tn
        source_metrics[("fp", location)] += np.prod(gt_df[column].shape) - new_tn

    return tp_text, fp_text, tn_text, fn_text
