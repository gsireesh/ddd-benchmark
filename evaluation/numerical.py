import numpy as np

NUMERICAL_THRESHOLD = 0.1


def evaluate_numerical_columns(
    gt_df, aligned_rows, numerical_columns, present_columns, absent_columns, source_metrics
):
    # For all present data, compute true and false positives, and false negatives
    tp_numeric, fp_numeric, tn_numeric, fn_numeric = 0, 0, 0, 0

    for column in set(numerical_columns).intersection(set(present_columns)):
        location = (
            gt_df[column + "_location"].values[0]
            if (column + "_location" in gt_df.columns)
            else "generic"
        )

        nonnull_filter = gt_df[column].notnull().values

        num_divergence_pos = np.abs(
            gt_df[column][nonnull_filter].values - aligned_rows[column][nonnull_filter].values
        )  # / only_numeric(gt_df[present_columns]).values

        new_tp = (num_divergence_pos <= NUMERICAL_THRESHOLD).sum()
        tp_numeric += new_tp
        source_metrics[("tp", location)] += new_tp

        new_fn = np.isnan(num_divergence_pos).sum()
        fn_numeric += new_fn
        source_metrics[("fn", location)] += new_fn

        should_be_null_fp = (aligned_rows[column][~nonnull_filter].notnull()).sum()
        wrong_value_fp = (num_divergence_pos > NUMERICAL_THRESHOLD).sum()

        tn_numeric += aligned_rows[~nonnull_filter].isnull().sum()
        fp_numeric += should_be_null_fp + wrong_value_fp
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

    return tp_numeric, fp_numeric, tn_numeric, fn_numeric
