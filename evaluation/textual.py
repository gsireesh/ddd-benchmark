import re

import numpy as np
import pandas as pd

from evaluation.utils import StatsContainer


def string_equals(string_a, string_b):
    return string_a == string_b


def normalize(string: str) -> str | None:
    if pd.isnull(string):
        return None
    mod = string
    mod = mod.strip()
    mod = mod.lower()
    mod = re.sub(r"\W+", "-", mod)
    return mod


def split_and_normalize(string: str) -> set[str]:
    if pd.isnull(string):
        return set()
    split_and_normed = {normalize(s) for s in re.split(",", string)}
    return split_and_normed


def evaluate_textual_columns(gt_df, aligned_rows, textual_columns, present_columns, absent_columns):
    stats = StatsContainer()

    for column in set(textual_columns).intersection(set(present_columns)):
        location = (
            gt_df[column + "_location"].values[0]
            if (column + "_location" in gt_df.columns)
            else "generic"
        )

        # # For all present data, compute true and false positives, and false negatives
        for gt_value, pred_value in zip(gt_df[column].values, aligned_rows[column].values):
            gt_normed_set = split_and_normalize(gt_value)
            pred_normed_set = split_and_normalize(pred_value)

            stats.record("tp", len(gt_normed_set.intersection(pred_normed_set)), location)
            stats.record("fp", len(pred_normed_set - gt_normed_set), location)
            stats.record("fn", len(gt_normed_set) if not pred_normed_set else 0, location)
            stats.record("tn", len(gt_normed_set) + len(pred_normed_set) == 0, location)

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

        stats.record("fp", gt_df[column].shape[0] - new_tn, location)

    return stats
