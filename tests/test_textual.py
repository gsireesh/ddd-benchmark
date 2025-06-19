import numpy as np
import pandas as pd
import pytest

from evaluation.textual import evaluate_textual_columns
from evaluation.utils import StatsContainer

def make_stats_dict(stats: StatsContainer):
    # Helper to convert StatsContainer to a dict for easy assertions
    result = {}
    for inst in stats.instances:
        result.setdefault(inst.stat_type, {})
        result[inst.stat_type][inst.location] = result[inst.stat_type].get(inst.location, 0) + inst.number
    return result

textual_test_cases = [
    # GT, Pred, Confusion, Test_ID
    pytest.param("A", "A",        "TP", id='TP: "A" vs "A"'),
    pytest.param("A", "true",     "FP", id='FP: "A" vs "true"'),
    pytest.param("A", "",         "FN", id='FN: "A" vs ""'),
    pytest.param("A", " ",        "FN", id='FN: "A" vs " "'),
    pytest.param("A", np.nan,     "FN", id='FN: "A" vs np.nan'),
    pytest.param("A", pd.NA,      "FN", id='FN: "A" vs pd.NA'),

    pytest.param("true", "A",     "FP", id='FP: "true" vs "A"'),
    pytest.param("true", "true",  "TP", id='TP: "true" vs "true"'),
    pytest.param("true", "",      "FN", id='FN: "true" vs ""'),
    pytest.param("true", " ",     "FN", id='FN: "true" vs " "'),
    pytest.param("true", np.nan,  "FN", id='FN: "true" vs np.nan'),
    pytest.param("true", pd.NA,   "FN", id='FN: "true" vs pd.NA'),

    pytest.param("", "A",         "FP", id='FP: "" vs "A"'),
    pytest.param("", "true",      "FP", id='FP: "" vs "true"'),
    pytest.param("", "",          "TN", id='TN: "" vs ""'),
    pytest.param("", " ",         "TN", id='TN: "" vs " "'),
    pytest.param("", np.nan,      "TN", id='TN: "" vs np.nan'),
    pytest.param("", pd.NA,       "TN", id='TN: "" vs pd.NA'),

    pytest.param(" ", "A",        "FP", id='FP: " " vs "A"'),
    pytest.param(" ", "true",     "FP", id='FP: " " vs "true"'),
    pytest.param(" ", "",         "TN", id='TN: " " vs ""'),
    pytest.param(" ", " ",        "TN", id='TN: " " vs " "'),
    pytest.param(" ", np.nan,     "TN", id='TN: " " vs np.nan'),
    pytest.param(" ", pd.NA,      "TN", id='TN: " " vs pd.NA'),

    pytest.param(np.nan, "A",     "FP", id='FP: np.nan vs "A"'),
    pytest.param(np.nan, "true",  "FP", id='FP: np.nan vs "true"'),
    pytest.param(np.nan, "",      "TN", id='TN: np.nan vs ""'),
    pytest.param(np.nan, " ",     "TN", id='TN: np.nan vs " "'),
    pytest.param(np.nan, np.nan,  "TN", id='TN: np.nan vs np.nan'),
    pytest.param(np.nan, pd.NA,   "TN", id='TN: np.nan vs pd.NA'),

    pytest.param(pd.NA, "A",      "FP", id='FP: pd.NA vs "A"'),
    pytest.param(pd.NA, "true",   "FP", id='FP: pd.NA vs "true"'),
    pytest.param(pd.NA, "",       "TN", id='TN: pd.NA vs ""'),
    pytest.param(pd.NA, " ",      "TN", id='TN: pd.NA vs " "'),
    pytest.param(pd.NA, np.nan,   "TN", id='TN: pd.NA vs np.nan'),
    pytest.param(pd.NA, pd.NA,    "TN", id='TN: pd.NA vs pd.NA'),
]

@pytest.mark.parametrize("gt_val, pred_val, confusion_matrix", textual_test_cases)
def test_textual_comparisons(gt_val, pred_val, confusion_matrix):
    stats = evaluate_textual_columns(
        pd.DataFrame({'a': [gt_val]}),
        pd.DataFrame({'a': [pred_val]}),
        ["a"],
        ["a"],
        []
    )
    d = make_stats_dict(stats)
    # print(d)
    assert d.get("tp", {}).get("generic", 0) == int(confusion_matrix == "TP"), f"{confusion_matrix}(TP)"
    assert d.get("fp", {}).get("generic", 0) == int(confusion_matrix == "FP"), f"{confusion_matrix}(FP)"
    assert d.get("fn", {}).get("generic", 0) == int(confusion_matrix == "FN"), f"{confusion_matrix}(FN)"
    assert d.get("tn", {}).get("generic", 0) == int(confusion_matrix == "TN"), f"{confusion_matrix}(TN)"

def test_location_key():
    stats = evaluate_textual_columns(
        pd.DataFrame({"a": ["A"], "a_location": ["loc1"]}),
        pd.DataFrame({"a": ["A"]}),
        ["a"],
        ["a"],
        []
    )
    d = make_stats_dict(stats)
    assert d["tp"]["loc1"] == 1