import numpy as np
import pandas as pd
import pytest

from evaluation.numerical import evaluate_numerical_columns
from evaluation.utils import StatsContainer

def make_stats_dict(stats: StatsContainer):
    # Helper to convert StatsContainer to a dict for easy assertions
    result = {}
    for inst in stats.instances:
        result.setdefault(inst.stat_type, {})
        result[inst.stat_type][inst.location] = result[inst.stat_type].get(inst.location, 0) + inst.number
    return result

numerical_test_cases = [
    # GT, Pred, TP, FP, FN, TN, Test_ID
    pytest.param(123, 123,          "TP",   id="TP: 123 vs 123"),
    pytest.param(123, 0.23,         "FP",   id="FP: 123 vs 0.23"),
    pytest.param(123, 0.023,        "FP",   id="FP: 123 vs 0.023"),
    pytest.param(123, 0.0,          "FP",   id="FP: 123 vs 0.0"),
    pytest.param(123, np.nan,       "FN",   id="FN: 123 vs NaN"),
    # pytest.param(123, pd.NA,        "FN",   id="FN: 123 vs pd.NA"),

    pytest.param(0.23, 123,         "FP",   id="FP: 0.23 vs 123"),
    pytest.param(0.23, 0.23,        "TP",   id="TP: 0.23 vs 0.23"),
    pytest.param(0.23, 0.023,       "FP",   id="FP: 0.23 vs 0.023"),
    pytest.param(0.23, 0.0,         "FP",   id="FP: 0.23 vs 0.0"),
    pytest.param(0.23, np.nan,      "FN",   id="FN: 0.23 vs NaN"),
    # pytest.param(0.23, pd.NA,       "FN",   id="FN: 0.23 vs pd.NA"),

    pytest.param(0.023, 123,        "FP",   id="FP: 0.023 vs 123"),
    pytest.param(0.023, 0.23,       "FP",   id="FP: 0.023 vs 0.23"),
    pytest.param(0.023, 0.023,      "TP",   id="TP: 0.023 vs 0.023"),
    pytest.param(0.023, 0.0,        "TP",   id="TP: 0.023 vs 0.0 (close enough)"),
    pytest.param(0.023, np.nan,     "FN",   id="FN: 0.023 vs NaN"),
    # pytest.param(0.023, pd.NA,      "FN",   id="FN: 0.023 vs pd.NA"),

    pytest.param(0.0, 123,          "FP",   id="FP: 0.0 vs 123"),
    pytest.param(0.0, 0.23,         "FP",   id="FP: 0.0 vs 0.23"),
    pytest.param(0.0, 0.023,        "TP",   id="TP: 0.0 vs 0.023 (close enough)"),
    pytest.param(0.0, 0.0,          "TP",   id="TP: 0.0 vs 0.0"),
    pytest.param(0.0, np.nan,       "FN",   id="FN: 0.0 vs NaN"),
    # pytest.param(0.0, pd.NA,        "FN",   id="FN: 0.0 vs pd.NA"),

    pytest.param(np.nan, 123,       "FP",   id="FP: NaN vs 123"),
    pytest.param(np.nan, 0.23,      "FP",   id="FP: NaN vs 0.23"),
    pytest.param(np.nan, 0.023,     "FP",   id="FP: NaN vs 0.023"),
    pytest.param(np.nan, 0.0,       "FP",   id="FP: NaN vs 0.0"),
    pytest.param(np.nan, np.nan,    "TN",   id="TN: NaN vs NaN"),
    # pytest.param(np.nan, pd.NA,     "TN",   id="TN: NaN vs pd.NA"),

    # pytest.param(pd.NA, 123,        "FP",   id="FP: pd.NA vs 123"),
    # pytest.param(pd.NA, 0.23,       "FP",   id="FP: pd.NA vs 0.23"),
    # pytest.param(pd.NA, 0.023,      "FP",   id="FP: pd.NA vs 0.023"),
    # pytest.param(pd.NA, 0.0,        "FP",   id="FP: pd.NA vs 0.0"),
    # pytest.param(pd.NA, np.nan,     "TN",   id="TN: pd.NA vs NaN"),
    # pytest.param(pd.NA, pd.NA,      "TN",   id="TN: pd.NA vs pd.NA"),
]

@pytest.mark.parametrize("gt_val, pred_val, confusion_matrix", numerical_test_cases)
def test_numerical_comparisons(gt_val, pred_val, confusion_matrix):
    """
    Tests the evaluate_numerical_columns function against a matrix of ground truth
    and prediction values.
    """    
    stats = evaluate_numerical_columns(
        pd.DataFrame({'a': [gt_val]}), 
        pd.DataFrame({'a': [pred_val]}), 
        ["a"], 
        ["a"], 
        []
        )
    d = make_stats_dict(stats)

    # Assertions
    assert d.get("tp", {}).get("generic", 0) == int(confusion_matrix == "TP")
    assert d.get("fp", {}).get("generic", 0) == int(confusion_matrix == "FP")
    assert d.get("fn", {}).get("generic", 0) == int(confusion_matrix == "FN")
    assert d.get("tn", {}).get("generic", 0) == int(confusion_matrix == "TN")

def test_location_key():
    stats = evaluate_numerical_columns(
        pd.DataFrame({"a": [1.0], "a_location": ["loc1"]}), 
        pd.DataFrame({"a": [1.0]}), 
        ["a"], 
        ["a"], 
        []
        )
    d = make_stats_dict(stats)
    assert d["tp"]["loc1"] == 1