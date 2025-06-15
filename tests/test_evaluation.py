import pytest
import pandas as pd

from evaluate_and_report import evaluate_predictions

golden_df = pd.DataFrame([
    {
        "doi": "10.1007/s11661-016-3375-0",
        "AA": "1100",
        "temper": "H18",
        "Hardness": None,
        "Hardness UNIT": None,
        "Has comp [True / False / nominal]": "TRUE",
        "Cu": 0.08,
        "Mn": 0.05,
    },
    {
        "doi": "10.1007/s11661-016-3375-0",
        "AA": "1100",
        "temper": "H18",
        "Hardness": None,
        "Hardness UNIT": None,
        "Has comp [True / False / nominal]": "TRUE",
        "Cu": 0.08,
        "Mn": 0.05,
    },
    {
        "doi": "10.1007/s11661-016-3375-0",
        "AA": "1100",
        "temper": "H18",
        "Hardness": None,
        "Hardness UNIT": None,
        "Has comp [True / False / nominal]": "TRUE",
        "Cu": 0.08,
        "Mn": 0.05,
    },
    {
        "doi": "10.1016/j.msea.2012.12.046",
        "AA": "1050",
        "temper": None,
        "Hardness": 23.0,
        "Hardness UNIT": "HV",
        "Has comp [True / False / nominal]": "TRUE",
        "Cu": None,
        "Mn": 0.055,
    }
])
column_config = {
    "numerical": ["Hardness", "Cu", "Mn"],
    "textual": ["AA", "temper", "Hardness UNIT", "Has comp [True / False / nominal]"],
}
relevant_dois = set(golden_df["doi"].to_list())


def test_evaluate_predictions_perfect():
    """
    Test the evaluation function with perfect predictions.
    """
    result = evaluate_predictions(
        golden_df.copy(), 
        golden_df.copy(), 
        column_config, 
        relevant_dois
        )
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0

def test_evaluate_predictions_null_predictions():
    """
    Test the evaluation function with null predictions.
    """
    result = evaluate_predictions(
        golden_df.copy(), 
        pd.DataFrame(columns=golden_df.columns), 
        column_config, 
        relevant_dois)
    
    assert result["precision"] is None
    assert result["recall"] is None
    assert result["f1"] is None

def test_evaluate_predictions_absent_columns():
    """
    Test the evaluation function with absent columns in predictions.
    Should raise AssertionError due to missing required columns.
    """
    pred_df = golden_df.copy()
    # delete column Cu from predictions
    pred_df = pred_df.drop(columns=["Cu"])

    with pytest.raises(AssertionError, match=r"Predictions dataframe missing required columns: .*Cu.*"):
        evaluate_predictions(
            golden_df.copy(), 
            pred_df, 
            column_config, 
            relevant_dois
        )

def test_evaluate_predictions_extra_columns():
    """
    Test the evaluation function with extra columns in predictions.
    Should ignore extra columns and not raise.
    """
    pred_df = golden_df.copy()
    pred_df["Extra"] = 123

    result = evaluate_predictions(
        golden_df.copy(),
        pred_df,
        column_config,
        relevant_dois
    )
    assert isinstance(result, dict)
    assert "precision" in result

def test_evaluate_predictions_partial_match():
    """
    Test the evaluation function with partially correct predictions.
    """
    pred_df = golden_df.copy()
    pred_df.loc[0, "AA"] = "WRONG"

    result = evaluate_predictions(
        golden_df.copy(),
        pred_df,
        column_config,
        relevant_dois
    )
    assert 0.0 < result["precision"] < 1.0
    assert result["recall"] == 1.0
    assert 0.0 < result["f1"] < 1.0