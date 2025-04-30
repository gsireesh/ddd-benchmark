import pandas as pd


def only_numeric(data, column_config):
    """Get only numeric "columns" from either a row or series"""
    return data[
        [
            column
            for column in (data.index if isinstance(data, pd.Series) else data)
            if column in column_config["numerical"]
        ]
    ]


def only_textual(data, column_config):
    """Get only textual "columns" from  either a row or series, and normalize the text."""
    textual_columns = column_config["textual"]
    if data.empty:
        return data
    if isinstance(data, pd.Series):
        filtered = data[[column for column in data.index if column in textual_columns]]
        normalized = filtered.str.lower().str.replace("\W+", "_", regex=True)
    else:
        filtered = data[[column for column in data if column in textual_columns]]
        normalized = filtered.apply(
            lambda x: (
                x.str.lower().str.replace("\W+", "_", regex=True) if not x.isnull().any() else x
            )
        )
    return normalized
