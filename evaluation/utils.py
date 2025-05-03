from dataclasses import dataclass
import pandas as pd


@dataclass
class StatsContainer:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def __add__(self, other):
        return StatsContainer(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            tn=self.tn + other.tn,
            fn=self.fn + other.fn,
        )


def get_all_list_columns(column_config):
    list_columns = []
    for config in column_config["aligned_lists"]:
        if isinstance(config["columns"][0], str):
            list_columns.extend(config["columns"])
        elif isinstance(config["columns"][0], tuple):
            for tup in config["columns"]:
                list_columns.extend(tup)
        else:
            raise AssertionError(
                f"Unsupported type found for column definition: {type(config['columns'][0])}. Expected either str or tuple."
            )
    return list_columns


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
