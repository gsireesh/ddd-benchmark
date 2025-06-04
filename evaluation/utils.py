from dataclasses import asdict, dataclass, field
import pandas as pd
from typing import Any


@dataclass
class StatInstance:
    stat_type: str
    number: int
    location: str
    doi: str | None


@dataclass
class StatsContainer:
    instances: list[StatInstance] = field(default_factory=list)

    def record(self, stat_type: str, number: int, location: str, doi: str | None = None):
        self.instances.append(StatInstance(stat_type, number, location, doi))

    def broadcast_doi(self, doi: str | None):
        self.instances = [
            StatInstance(inst.stat_type, inst.number, inst.location, doi) for inst in self.instances
        ]

    def to_dataframe(self):
        return pd.DataFrame([asdict(inst) for inst in self.instances])

    def __add__(self, other: "StatsContainer") -> "StatsContainer":
        return StatsContainer(self.instances + other.instances)


def get_all_list_columns(column_config: dict[str, list[str]]) -> list[str]:
    """
    Get list columns from the column configuration, config["aligned_lists"][_]["columns"].
    If the columns are defined as tuples, they will be flattened into a single list.
    """
    list_columns = []
    if "aligned_lists" not in column_config:
        return list_columns
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


def only_numeric(data: pd.Series, column_config: dict[str, list[str]]) -> pd.Series:
    """Get only numeric "columns" from series"""
    return data[
        [
            column
            for column in (data.index if isinstance(data, pd.Series) else data)
            if column in column_config["numerical"]
        ]
    ]


def only_textual(data: pd.Series, column_config: dict[str, list[str]]) -> pd.Series:
    """Get only textual "columns" from series, and normalize the text."""
    textual_columns = column_config["textual"]
    if data.empty:
        return data
    if isinstance(data, pd.Series):
        filtered = data[[column for column in data.index if column in textual_columns]]
        normalized = filtered.mask(
            filtered.notnull(),
            filtered.str.lower().str.replace(r"\W+", "_", regex=True),
        )
    else:
        filtered = data[[column for column in data if column in textual_columns]]
        normalized = filtered.apply(
            lambda x: (
                x.str.lower().str.replace(r"\W+", "_", regex=True) if not x.isnull().any() else x
            )
        )
    return normalized
