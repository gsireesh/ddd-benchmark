from dataclasses import asdict, dataclass, field
import pandas as pd


@dataclass
class StatInstance:
    stat_type: str
    number: int
    location: str
    doi: str | None


@dataclass
class StatsContainer:
    instances: list[StatInstance] = field(default_factory=list)

    def record(self, stat_type: str, number: int, location: str, doi: str = None):
        self.instances.append(StatInstance(stat_type, number, location, doi))

    def broadcast_doi(self, doi):
        self.instances = [
            StatInstance(inst.stat_type, inst.number, inst.location, doi) for inst in self.instances
        ]

    def to_dataframe(self):
        return pd.DataFrame([asdict(inst) for inst in self.instances])

    def __add__(self, other):
        return StatsContainer(self.instances + other.instances)


def get_all_list_columns(column_config):
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
        normalized = filtered.mask(
            filtered.notnull(),
            filtered.str.lower().str.replace("\W+", "_", regex=True),
        )
    else:
        filtered = data[[column for column in data if column in textual_columns]]
        normalized = filtered.apply(
            lambda x: (
                x.str.lower().str.replace("\W+", "_", regex=True) if not x.isnull().any() else x
            )
        )
    return normalized
