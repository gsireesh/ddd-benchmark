meta_columns = ["doi", "From?", "article_type", "journal", "publisher"]

evaluation_column_config = {
    "numerical": ["Si", "Ge", "Al", "OH", "H2O", "HF", "SDA", "B", "Time", "Temp"],
    "textual": ["SDA Type", "Extracted"],
    "aligned_lists": [],
}

columns_to_predict = [
    "Si",
    "Ge",
    "Al",
    "OH",
    "H2O",
    "HF",
    "SDA",
    "B",
    "Time",
    "Temp",
    "SDA Type",
    "Extracted",
]

train_set_dois = [
    "10.1016/j.micromeso.2006.10.023",
    "10.1016/j.solidstatesciences.2007.08.002",
    "10.1007/s10934-015-0051-5",
    "10.1002/anie.200461911",
    "10.1007/s11244-013-0170-7",
]
