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

train_set_dois = ["10.1039/c7dt03751a", "10.1039/c5ce02312b", "10.1016/j.micromeso.2006.10.023"]
