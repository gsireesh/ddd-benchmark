from data.zeolite.constraints import columns_to_predict, evaluation_column_config

metadata_by_dataset = {
    "zeolite": {
        "data_directory": "data/zeolite",
        "metadata_csv": "data/zeolite/publisher_metadata.csv",
        "ground_truth_csv": "data/zeolite/zeosyn_gold.csv",
        "evaluation_config": evaluation_column_config,
    },
    "aluminum_composition": {
        "data_directory": "data/aluminum",
        "metadata_csv": "data/aluminum/publisher_metadata.csv",
        "ground_truth_csv": "",
    },
}
