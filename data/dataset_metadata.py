from data.aluminum.constraints import evaluation_column_config as al_column_config
from data.zeolite.constraints import (
    evaluation_column_config as zeolite_column_config,
)


metadata_by_dataset = {
    "zeolite": {
        "data_directory": "data/zeolite",
        "metadata_csv": "data/zeolite/publisher_metadata.csv",
        "ground_truth_csv": "data/zeolite/zeolite_data.csv",
        "evaluation_config": zeolite_column_config,
    },
    "aluminum": {
        "data_directory": "data/aluminum",
        "metadata_csv": "data/aluminum/publisher_metadata.csv",
        "ground_truth_csv": "data/aluminum/al_data.csv",
        "evaluation_config": al_column_config,
    },
}
