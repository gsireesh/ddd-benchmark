import os
import re

import fire
import pandas as pd

from evaluate_and_report import evaluate_predictions_wrapper


def compute_all_results(predictions_directory: str = "predictions"):
    for predictions_file in os.listdir(predictions_directory):
        parsed = re.fullmatch(
            r"(?P<dataset>.*?)_(?P<model>.*?)_(?P<modality>.*?).csv", predictions_file
        )
        if not parsed:
            continue
        dataset = parsed.group("dataset")
        modality = parsed.group("modality")
        model = parsed.group("model")
        if modality not in ["html"]:
            continue
        # if model != "claude-trim":
        #     continue
        evaluate_predictions_wrapper(
            dataset=dataset,
            modality=modality,
            predictions_path=os.path.join(predictions_directory, predictions_file),
            mode="all_available",
        )


if __name__ == "__main__":
    fire.Fire(compute_all_results)
