import json
import os
from pathlib import Path
import re
import time

from fire import Fire
from tqdm.notebook import tqdm
import pandas as pd


from llm import OpenAI
from llm.prompts import prompt_db
from llm.prompting import build_prompt
from llm.utils import convert_temperature

time_unit_key = "Time unit [D/m/h]"
temp_unit_key = "Temp unit [C/K/F]"

# Names of properties in the prompt
properties = [
    "Silicon",
    "Germanium",
    "Aluminum",
    "OH",
    "H2O",
    "HF",
    "SDA ratio",
    "Boron",
    "Time",
    time_unit_key,
    "Temp",
    temp_unit_key,
    "Extracted product",
    "SDA Type",
]

# Remapping some of the names for the reference dataset
map = {
    "Silicon": "Si",
    "Germanium": "Ge",
    "Aluminum": "Al",
    "Boron": "B",
    "Extracted product": "Extracted",
    "SDA ratio": "SDA",
}


def convert_df(df):
    "Rename columns, do some basic type and unit conversions for time and temperature."
    new_rows = []
    for i, row in df.iterrows():
        print(row)
        new_row = {}
        for k in row.keys():
            if k in map:
                new_row[map[k]] = row[k]
            else:
                new_row[k] = row[k]

        time_unit = None
        if time_unit_key in row:
            if isinstance(row[time_unit_key], str):
                time_unit = row[time_unit_key].lower()

        time_value = None
        if "Time" in row:
            if isinstance(row["Time"], str):
                try:
                    time_value = float(row["Time"])
                except:
                    time_value = None
            elif isinstance(row["Time"], float):
                time_value = row["Time"]

        if time_unit is None:
            time_unit = "h"

        new_time = None
        if time_value is not None:
            if "d" in time_unit:
                time_dilation = 24
            if "h" in time_unit:
                time_dilation = 1
            if "m" in time_unit:
                time_dilation = 1 / 60
            new_time = time_value * time_dilation

        temp_unit = None
        if temp_unit_key in row:
            if isinstance(row[temp_unit_key], str):
                temp_unit = row[temp_unit_key].lower()

        temp_value = None
        if "Temp" in row:
            if isinstance(row["Temp"], str):
                try:
                    temp_value = float(row["Temp"])
                except:
                    temp_value = None
            elif isinstance(row["Temp"], float):
                temp_value = row["Temp"]

        if temp_unit is None:
            temp_unit = "C"

        new_temp = None
        if temp_value is not None:
            new_temp = convert_temperature(
                float(row["Temp"]),
                "C" if "c" in temp_unit else ("F" if "f" in temp_unit else "K"),
                "C",
            )

        new_row["Time"] = new_time
        new_row["Temp"] = new_temp
        new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)[
        [
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
            "source",
            "doi",
            "doc_processing_time",
            "total_response_chars",
        ]
    ]
    return new_df


def get_llm(model_name, secrets):
    if "gpt" in model_name:
        llm = OpenAI(
            model_name="model_name",
            org_key=secrets["openai_org"],
            project_key=secrets["openai_project"],
        )
    else:
        raise AssertionError(f"Model name {model_name} does not map onto known provider.")


def get_responses(llm, model_name, modality, prompt_author, indir, ext):

    if ext == "png":
        prompt = build_prompt(model_name, modality, prompt_author, properties, prompt_db)

    for f in tqdm(sorted(indir.glob(f"*{ext}")), desc="Processing documents", position=0):
        if ext == "ml":
            with open(f) as fin:
                xml_data = fin.read()
                prompt = build_prompt(
                    model_name, modality, prompt_author, properties, prompt_db, context=xml_data
                )

        try:
            start = time.time()
            pbar = tqdm(desc="Streaming response characters", position=1)
            if ext == "png":
                response = llm(prompt, image_path=str(f), use_json=True, pbar=pbar)
            else:
                response = llm(prompt, use_json=True, pbar=pbar)
            stop = time.time()
            response["source"] = str(f.name)
            response["doi"] = re.sub(".png|.html|.xml", "", str(f.name)).replace("_", "/")
            yield {"response": response, "time": stop - start}
        except Exception as e:
            print(f"ERROR for {f}")
            print(e)
        break


def format_response_to_recipe_df(responses):
    clean_recipes = []
    for x in responses:
        response = x["response"]
        timing = x["time"]
        if "recipes" in response:
            for recipe in response["recipes"]:
                recipe_template = {k: None for k in properties}
                for k in recipe:
                    if k in properties:
                        recipe_template[k] = recipe[k]
                recipe_template["source"] = response["source"]
                recipe_template["doi"] = response["doi"]
                recipe_template["doc_processing_time"] = timing
                recipe_template["total_response_chars"] = len(json.dumps(response))
                clean_recipes.append(recipe_template)

    df = pd.DataFrame(clean_recipes)
    clean_df = convert_df(df)

    return df, clean_df


def generate_predictions(
    model_name: str,
    modality: str,
    prompt_author: str,
    keyfile: Path,
    input_directory: str,
    output_directory: str = "results/",
) -> None:
    with open(keyfile) as f:
        secrets = json.load(f)

    llm = get_llm(model_name, secrets)
    responses = get_responses(model_name, modality, prompt_author, keyfile, input_directory)
    response_df, converted_df = format_response_to_recipe_df(responses)

    outfile = Path(
        os.path.join(output_directory, f"{model_name}_{modality}_" f"{prompt_author}.test.csv")
    )

    response_df.to_csv(f"{outfile}.tmp")
    converted_df.to_csv(outfile)


if __name__ == "__main__":
    Fire(generate_predictions)
