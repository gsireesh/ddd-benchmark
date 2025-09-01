import pandas as pd
import os
from pathlib import Path
import json

class Config:
    DATASET_CATEGORY = "zeolite"   # zeolite, aluminum
    MODE = "html"  # pdf, xml, html, md, txt
    MODEL = "claude"  # claude, gpt4o
    ROOT_DIR = Path(__file__).parent.parent 
    RESPONSE_DIR = ROOT_DIR / MODEL / DATASET_CATEGORY / MODE / "responses"
    OUTPUT_DIR = ROOT_DIR / "predictions"

    if not RESPONSE_DIR.exists():
        raise FileNotFoundError(f"Response directory {RESPONSE_DIR} does not exist.")
    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(f"Output directory {OUTPUT_DIR} does not exist.")
    # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def response_to_recipes(file_path: Path) -> list[dict[str,str|float|int]]:
    """
    Extract recipes from a JSON in a plain text response file.
    """
    response_text = file_path.read_text()
    try:
        start = response_text.index("{")
        end = response_text.rindex("}") + 1
        json_str = response_text[start:end]
        json_dict = json.loads(json_str, strict=False)
        return json_dict["recipes"]
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid JSON format in the response file: {e}")
    
def clean_recipe(recipes: list[dict[str,str|float|int]], doi:str) -> list[dict[str,str|float|int]]:
    for recipe in recipes:
        recipe["doi"] = doi
        # if "precursors" in recipe:
        #     recipe["precursors"] = recipe["precursors"].split(", ")
    return recipes
    
def zeolite_postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns that are relevant for zeolite recipes.
    """
    zeolite_columns = [
        "doi",
        "Si",
        "Ge",
        "Al",
        "OH",
        "H2O",
        "HF",
        "B",
        "osda_names", 
        "osda_ratios", 
        "crystallization_temperature", 
        "crystallization_time", 
        "product_names",
    ]
    return df[zeolite_columns].rename(columns={
        "osda_names": "SDA_Type", 
        "osda_ratios": "SDA", 
        "crystallization_temperature":"Temp", 
        "crystallization_time":"Time",
        "product_names":"Extracted"
        })

def aluminum_postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns that are relevant for aluminum recipes.
    """
    aluminum_columns = [
        "doi",
        "AA",
        "temper",
        "YS [MPa]",
        "UTS [MPa]",
        "elong [%]",
        "Hardness",
        "Hardness UNIT",
        "Has composition",
        "Cu",
        "Mn",
        "Si",
        "Mg",
        "Zn",
        "Cr",
        "Fe",
        "Ti",
        "Zr",
        "Ag",
        "Be",
        "Bi",
        "C",
        "Ca",
        "Ce",
        "Er",
        "Ga",
        "Ge",
        "Hf",
        "La",
        "Li",
        "Ni",
        "P",
        "Pb",
        "Sc",
        "Sn",
        "Sr",
        "V",
        "Yb",
    ]
    for col in aluminum_columns:
        if col not in df.columns: df[col] = ""
    df_Al = df[aluminum_columns]
    df_Al = df_Al.rename(columns={
        "Has composition": "Has comp [True / False / nominal]"
    })
    # df_Al.loc[:, "Has comp [True / False / nominal]"] = df_Al["Has comp [True / False / nominal]"].fillna(df_Al["Has comp"])
    # df_Al = df_Al.drop(columns=["Has comp"])
    return df_Al

def response_to_csv() -> None:
    """Convert response files to a CSV file."""
    response_files = list(Config.RESPONSE_DIR.glob("*.json"))
    if not response_files:
        print(f"No response files found in {Config.RESPONSE_DIR}.")
        return

    data = []
    for file_path in response_files:
        try:
            recipes = response_to_recipes(file_path)
            doi = "/".join(file_path.stem.split("_")[1:])
            recipe_list = clean_recipe(recipes, doi)
            data.extend(recipe_list)
        except ValueError as e:
            print(f"Error processing {file_path}: {e}")

    df = pd.DataFrame(data)
    columns = df.columns.tolist()
    columns.remove("doi")
    columns.insert(0, "doi")
    df = df[columns]
    df = df.sort_values(by=["doi"])

    if Config.DATASET_CATEGORY == "zeolite":
        df = zeolite_postprocess(df)

    elif Config.DATASET_CATEGORY == "aluminum":
        df = aluminum_postprocess(df)

    output_file = Config.OUTPUT_DIR / f"{Config.DATASET_CATEGORY}_{Config.MODEL}_{Config.MODE}.csv".lower()
    df.to_csv(output_file, index=False)
    print(f"Responses saved to {output_file.name}")

if __name__ == "__main__":
    response_to_csv()