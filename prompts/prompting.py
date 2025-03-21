import json


def build_prompt(model, modal, author, properties, prompt_db, context=None):
    if context is None:
        context = ""
    template = prompt_db[(model, modal, author)]
    recipe_format = "{"
    for p in properties:
        recipe_format += f'"{p}": <value>, '
    recipe_format = recipe_format[:-2]
    recipe_format += "}"
    reagents = "Si, Ge, Al, OH, H20"
    makers = "OpenAI" if "gpt" in model else "Anthropic"
    recipe_type = "zeolite synthesis"
    prompt = template.format(
        recipe_format=recipe_format,
        model_makers=makers,
        reagents=reagents,
        recipe_type=recipe_type,
        properties=" ".join(properties),
        context=context,
    )
    return prompt


def format_into_json_keys(keys: list[str]):
    return json.dumps({key: "<value>" for key in keys})


model_prefix_maker_map = {
    "gpt": "OpenAI",
    "claude": "Anthropic",
}

time_unit_key = "Time unit [D/m/h]"
temp_unit_key = "Temp unit [C/K/F]"

variables_by_dataset = {
    "zeolite": {
        "properties": [
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
        ],
        "recipe_format": format_into_json_keys(
            [
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
        ),
        "reagents": "Si, Ge, Al, OH, H20",
        "recipe_type": "zeolite synthesis",
    }
}


def infer_prompt_variables(model_name, dataset):
    model_maker = [
        maker
        for (model_prefix, maker) in model_prefix_maker_map.items()
        if model_name.startswith(model_prefix)
    ][0]

    dataset_specific_variables = variables_by_dataset[dataset]

    return {"model_makers": model_maker, **dataset_specific_variables}
