import json
import os
from pathlib import Path

from fire import Fire
import pandas as pd

from data.dataset_metadata import metadata_by_dataset
from llm import AnthropicPredictor, DDDPredictorABC, OpenAIPredictor, MolmoPredictor
from prompts import get_prompt_builders, infer_prompt_variables


RESULTS_DIRECTORY = "results"


def get_llm(model_name, secrets) -> DDDPredictorABC:
    if "gpt" in model_name:
        llm = OpenAIPredictor(
            model_name=model_name,
            org_id=secrets["openai_org"],
            api_key=secrets["openai_project"],
        )
    elif "claude" in model_name:
        llm = AnthropicPredictor(model_name=model_name, api_key=secrets["anthropic_api_key"])
    elif model_name == "molmo":
        llm = MolmoPredictor(model_name=model_name)
    else:
        raise AssertionError(f"Model name {model_name} does not map onto known provider.")
    return llm


def generate_predictions(
    model_name: str,
    modality: str,
    dataset: str,
    prompt_name_filter: str = "",
    secrets_file: str = "secrets.json",
) -> None:
    with open(secrets_file) as f:
        secrets = json.load(f)

    llm = get_llm(model_name, secrets)
    dataset_metadata = metadata_by_dataset[dataset]
    data_directory = dataset_metadata["data_directory"]
    meta_df = pd.read_csv(dataset_metadata["metadata_csv"])

    prompt_builders = get_prompt_builders(
        model_name=model_name,
        dataset=dataset,
        modality=modality,
        prompt_name_filter=prompt_name_filter,
    )
    prompt_variables = infer_prompt_variables(model_name, dataset)

    for prompt_builder in prompt_builders:
        prompt_dfs = []
        for doi in meta_df["doi"]:
            if modality == "xml":
                filename = doi.replace("/", "_") + ".xml"
                if not os.path.exists(filename):
                    continue
                with open(os.path.join(data_directory, "xml", filename)) as f:
                    xml_content = f.read()
                prompt = prompt_builder.template.format(**prompt_variables, context=xml_content)
                doi_df = llm.predict_from_xml(prompt)
            elif modality == "pdf":
                if llm.accepts_pdf:
                    pdf_filename = os.path.join(
                        data_directory, "pdf", doi.replace("/", "_") + ".pdf"
                    )
                    if not os.path.exists(pdf_filename):
                        continue
                    prompt = prompt_builder.template.format(**prompt_variables)
                    doi_df = llm.predict_from_pdf(prompt, pdf_filename)
                else:
                    filename_glob = doi.replace("/", "_") + "_*.png"
                    image_files = list(Path(data_directory, "page_images").glob(filename_glob))
                    if not os.path.exists(image_files[0]):
                        continue
                    prompt = prompt_builder.template.format(**prompt_variables)
                    doi_df = llm.predict_from_page_images(prompt, image_files)
            else:
                raise AssertionError(f"Modality {modality} not in accepted set: xml, pdfs")

            doi_df["doi"] = doi
            doi_df["id"] = prompt_builder.id
            prompt_dfs.append(doi_df)
        prompt_df = pd.concat(prompt_dfs)
        outfile = os.path.join(
            RESULTS_DIRECTORY, f"{model_name}_{modality}_" f"{prompt_builder.id}.test.csv"
        )
        prompt_df.to_csv(outfile, index=False)


if __name__ == "__main__":
    Fire(generate_predictions)
