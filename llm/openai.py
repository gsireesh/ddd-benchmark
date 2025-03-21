import json
from typing import Iterable

import openai
import pandas as pd
from tqdm import tqdm

from llm.DDDPredictor import DDDPredictorABC
from llm.utils import base64_encode_file


# def get_responses(llm, model_name, modality, prompt_author, indir, ext):
#
#     if ext == "png":
#         prompt = build_prompt(model_name, modality, prompt_author, properties, prompt_db)
#
#     for f in tqdm(sorted(indir.glob(f"*{ext}")), desc="Processing documents", position=0):
#         if ext == "ml":
#             with open(f) as fin:
#                 xml_data = fin.read()
#                 prompt = build_prompt(
#                     model_name, modality, prompt_author, properties, prompt_db, context=xml_data
#                 )
#
#         try:
#             start = time.time()
#             pbar = tqdm(desc="Streaming response characters", position=1)
#             if ext == "png":
#                 response = llm(prompt, image_path=str(f), use_json=True, pbar=pbar)
#             else:
#                 response = llm(prompt, use_json=True, pbar=pbar)
#             stop = time.time()
#             response["source"] = str(f.name)
#             response["doi"] = re.sub(".png|.html|.xml", "", str(f.name)).replace("_", "/")
#             yield {"response": response, "time": stop - start}
#         except Exception as e:
#             print(f"ERROR for {f}")
#             print(e)
#         break
#
#
# def format_response_to_recipe_df(responses):
#     clean_recipes = []
#     for x in responses:
#         response = x["response"]
#         timing = x["time"]
#         if "recipes" in response:
#             for recipe in response["recipes"]:
#                 recipe_template = {k: None for k in properties}
#                 for k in recipe:
#                     if k in properties:
#                         recipe_template[k] = recipe[k]
#                 recipe_template["source"] = response["source"]
#                 recipe_template["doi"] = response["doi"]
#                 recipe_template["doc_processing_time"] = timing
#                 recipe_template["total_response_chars"] = len(json.dumps(response))
#                 clean_recipes.append(recipe_template)
#
#     df = pd.DataFrame(clean_recipes)
#     clean_df = convert_df(df)
#
#     return df, clean_df
#
#
def construct_response_from_stream(response_stream) -> str:
    pbar = tqdm(desc="Streaming response characters", position=1)
    contents = ""
    for chunk in response_stream:
        delta = chunk.choices[0].delta.content
        if delta is None:
            continue
        contents += delta
        pbar.update(len(delta))

    return contents


def postprocess_response(response_content: str) -> pd.DataFrame:
    return response_content


class OpenAIPredictor(DDDPredictorABC):

    def __init__(self, model_name, org_id, api_key, stream=False):
        self.model_name = model_name
        self.client = openai.OpenAI(
            organization=org_id,
            api_key=api_key,
        )
        self.stream = stream

    def predict_from_xml(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            stream=self.stream,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        if self.stream:
            contents = construct_response_from_stream(response)
        else:
            contents = response.choices[0].message.content

        if contents is not None:
            return postprocess_response(contents)
        else:
            return {}

    def predict_from_pdf(self, prompt: str, pdf_filename: str):
        pdf_content = base64_encode_file(pdf_filename)
        response = self.client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            stream=self.stream,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {
                                "filename": pdf_filename,
                                "file_data": f"data:application/pdf;base64,{pdf_content}",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        if self.stream:
            contents = construct_response_from_stream(response)
        else:
            contents = response.choices[0].message.content

        if contents is not None:
            return postprocess_response(contents)
        else:
            return {}

    def predict_from_page_images(self, prompt: str, image_filenames: Iterable[str]):
        response = self.client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            stream=self.stream,
            messages=[
                {
                    "role": "user",
                    "content": [
                        *[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_encode_file(image_filename)}",
                                },
                            }
                            for image_filename in image_filenames
                        ],
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        if self.stream:
            contents = construct_response_from_stream(response)
        else:
            contents = response.choices[0].message.content

        if contents is not None:
            return postprocess_response(contents)
        else:
            return {}


class OpenAI(object):
    def __init__(self, model_name, org_key, project_key):
        self.model_name = model_name
        self.client = openai.OpenAI(
            organization=org_key,
            api_key=project_key,
        )

    def __call__(self, prompt, image_path=None, use_json=True, pbar=None):
        if image_path is not None:
            base64_image = base64_encode_file(image_path)
            response = self.client.chat.completions.create(
                model=self.model_name,
                response_format={"type": "json_object"} if use_json else None,
                stream=(pbar is not None),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                response_format={"type": "json_object"} if use_json else None,
                stream=(pbar is not None),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )

        if pbar is not None:
            contents = ""
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta is None:
                    continue
                contents += delta
                pbar.update(len(delta))
        else:
            contents = response.choices[0].message.content

        if contents is None:
            return {}
