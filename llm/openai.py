import os
import openai
import json
import base64

from llm.utils import base64_encode_file
from llm.DDDPredictor import DDDPredictorABC


class OpenAIPredictor(DDDPredictorABC):

    def __init__(self, model_name, org_id, api_key):
        self.model_name = model_name
        self.client = openai.OpenAI(
            organization=org_id,
            api_key=api_key,
        )

    def predict_from_xml(self, prompt, use_json=True, pbar=None):
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
        elif use_json:
            try:
                return json.loads(contents)
            except:
                return {}
        else:
            return contents


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
        elif use_json:
            try:
                return json.loads(contents)
            except:
                return {}
        else:
            return contents
