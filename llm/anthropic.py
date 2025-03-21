from json import JSONDecodeError
import os
from typing import Iterable

import anthropic
import json

from llm import DDDPredictorABC
from llm.utils import base64_encode_file


def postprocess_response(response_content: str) -> pd.DataFrame:
    return response_content


class AnthropicPredictor(DDDPredictorABC):

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)

    def predict_from_xml(self, prompt: str):
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=8192,
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
        contents = response.content[0].text

        if contents is not None:
            return postprocess_response(contents)
        else:
            return {}

    def predict_from_pdf(self, prompt: str, pdf_filename: str):
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=8192,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": base64_encode_file(pdf_filename),
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
        contents = response.content[0].text

        if contents is not None:
            return postprocess_response(contents)
        else:
            return {}

    def predict_from_page_images(self, prompt: str, image_filenames: Iterable[str]):
        media_type = f"image/{os.path.splitext(next(image_filenames.__iter__()))}"
        # Constructs tuples of messages for each page: one text message describing the image,
        # and then the image itself
        image_dict_pairs = [
            (
                {"type": "text", "text": f"Image of page {i+1}"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_encode_file(image_filename),
                    },
                },
            )
            for i, image_filename in enumerate(image_filenames)
        ]
        ## flattens the list of tuples of messages into a list of messages
        flat_image_messages = [
            message for message_pair in image_dict_pairs for message in message_pair
        ]

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=8192,
            messages=[
                {
                    "role": "user",
                    "content": [
                        *flat_image_messages,
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )
        contents = response.content[0].text

        if contents is not None:
            return postprocess_response(contents)
        else:
            return {}


class Anthropic:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)

    def __call__(self, prompt, image_path=None, pdf_path=None, pbar=None, **kwargs):
        if image_path is not None:
            base64_image = base64_encode_file(image_path)
            response = self.client.messages.create(
                model=self.model_name,
                stream=pbar is not None,
                max_tokens=8192,
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
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=8192,
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
        contents = response.content[0].text

        if contents is None:
            return {}
        try:
            return json.loads(contents)
        except JSONDecodeError as e:
            return {"response": contents, "error": e}
