from json import JSONDecodeError

import anthropic
import json

from llm.utils import base64_encode_file


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
