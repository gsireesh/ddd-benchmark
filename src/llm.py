import os
import json
import base64
from abc import ABC, abstractmethod

import openai
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image

class LLMBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, prompt, image_path=None, use_json=True, pbar=None):
        pass

class Molmo(LLMBase):
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(
            'allenai/Molmo-7B-O-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            'allenai/Molmo-7B-O-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

    def build_inputs(self, prompt, image_path):
        inputs = self.processor.process(
            images=[Image.open(image_path)],
            text=prompt
        )
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        return inputs
    
    def __call__(self, prompt, image_path=None, use_json=True, pbar=None):
        inputs = self.build_inputs(prompt, image_path)
        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer
        )

        # only get generated tokens; decode them to text
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text



class OpenAI(LLMBase):
    def __init__(self, model_name, org_key, project_key):
        self.model_name = model_name
        self.client = openai.OpenAI(
            organization=org_key,
            api_key=project_key,
        )

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def __call__(self, prompt, image_path=None, use_json=True, pbar=None):
        if image_path is not None:
            base64_image = self.encode_image(image_path)
            response = self.client.chat.completions.create(
                model=self.model_name,
                response_format={"type": "json_object"} if use_json else None,
                stream=(pbar is not None),
                messages=[ { "role": "user", "content": [
                    { "type": "text", "text": prompt, },
                    { "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, },
                ], } ],
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                response_format={"type": "json_object"} if use_json else None,
                stream=(pbar is not None),
                messages=[ { "role": "user", "content": [
                    { "type": "text", "text": prompt, },
                ], } ],
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
