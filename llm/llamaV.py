
from llm import DDDPredictorABC
from typing import Iterable
import pandas as pd
from transformers import MllamaForConditionalGeneration, AutoProcessor, GenerationConfig
from PIL import Image
import torch

def postprocess_response(response_content: str) -> pd.DataFrame:
    return response_content

class LlamaVisionPredictor(DDDPredictorABC):
    def __init__(self, blank_page_path: str):
        self.blank_page_path = blank_page_path
        self.model = LlamaVision()

    @property
    def accepts_pdf(self) -> bool:
        return False
    
    def predict_from_pdf(self, prompt: str):
        raise NotImplementedError

    def predict_from_xml(self, prompt: str):
        response = self.model(
            prompt=prompt,
            image_paths=[self.blank_page_path],
        )
        if response is not None:
            return postprocess_response(response)
        else:
            return {}
        
    def predict_from_page_images(self, prompt: str, image_filenames: Iterable[str]):
        response = self.model(prompt, image_filenames)
        if response is not None:
            return postprocess_response(response)
        else:
            return {}

class LlamaVision(object):
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            device_map='auto'
        )
        self.model = MllamaForConditionalGeneration.from_pretrained(
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )

    def build_inputs(self, prompt, image_paths):
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        images = [Image.open(ip) for ip in image_paths]

        inputs = self.processor(
            images=images,
            text=input_text,
            add_special_tokens=False,
            return_tensors='pt'
        ).to(self.model.device)

        return inputs
    
    def __call__(self, prompt, image_paths=None, use_json=True, pbar=None):
        if image_paths is None:
            image_paths = []

        inputs = self.build_inputs(prompt, image_paths)

        output = self.model.generate(
            **inputs,
            max_new_tokens=8192,
        )

        generated_tokens = output[0,inputs['input_ids'].size(1):]

        # only get generated tokens; decode them to text
        return self.processor.decode(generated_tokens, skip_special_tokens=True)