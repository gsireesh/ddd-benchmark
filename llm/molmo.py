from llm import DDDPredictorABC
from typing import Iterable
import pandas as pd
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image

def postprocess_response(response_content: str) -> pd.DataFrame:
    return response_content

class MolmoPredictor(DDDPredictorABC):
    def __init__(self, blank_page_path: str):
        self.blank_page_path = blank_page_path
        self.model = Molmo()

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

class Molmo(object):
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

    def build_inputs(self, prompt, image_paths):
        inputs = self.processor.process(
            images=[Image.open(ip) for ip in image_paths],
            text=prompt
        )
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        return inputs
    
    def __call__(self, prompt, image_paths=None, use_json=True, pbar=None):
        if image_paths is None:
            image_paths = []
        inputs = self.build_inputs(prompt, image_paths)
        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=8192, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer
        )

        # only get generated tokens; decode them to text
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text