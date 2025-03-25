from llm import DDDPredictorABC
from typing import Iterable
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

def postprocess_response(response_content: str) -> pd.DataFrame:
    return response_content

class OlmoPredictor(DDDPredictorABC):
    def __init__(self):
        self.model = OLMo()

    @property
    def accepts_pdf(self) -> bool:
        return False
    
    def predict_from_pdf(self, prompt: str):
        raise NotImplementedError

    def predict_from_xml(self, prompt: str):
        response = self.model(
            prompt=prompt,
        )
        if response is not None:
            return postprocess_response(response)
        else:
            return {}
        
    def predict_from_page_images(self, prompt: str, image_filenames: Iterable[str]):
        raise NotImplementedError


class OLMo(object):
    def __init__(self, system_prompt=None):
        self.model = AutoModelForCausalLM.from_pretrained(
            "allenai/OLMo-7B-0724-Instruct-hf",
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "allenai/OLMo-7B-0724-Instruct-hf",
            device_map="auto",
        )
        self.system_prompt = system_prompt

    def build_inputs(self, prompt):
        messages = []
        messages.append({"role": "user", "content": prompt})
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer.encode(inputs, add_special_tokens=False, return_tensors="pt")
        return inputs

    def __call__(self, prompt, image_paths=None, use_json=True, pbar=None):
        inputs = self.build_inputs(prompt)
        output = self.model.generate(
            input_ids=inputs.to(self.model.device),
            max_new_tokens=8192,
        )
        generated_tokens = output[0, inputs.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text