from llm import DDDPredictorABC
from typing import Iterable
import transformers
import torch
import pandas as pd

def postprocess_response(response_content: str) -> pd.DataFrame:
    return response_content

class LlamaPredictor(DDDPredictorABC):
    def __init__(self):
        self.model = Llama()

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

class HuggingfaceLLM(object):
    def __init__(self, model_name, system_prompt=None):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.system_prompt = system_prompt
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def __call__(self, prompt, image_paths=None, use_json=True, pbar=None):
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        input_ = self.pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        outputs = self.pipeline(
            input_,
            max_new_tokens=8192,
            eos_token_id=self.terminators,
            do_sample=False,
            num_return_sequences=1,
        )

        results = [o["generated_text"][len(input_):] for o in outputs]
        return results[0]
    
class Llama(HuggingfaceLLM):
    def __init__(self, system_prompt=None):
        super().__init__("meta-llama/Llama-3.1-8B-Instruct", system_prompt)