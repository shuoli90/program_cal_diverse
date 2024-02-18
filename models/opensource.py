from transformers import pipeline
import functools

class OpensourceModel:
    
    def __init__(self, model_name='codellama/CodeLlama-7b-hf', **kwargs):
        self.pipe = pipeline(model=model_name, device_map="auto", **kwargs)
        if self.pipe.tokenizer.pad_token is None:
            self.pipe.tokenizer.pad_token = self.pipe.tokenizer.eos_token

    def generate(self, prompts, **kwargs):
        results = self.pipe(prompts, **kwargs)
        gen_texts = [result['generated_text'] for result in results]
        return gen_texts
    
    @functools.cached_property
    def model(self):
        return self.pipe.model
    
    @functools.cached_property
    def tokenizer(self):
        return self.pipe.tokenizer
