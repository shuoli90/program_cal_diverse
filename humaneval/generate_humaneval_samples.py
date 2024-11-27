import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data import write_jsonl, read_problems

MODEL_LIST = [
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.1-70B-Instruct",
    "codellama/CodeLlama-7b-hf",
    "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-34b-hf",
    "codellama/CodeLlama-34b-Instruct-hf",
    "codellama/CodeLlama-70b-hf",
    "codellama/CodeLlama-70b-Instruct-hf"
]
MAX_NEW_TOKENS = 1024
MAX_LENGTH = 1024
NUM_SAMPLES_PER_TASK = 100

class HFInferenceModel:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B", device=0, **kwargs):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_fast=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        kwargs = {**dict(model=model_name, use_fast=True), **kwargs}
    
    @torch.no_grad()
    def generate_one_completion(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True,  max_length=MAX_LENGTH).to('cuda')
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=MAX_NEW_TOKENS,
            **kwargs)

        completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return completion[len(prompt):].strip()
    
def generate_samples_for_model(model_name):
    print(f"Starting generation for model: {model_name}")
    model = HFInferenceModel(model_name=model_name)
    problems = read_problems()
    num_samples_per_task = NUM_SAMPLES_PER_TASK

    samples = []
    for task_id in tqdm(problems, desc=f"Generating completions for tasks ({model_name})"):
        for _ in tqdm(range(num_samples_per_task), desc=f"Task {task_id} ({model_name})", leave=False):
            samples.append(dict(task_id=task_id, completion=model.generate_one_completion(problems[task_id]["prompt"])))

    file_name = model_name.split("/")[-1]
    write_jsonl(f"../collected/{file_name}_samples.jsonl", samples)
    print(f"Completed generation for model: {model_name}")


with ThreadPoolExecutor(max_workers=len(MODEL_LIST)) as executor:
    executor.map(generate_samples_for_model, MODEL_LIST)