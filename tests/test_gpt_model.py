import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from models import gpt
from utils import templates

if __name__ == '__main__':
    model = gpt.GPTModel(torch_dtype=torch.bfloat16)
    prompt = 'Generate a function to calculate the factorial of a number.'
    prompt = templates.vanilla_template(prompt)
    gen_programs = model.generate(prompt, max_length=200, num_return_sequences=1)