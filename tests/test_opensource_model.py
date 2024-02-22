import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from models import opensource
from utils import textprocessing

if __name__ == '__main__':
    model = opensource.OpensourceModel(torch_dtype=torch.bfloat16)
    question = 'Generate a function to calculate the factorial of a number.'
    prompt = textprocessing.vanilla_template(question) 
    gen_programs = model.generate(prompt, max_length=200, num_return_sequences=1, return_full_text=False)