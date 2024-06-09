import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from models import claude

if __name__ == '__main__':
    model = claude.ClaudeModel(model_name='SONNET')
    prompt = 'Generate a function to calculate the factorial of a number in python.'
    gen_programs = model.generate(prompt, max_length=200, num_return_sequences=3)
    print(gen_programs)
    breakpoint()