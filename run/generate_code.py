import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import argparse
from models import opensource, gpt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text from a prompt')
    parser.add_argument('--model', type=str, default='codellama/CodeLlama-7b-hf', 
                        help='model to use')
    parser.add_argument('--prompt', type=str, default='Once upon a time')
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    args = parser.parse_args()

    # Setup generation pipeline
    if 'gpt' in args.model:
        pipe = gpt.GPTModel(model_name=args.model)
    else:
        pipe = opensource.OpensourceModel(model_name=args.model)
    
    # Generate text
    generated = pipe.generate(args.prompt, max_length=args.max_length, num_return_sequences=args.num_return_sequences)
