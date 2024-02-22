import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from models import opensource, gpt
from utils import textprocessing
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text from a prompt')
    parser.add_argument('--model', type=str,
                        default='codellama/CodeLlama-7b-hf',
                        help='model to use')
    parser.add_argument('--prompt', type=str, default='Once upon a time')
    parser.add_argument('--template', type=str, default='vanilla')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    args = parser.parse_args()

    # Setup generation pipeline
    if 'gpt' in args.model:
        pipe = gpt.GPTModel(model_name=args.model)
    else:
        pipe = opensource.OpensourceModel(model_name=args.model)

    if args.template == 'vanilla':
        prompt = textprocessing.vanilla_template(args.prompt)
    else:
        raise ValueError("Unknow template")

    # Generate text
    generations= pipe.generate(args.prompt, temperature=args.temperature,
                              max_length=args.max_length,
                              num_return_sequences=args.num_return_sequences,
                              return_full_text=False)
    generation = textprocessing.extract_functions(generations[0])
    print(generation)
    breakpoint()