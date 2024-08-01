import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from models import gpt
from utils import textprocessing
from transformers import AutoTokenizer



if __name__ == '__main__':
    
    for model_name in ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo-0125"]: 
        print(f"------------------ Model: {model_name} ------------------")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = gpt.GPTModel(torch_dtype=torch.bfloat16, model_name="gpt-3.5-turbo-instruct")
        prompt = 'Generate a function to calculate the factorial of a number.'
        prompt = textprocessing.vanilla_template(prompt)
        gen_programs = model.generate(prompt, 
                                    max_new_tokens=200,
                                    num_samples=5, 
                                    temperature=1.0, 
                                    top_p=1.0, 
                                    do_sample=True)
        assert len(gen_programs) == 5
        assert all (isinstance(program, str) for program in gen_programs)
        assert all (len(program) > 0 for program in gen_programs)
        tokenized_programs = [tokenizer(program) for program in gen_programs]
        assert all (len(tokens) > 0 for tokens in tokenized_programs)
        assert all (len(tokens) < 400 for tokens in tokenized_programs)
        for i, program in enumerate(gen_programs):
            print("******************************")
            print(f"Program {i+1}:\n{program}")
            
        # make sure new tokens is working 
        gen_programs_really_short = model.generate(prompt, 
                                    max_new_tokens=10,
                                    num_samples=2, 
                                    temperature=1.0, 
                                    top_p=1.0, 
                                    do_sample=True)
        tokenized_programs_really_short = [tokenizer(program) for program in gen_programs_really_short]
        assert all (len(tokens) > 0 for tokens in tokenized_programs_really_short)
        assert all (len(tokens) < 25 for tokens in tokenized_programs_really_short)
        
        