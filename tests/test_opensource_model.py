import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from models import opensource
from utils import templates
from transformers import LlamaForCausalLM, CodeLlamaTokenizer
from estimators import whitebox

if __name__ == '__main__':
    pipe = opensource.OpensourceModel(torch_dtype=torch.bfloat16)
    question = 'Generate a function to calculate the factorial of a number.'
    prompt = templates.vanilla_template(question) 

    input_ids = pipe.tokenizer(prompt, return_tensors='pt').input_ids
    generated = pipe.model.generate(input_ids, max_length=50, num_return_sequences=1, 
                                     return_dict_in_generate=True, output_scores=True)
    transition_scores = pipe.model.compute_transition_scores(
        generated.sequences, 
        generated.scores, 
        normalize_logits=True
    )
    print('log probabilities:', transition_scores)
    print('sequences log probabilities:', torch.mean(transition_scores, dim=1))
    breakpoint()