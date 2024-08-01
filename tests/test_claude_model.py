import os
import sys
import logging

from transformers import AutoTokenizer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import claude

def test_claude_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    for model_name in ["SONNET", "HAIKU"]:
        print(f"\n\n------------------ Model: {model_name} ------------------\n\n")
        
        model = claude.ClaudeModel(model_name=model_name)
        prompt = 'Generate a function to calculate the factorial of a number.'
        
        # Test basic generation
        gen_responses = model.generate(prompt, 
                                       max_new_tokens=200,
                                       num_samples=5, 
                                       temperature=1.0, 
                                       top_p=1.0, 
                                       do_sample=True)
        
        assert len(gen_responses) == 5, f"Expected 5 responses, got {len(gen_responses)}"
        assert all(isinstance(response, str) for response in gen_responses), "All responses should be strings"
        assert all(len(response) > 0 for response in gen_responses), "All responses should be non-empty"
        
        for i, response in enumerate(gen_responses):
            print("******************************")
            print(f"Response {i+1}:\n{response}")
        
        gen_programs_really_short = model.generate(prompt, 
                                    max_new_tokens=10,
                                    num_samples=2, 
                                    temperature=1.0, 
                                    top_p=1.0, 
                                    do_sample=True)
        tokenized_programs_really_short = [tokenizer(program) for program in gen_programs_really_short]
        assert all (len(tokens) > 0 for tokens in tokenized_programs_really_short)
        assert all (len(tokens) < 25 for tokens in tokenized_programs_really_short)
        
        
        

if __name__ == '__main__':
    test_claude_model()