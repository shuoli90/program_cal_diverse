import os
from pathlib import Path
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from openai import OpenAI
from transformers import AutoTokenizer
import logging 
import openai 

logging.basicConfig(level=logging.INFO)

dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)
with open("/home/shypula/trustml_key.txt", "r") as f:
    api_key = f.read().strip()
with open("/home/shypula/trustml_organization.txt", "r") as f:
    organization = f.read().strip()
    
client = OpenAI(
    api_key=api_key,
    organization=organization
)


def chat_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
# openai.InternalServerError
# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
@retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(10), retry_error_callback=(lambda e: isinstance(e, openai.InternalServerError)))
def chatcompletions_with_backoff(model, messages, n, **kwargs):
    if model in ["gpt-3.5-turbo-instruct", "babbage-002", "davinci-002"]:
        return client.completions.create(
            model=model, 
            prompt=messages[0]['content'],
            n=n,
            **kwargs)
    else: 
        return client.chat.completions.create(
            model=model, 
            messages=messages,
            n=n,
            **kwargs)

class GPTModel:

    def __init__(self, model_name="gpt-3.5-turbo-0613",**kwargs):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained("openai-gpt")

    def generate(self, prompt: str, 
                 num_samples=1, 
                 max_new_tokens=50, 
                 top_k=None, # not implemented
                 top_p=1.0, 
                 do_sample=True, 
                 return_full_text=False,  # not implemented
                 temperature=1.0, 
                 return_dict_in_generate=False, # not implemented
                 batch_size=-1):
        
        if return_dict_in_generate or return_full_text or top_k:
            raise NotImplementedError("return_dict_in_generate, return_full_text, and top_k not implemented.")
        
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string.")
        
        if batch_size == -1:
            batch_size = num_samples

        message = {"role": "user",
                    "content": prompt} 
        completions = []

        while len(completions) < num_samples:
            this_batch_size = min(batch_size, num_samples - len(completions))
            
            try: 
                response = chatcompletions_with_backoff(
                    model=self.model_name,
                    messages=[message],
                    n=this_batch_size,
                    max_tokens=max_new_tokens,
                    top_p=top_p,
                    temperature=temperature if do_sample else 0,
                    # **kwargs # to be refined
                )
                completions.extend(response.choices)
            except openai.InternalServerError as e:
                logging.error(f"Internal Server Error: {e}")
                continue
            
            
        if self.model_name in ["gpt-3.5-turbo-instruct", "babbage-002", "davinci-002"]:
            gen_text = [choice.text.strip() for choice in completions]
        else: 
            gen_text = [choice.message.content.strip() for choice in completions]
            
        # responses_list = []
        # for completion in completions:
        #     if self.model_name in ["gpt-3.5-turbo-instruct", "babbage-002", "davinci-002"]:
        #         responses = [{'generated_text': choice.text.strip()}
        #                     for choice
        #                     in completion.choices]
        #     else: 
        #         responses = [{'generated_text': choice.message.content.strip()}
        #             for choice
        #             in completion.choices]
        #     # responses_list.append(responses)
        #     responses_list.extend(responses)
            
        # if return_full_text:
        #     for prompt, response in zip(prompts, responses_list):
        #         for item in response:
        #             item['generated_text'] = f"{prompt} {item['generated_text']}"
        # gen_text = [item['generated_text'] for response in responses_list for item in response]
        return gen_text