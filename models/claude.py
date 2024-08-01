import boto3
from botocore.config import Config
from pathlib import Path
import json
from dotenv import load_dotenv
import logging 

logging.basicConfig(level=logging.INFO)

with open("/home/shypula/trustml_aws_access_key.txt", "r") as f:
    aws_access_key_id = f.read().strip()
with open("/home/shypula/trustml_aws_secret_key.txt", "r") as f:
    aws_secret_access_key = f.read().strip()

config = Config(read_timeout=3000)

# Anthropic API
# {
#     "anthropic_version": "bedrock-2023-05-31",    
#     "max_tokens": int,
#     "system": string,    
#     "messages": [
#         {
#             "role": string,
#             "content": [
#                 { "type": "image", "source": { "type": "base64", "media_type": "image/jpeg", "data": "content image bytes" } },
#                 { "type": "text", "text": "content text" }
#       ]
#         }
#     ],
#     "temperature": float,
#     "top_p": float,
#     "top_k": int,
#     "tools": [
#         {
#                 "name": string,
#                 "description": string,
#                 "input_schema": json
            
#         }
#     ],
#     "tool_choice": {
#         "type" :  string,
#         "name" : string,
#     },
#     "stop_sequences": [string]
# }              


# Bedrock Runtime
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
    aws_access_key_id=aws_access_key_id,  # optional - set this value if you haven't run `aws configure
    aws_secret_access_key=aws_secret_access_key,  # optional - set this value if you haven't run `aws configure`
    config=config,
    # aws_session_token=SESSION_TOKEN,   # optional - set this value if you haven't run `aws configure`
)

MODELS= {
    'HAIKU': 'anthropic.claude-3-haiku-20240307-v1:0',
    'SONNET': 'anthropic.claude-3-sonnet-20240229-v1:0',
    'OPUS': 'anthropic.claude-3-opus-20240229-v1:0'
}

class ClaudeModel:

    def __init__(self, model_name='SONNET', **kwargs):
        self.model_name = MODELS[model_name]
    
    def generate(self, 
                 prompt, 
                 num_samples=1, 
                 max_new_tokens=1500, 
                 top_k=None, 
                 top_p=1.0, 
                 return_full_text=False, 
                 temperature=1.0, 
                 do_sample=True,
                 return_dict_in_generate=False,
                 batch_size=1): 
        
        if batch_size != 1:
            logging.warning("Batch size is not implemented. It will effectively be set to 1.")
            
        if return_full_text or return_dict_in_generate:
            raise NotImplementedError("return_full_text and return_dict_in_generate not implemented.")
            
        
        if not do_sample:
            if not temperature == 0.0:
                logging.warning("Setting temperature to 0.0 for greedy decoding.")
            temperature = 0.0
            
        model_kwargs =  { 
            "max_tokens": max_new_tokens, 
            "temperature": temperature, 
            "top_p": top_p, 
            "stop_sequences": ["\n\nHuman"],
        }
        
        if top_k is not None:
            model_kwargs["top_k"] = top_k

        # Input configuration
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            # "system": "You are a honest and helpful bot.",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
        }
        body.update(model_kwargs)
        responses = []
        for _ in range(num_samples):
            response = bedrock_runtime.invoke_model(
                modelId=self.model_name,
                body=json.dumps(body),
            )
            # Process and print the response
            result = json.loads(response.get("body").read()).get("content", [])[0].get("text", "")
            responses.append(result)
        return responses
