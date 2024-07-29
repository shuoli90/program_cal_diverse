import boto3
from pathlib import Path
import json
from dotenv import load_dotenv

# Bedrock Runtime
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
    aws_access_key_id="",      # optional - set this value if you haven't run `aws configure` 
    aws_secret_access_key="",  # optional - set this value if you haven't run `aws configure`
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
    
    def generate(self, prompt, num_samples=1, max_length=50, top_k=250, top_p=1, return_full_text=False, temperature=1.0, **kwargs):
        
        model_kwargs =  { 
            "max_tokens": max_length, 
            "temperature": temperature, 
            "top_p": top_p, 
            "stop_sequences": ["\n\nHuman"],
        }

        # Input configuration
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": "You are a honest and helpful bot.",
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
