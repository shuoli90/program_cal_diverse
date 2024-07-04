from transformers import pipeline
import functools
from text_generation import Client
import os 
import subprocess 
import time
import logging
import json
from tqdm import tqdm
import random
import requests

import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
run_dir = os.path.abspath(os.path.dirname(__file__))
# text_inference_template_path = os.path.join(project_dir, "text_inference_server.sh")

logging.basicConfig(level=logging.INFO)

class HFInferenceModel:
    
    def __init__(self, url="http://127.0.0.1", port=8080, timeout=100): 
        self.url = f"{url}:{port}"
        self.client = Client(self.url, timeout=timeout)

    def generate(self, prompt, max_new_tokens=512, num_samples=20, temperature=1.0, 
                    do_sample=True, top_p=1.0, top_k=None, return_dict_in_generate=False, batch_size=None, **kwargs):
        # remove best_of because we use num_samples
        if "best_of" in kwargs:
            kwargs.pop("best_of")
        if "return_dict_in_generate" in kwargs:
            kwargs.pop("return_dict_in_generate")
        if 'num_return_sequences' in kwargs:
            num_samples = kwargs.pop('num_return_sequences')
            logging.warning(f"Overriding num_return_sequences with num_samples: {num_samples}, pay attention to this!!!!!")
        if 'max_length' in kwargs:
            max_new_tokens = kwargs.pop('max_length')
            logging.warning(f"Overriding max_length with max_new_tokens: {max_new_tokens}, pay attention to this!!!!!")
        if top_p == 1.0 or kwargs.get("top_p", 1.0) == 1.0:
            top_p = None
            
        batch_size = batch_size or num_samples # if batch_size is None, use num_samples
        
        # check for kwargs 
        if kwargs:
            logging.warning(f"Unused kwargs: {kwargs}")
        
        # repeat until we get all completions, with batch_size
        pbar = tqdm(total=num_samples, desc=f"Generating {num_samples} samples")
        all_responses = []
        max_retries = 5
        retries_left = 5
        backoff_factor = 0.2
        while len(all_responses) < num_samples:    
            this_batch_size = min(batch_size, num_samples - len(all_responses))
            try: 
                completions = self.client.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    best_of=this_batch_size,
                )
                retries_left = max_retries
            except (ConnectionResetError, TimeoutError, requests.exceptions.RequestException) as e:
                if retries_left == 0:
                    logging.error(f"Connection reset error, no retries left, raising error")
                    raise e
                else:                    
                    wait_time = backoff_factor * (2 ** (max_retries - retries_left)) 
                    time.sleep(wait_time)
                    retries_left -= 1
                    continue
                    
            # get all completions from output
            best_of_sequences = [
                completions.details.best_of_sequences[i].generated_text
                for i in range(len(completions.details.best_of_sequences))
            ]
            new_responses = [completions.generated_text] + best_of_sequences
            all_responses.extend(new_responses)
            pbar.update(len(new_responses))
    
        return all_responses

    
class HFInferenceManager:   
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B", parallel_samples=20, port=8080, devices_list="0,1,2,3", startup_timeout=60, volume="saved_models", hf_key=None):
        # self.run_path = create_temp_inference_template(model_name, parallel_samples, port, devices_list)
        # run the script, and track to make sure it's running and that we can stop / restart it
        self.model_name = model_name
        self.parallel_samples = parallel_samples
        self.port = port
        self.devices_list = devices_list
        self.startup_timeout = startup_timeout
        self.volume = volume
        self.container_id = None
        self.hf_key = hf_key
        self.start_generation_container()

    def start_generation_container(self): 
        # command = f"docker run --detach --gpus all --shm-size 1g -p {port}:80 -v {volume}:/data ghcr.io/huggingface/text-generation-inference:latest --model-id {model} --max-best-of {max_best_of}"
        # with 1,2,3,4,5,6,7 gpus 
        # command = f"docker run --detach --gpus 1,2,3,4,5,6,7 --shm-size 1g -p {port}:80 -v {volume}:/data ghcr.io/huggingface/text-generation-inference:latest --model-id {model} --max-best-of {max_best_of}"
        # if not model.startswith("codellama"):
        #     model = f"data/{model}"
        max_tries = 5
        is_success=False
        while max_tries > 0 and not is_success:
            model, max_best_of, port, devices_list, volume, startup_timeout = self.model_name, self.parallel_samples, self.port, self.devices_list, self.volume, self.startup_timeout
            # command = f"docker run --detach -e HUGGING_FACE_HUB_TOKEN={self.hf_key} -e NVIDIA_VISIBLE_DEVICES={devices_list} --shm-size 1g -p {port}:80 -v {volume}:/data ghcr.io/huggingface/text-generation-inference::2.0.4 --model-id {model} --max-best-of {max_best_of}"
            # --cpus=30
            command = f"docker run --detach -e NCCL_P2P_DISABLE=1 -e HUGGING_FACE_HUB_TOKEN={self.hf_key} --gpus '\"device={devices_list}\"' --shm-size 1g -p {port}:80 -v {volume}:/data ghcr.io/huggingface/text-generation-inference:2.0.4 --model-id {model} --max-best-of {max_best_of}"
            # command = f"docker run --detach -e HUGGING_FACE_HUB_TOKEN={self.hf_key} --gpus '\"device={devices_list}\"' -e MAX_BATCH_SIZE=1 --shm-size 1g -p {port}:80 -v {volume}:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id {model} --max-best-of {max_best_of}"
            print("Starting container with command\n", command)
            container_id = subprocess.check_output(command, shell=True).decode().strip()
            # wait until the logs say Connected
            while True:
                logging.info(f"Waiting for container to start with id {container_id} and timeout {startup_timeout} left")
                logs = subprocess.check_output(f"docker logs {container_id}", shell=True).decode()
                if "Connected" in logs:
                    is_success=True
                    break
                if "Error" in logs:
                    max_tries -= 1
                    logging.error(f"Error starting container, {max_tries} left")
                    logging.info(f"Stopping container with id {container_id}")
                    self.stop_generation_container()
                    logging.info(f"Removing container with id {container_id}")
                    self.remove_generation_container()
                    break
                time.sleep(5)
                startup_timeout -= 5
                if startup_timeout <= 0:
                    raise TimeoutError("Timeout waiting for container to start")
        self.container_id = container_id
        
    def restart_generation_container(self): 
        container_id = self.container_id
        startup_timeout = self.startup_timeout
        
        if not self.is_container_running(container_id):
            old_logs = subprocess.check_output(f"docker logs {container_id}", shell=True).decode()
            subprocess.run(f"docker start {container_id}", shell=True)
            # wait until the logs say Connected
            while startup_timeout > 0:
                logging.info(f"Waiting for container to start with id {container_id} and timeout {startup_timeout} left")
                logs = subprocess.check_output(f"docker logs {container_id}", shell=True).decode()
                new_logs = logs.replace(old_logs, "")
                if "Connected" in new_logs:
                    return 
                time.sleep(5)
                startup_timeout -= 5
            # if we reach here, we've timed out
            raise TimeoutError("Timeout waiting for container to start")
        
    def is_container_running(self): 
        container_id = self.container_id
        try:
            # Using docker inspect to get container status
            result = subprocess.check_output(["docker", "inspect", container_id])
            container_info = json.loads(result)
            # Checking if the container's state is 'running'
            return container_info[0]["State"]["Running"]
        except subprocess.CalledProcessError as e:
            print(f"Error checking container status: {e}")
            return False

    def stop_generation_container(self): 
        subprocess.run(f"docker stop {self.container_id}", shell=True)

    def remove_generation_container(self):
        subprocess.run(f"docker rm {self.container_id}", shell=True)
        
        
class HFInferenceService: 
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B", parallel_samples=20, port=8080, devices_list="0,1,2,3", startup_timeout=60, volume="saved_models", generation_timeout=100, hf_key=None):
        self.manager = HFInferenceManager(model_name, parallel_samples, port, devices_list, startup_timeout=startup_timeout, volume=volume, hf_key=hf_key)
        self.model = HFInferenceModel(url="http://127.0.0.1", port=port, timeout=generation_timeout)
        
    def restart_service(self):
        self.manager.restart_generation_container()
    
    def stop_service(self):
        self.manager.stop_generation_container()
        
    def remove_service(self):
        self.manager.remove_generation_container()
        
    def stop_and_remove_if_running(self):
        if self.manager.is_container_running():
            self.stop_service()
            self.remove_service
        
    def generate(self, prompt, max_new_tokens=512, num_samples=20, temperature=1.0, 
                    do_sample=True, top_p=1.0, top_k=None, return_dict_in_generate=False, batch_size=10, **kwargs):
        completions = self.model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            num_samples=num_samples,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            # return_dict_in_generate=return_dict_in_generate,
            batch_size=batch_size,
            **kwargs
        )
        return completions

        