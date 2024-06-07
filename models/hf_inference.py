from transformers import pipeline
import functools
from text_generation import Client

import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
run_dir = os.path.abspath(os.path.dirname(__file__))
text_inference_template_path = os.path.join(project_dir, "text_inference_server.sh")


class HFInferenceModel:
    
    def __init__(self, url="http://127.0.0.1:8080", timeout=100): 
        client = Client(url, timeout=timeout)

    def generate(self, prompt, max_new_tokens=512, num_samples=20, temperature=1.0, 
                    do_sample=True, top_p=1.0, top_k=50, **kwargs):
        completions = client.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            best_of=num_samples,
            **kwargs
        )
        return completions
    
    

# model='PATH_TO_MODEL' # 'codellama/CodeLlama-7b-hf' for example
# volume=$PWD/../saved_models/ # share a volume with the Docker container to avoid downloading weights every run
# max_best_of=PARALLEL_SAMPLES # max number of samples to generate in parallel

# docker run -e NVIDIA_VISIBLE_DEVICES="DEVICES_LIST" --shm-size 1g -p PORT:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest \
# --model-id $model --max-best-of $max_best_of

def format_text_inference_template(model_name, parallel_samples=20, port=8080, devices_list="0,1,2,3"):
    with open(text_inference_template_path, 'r') as f:
        text_inference_template = f.read()
    text_inference_template = text_inference_template.replace("PATH_TO_MODEL", model_name)
    text_inference_template = text_inference_template.replace("PARALLEL_SAMPLES", parallel_samples)
    text_inference_template = text_inference_template.replace("PORT", port)
    text_inference_template = text_inference_template.replace("DEVICES_LIST", devices_list)
    return text_inference_template

from uuid import uuid4
def create_temp_inference_template(model_name, parallel_samples=20, port=8080, devices_list="0,1,2,3"):
    formatted_text_inference_template = format_text_inference_template(model_name, parallel_samples, port, devices_list)
    run_path = os.path.join(run_dir, f"text_inference_{uuid4()}.sh")
    with open(run_path, 'w') as f:
        f.write(formatted_text_inference_template)
    return run_path

    
class HFInferenceManager:   
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B", parallel_samples=20, port=8080, devices_list="0,1,2,3", timeout=100, startup_timeout=60):
        self.run_path = create_temp_inference_template(model_name, parallel_samples, port, devices_list)
        # run the script, and track to make sure it's running and that we can stop / restart it
        self.model_name = model_name
        self.parallel_samples = parallel_samples
        self.port = port
        self.devices_list = devices_list
        self.timeout = timeout
        self.client = None
        self.start_generation_container()

    def start_generation_container(self)
        # command = f"docker run --detach --gpus all --shm-size 1g -p {port}:80 -v {volume}:/data ghcr.io/huggingface/text-generation-inference:latest --model-id {model} --max-best-of {max_best_of}"
        # with 1,2,3,4,5,6,7 gpus 
        # command = f"docker run --detach --gpus 1,2,3,4,5,6,7 --shm-size 1g -p {port}:80 -v {volume}:/data ghcr.io/huggingface/text-generation-inference:latest --model-id {model} --max-best-of {max_best_of}"
        # if not model.startswith("codellama"):
        #     model = f"data/{model}"
        port, volume, model, max_best_of = self.port, self.volume, self.model_name, self.parallel_samples
        command = f"docker run --detach -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --shm-size 1g -p {port}:80 -v {volume}:/data ghcr.io/huggingface/text-generation-inference:latest --model-id {model} --max-best-of {max_best_of}"
        container_id = subprocess.check_output(command, shell=True).decode().strip()
        # wait until the logs say Connected
        while True:
            logging.info(f"Waiting for container to start with id {container_id} and timeout {startup_timeout} left")
            logs = subprocess.check_output(f"docker logs {container_id}", shell=True).decode()
            if "Connected" in logs:
                break
            time.sleep(5)
            startup_timeout -= 5
            if startup_timeout <= 0:
                raise TimeoutError("Timeout waiting for container to start")
        self.container_id = container_id
        
    def restart_generation_container(self): 
        container_id = self.container_id
        startup_timeout = self.startup_timeout
        
        if not is_container_running(container_id):
            old_logs = subprocess.check_output(f"docker logs {container_id}", shell=True).decode()
            subprocess.run(f"docker start {container_id}", shell=True)
            # wait until the logs say Connected
            while True:
                logging.info(f"Waiting for container to start with id {container_id} and timeout {startup_timeout} left")
                logs = subprocess.check_output(f"docker logs {container_id}", shell=True).decode()
                new_logs = logs.replace(old_logs, "")
                if "Connected" in new_logs:
                    break
                time.sleep(5)
                startup_timeout -= 5
                if startup_timeout <= 0:
                    raise TimeoutError("Timeout waiting for container to start")
        
    def is_container_running(self)
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

    def stop_generation_container(self)
        subprocess.run(f"docker stop {self.container_id}", shell=True)

    def remove_generation_container(self):
        subprocess.run(f"docker rm {self.container_id}", shell=True)
            