import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.hf_inference import HFInferenceManager, HFInferenceService, HFInferenceModel

PATH_TO_HF_TOKEN="/home/shypula/hf_token.txt"
with open(PATH_TO_HF_TOKEN, "r") as f:
    hf_key = f.read().strip()

def test_container_manager():
    print("Testing ContainerManager...")
    manager = HFInferenceManager("meta-llama/Meta-Llama-3-8B", 5, 9999, "4,5,6,7", startup_timeout=600, volume="saved_models", hf_key=hf_key)
    assert manager.container_id is not None, "Failed to start container"
    print("Container started with ID:", manager.container_id)
    
    running = manager.is_container_running()
    assert running, "Container is not running as expected"
    print("Container is running: Pass")
    
    manager.stop_container()
    print("Container stopped successfully.")

    manager.remove_container()
    print("Container removed successfully.")
    print("ContainerManager tests passed.\n")

def test_hf_inference_service():
    print("Testing HFInferenceService...")
    service = HFInferenceService("meta-llama/Meta-Llama-3-8B", 5, 9999, "4,5,6,7", volume="saved_models", startup_timeout=600, generation_timeout=60, hf_key=hf_key)
    
    results = service.generate("write a function to calculate the open and read a file, and then turn all whitespace into the character `@`", 256, 2, 1.2, True, 0.9) 
    assert type(results) == list and len(results) > 0, "Failed to generate text"
    print("Text generation successful. Output:")
    for result in results:
        print(result)
    
    service.stop_service()
    service.remove_service()
    print("HFInferenceService tests passed.\n")

def test_hf_inference_model_and_manager():
    manager = HFInferenceManager("meta-llama/Meta-Llama-3-8B", 5, 9999, "4,5,6,7", startup_timeout=600, volume="saved_models")
    assert manager.container_id is not None, "Failed to start container"
    print("Container started with ID:", manager.container_id)
    print("Manager tests passed.\n")
    print("Testing HFInferenceModel...")
    model = HFInferenceModel("http://127.0.0.1", 9999, 60)
    results = model.generate("Write a function to calculate the length of the collatz sequence for a given number.", 256, 2, 1.2, True, 0.9)
    assert type(results) == list and len(results) > 0, "Failed to generate text"
    print("Text generation successful. Output:", results)
    for result in results:
        print(result)
    print("HFInferenceModel tests passed.\n")

if __name__ == '__main__':
    test_container_manager()
    test_hf_inference_service()
    test_hf_inference_model_and_manager()
