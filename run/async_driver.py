import os
import sys
import yaml
import subprocess
from datetime import datetime
import time
from dataclasses import dataclass
import subprocess


RUN_NAME="directed_debug"

ALL_EXPERIMENT_OUTPUT_ROOT = "/data1/shypula/prog_diversity/all_experiments/"

if not os.path.exists(ALL_EXPERIMENT_OUTPUT_ROOT):
    os.makedirs(ALL_EXPERIMENT_OUTPUT_ROOT, exist_ok=True)
    print(f"Created directory {ALL_EXPERIMENT_OUTPUT_ROOT}.")
    
PATH_TO_HF_TOKEN="/home/shypula/hf_token.txt"


MAX_LENGTH=1500
REPITITION_PENALTY=1.0
PARALLEL_MODEL_SAMPLES=5
PORT=9999
STARTUP_TIMEOUT=2000
VOLUME="saved_models"
GENERATION_TIMEOUT=1000
EVAL_WORKERS=20
EVAl_TIMEOUT=60
DOCKER_MAX_WORKERS=20
DOCKER_COMMUNICATION_TIMEOUT=2000
MAX_PROGRAMS=3
DIRECTED_DF_PATH="../data/high_solve_rate_problems/val_descriptions_and_testcases.jsonl"
OPEN_DF_PATH='../data/open_ended/open_ended_final/dataset.jsonl'

######## Important / To-Change Parameters ########

IS_DIRECTED=True

PATH_TO_DATASET = DIRECTED_DF_PATH if IS_DIRECTED else OPEN_DF_PATH
    
# DEVICES="0,1,2,3,4,5,6,7"
DEVICES="6,7"

CONFIGS = [  
           # params: model, temperature, top_p, num_return_sequences, template, batch_size
           ['meta-llama/Meta-Llama-3-8B-Instruct', 1.0, 1.0, 20, 'directed_default', 10],
           ['meta-llama/Meta-Llama-3-8B-Instruct', 1.0, 0.5, 20, 'directed_default', 10],
        #    ['meta-llama/Meta-Llama-3-8B', 1.0, 1.0, 100, 'open_ended_default', 25],
        #    ['meta-llama/Meta-Llama-3-70B-Instruct', 1.0, 1.0, 10, 'directed_default', 5],
        #    ['codellama/CodeLlama-34b-Instruct-hf', 1.0, 1.0, 100, 'open_ended_default', 25],
        #    ['tatsu-lab/alpaca-7b-wdiff', 1.0, 1.0, 100, 'open_ended_two_shot_cot', 25],
           
           
        # #    ['meta-llama/Meta-Llama-3-8B', 1.0, 1.0, 100, 'open_ended_default', 25], 
        #    ['meta-llama/Meta-Llama-3-8B', 1.0, 1.0, 100, 'open_ended_two_shot', 25],
        #    ['meta-llama/Meta-Llama-3-8B', 1.0, 1.0, 100, 'open_ended_two_shot_cot', 25],
           
        #    ['meta-llama/Meta-Llama-3-8B-Instruct', 1.0, 1.0, 100, 'open_ended_default', 25],
        #    ['meta-llama/Meta-Llama-3-8B-Instruct', 1.0, 1.0, 100, 'open_ended_two_shot', 25],
        #    ['meta-llama/Meta-Llama-3-8B-Instruct', 1.0, 1.0, 100, 'open_ended_two_shot_cot', 25],
           
        #    ['meta-llama/Meta-Llama-3-70B', 1.0, 1.0, 100, 'open_ended_default', 25],
        #     ['meta-llama/Meta-Llama-3-70B', 1.0, 1.0, 100, 'open_ended_two_shot', 25],
        #     ['meta-llama/Meta-Llama-3-70B', 1.0, 1.0, 100, 'open_ended_two_shot_cot', 25],
            
        #     # ['meta-llama/Meta-Llama-3-70B-Instruct', 1.0, 1.0, 100, 'open_ended_default', 25],
        #     ['meta-llama/Meta-Llama-3-70B-Instruct', 1.0, 1.0, 100, 'open_ended_two_shot', 25],
        #     ['meta-llama/Meta-Llama-3-70B-Instruct', 1.0, 1.0, 100, 'open_ended_two_shot_cot', 25],
            
        #     ['codellama/CodeLlama-7b-hf', 1.0, 1.0, 100, 'open_ended_default', 25],
        #     ['codellama/CodeLlama-7b-hf', 1.0, 1.0, 100, 'open_ended_two_shot', 25],
        #     ['codellama/CodeLlama-7b-hf', 1.0, 1.0, 100, 'open_ended_two_shot_cot', 25],
            
        #     ['codellama/CodeLlama-7b-Instruct-hf', 1.0, 1.0, 100, 'open_ended_default', 25],
        #     ['codellama/CodeLlama-7b-Instruct-hf', 1.0, 1.0, 100, 'open_ended_two_shot', 25],
        #     ['codellama/CodeLlama-7b-Instruct-hf', 1.0, 1.0, 100, 'open_ended_two_shot_cot', 25],
            
        #     ['codellama/CodeLlama-34b-hf', 1.0, 1.0, 100, 'open_ended_default', 25],
        #     ['codellama/CodeLlama-34b-hf', 1.0, 1.0, 100, 'open_ended_two_shot', 25],
        #     ['codellama/CodeLlama-34b-hf', 1.0, 1.0, 100, 'open_ended_two_shot_cot', 25],
            
        #     # ['codellama/CodeLlama-34b-Instruct-hf', 1.0, 1.0, 100, 'open_ended_default', 25],
        #     ['codellama/CodeLlama-34b-Instruct-hf', 1.0, 1.0, 100, 'open_ended_two_shot', 25],
        #     ['codellama/CodeLlama-34b-Instruct-hf', 1.0, 1.0, 100, 'open_ended_two_shot_cot', 25],
            
        #     # ['tatsu-lab/alpaca-7b-wdiff', 1.0, 1.0, 100, 'open_ended_two_shot_cot', 25],
        #     ['tatsu-lab/alpaca-7b-wdiff', 1.0, 1.0, 100, 'open_ended_default', 25],
        #     ['tatsu-lab/alpaca-7b-wdiff', 1.0, 1.0, 100, 'open_ended_two_shot', 25],

           
        #    ['tatsu-lab/alpaca-farm-sft10k-wdiff', 1.0, 1.0, 100, 'open_ended_default', 25],
        #    ['tatsu-lab/alpaca-farm-sft10k-wdiff', 1.0, 1.0, 100, 'open_ended_two_shot', 25],   
        #    ['tatsu-lab/alpaca-farm-sft10k-wdiff', 1.0, 1.0, 100, 'open_ended_two_shot_cot', 25],
           
        #    ['tatsu-lab/alpaca-farm-ppo-human-wdiff', 1.0, 1.0, 100, 'open_ended_default', 25],
        #    ['tatsu-lab/alpaca-farm-ppo-human-wdiff', 1.0, 1.0, 100, 'open_ended_two_shot', 25],
        #    ['tatsu-lab/alpaca-farm-ppo-human-wdiff', 1.0, 1.0, 100, 'open_ended_two_shot_cot', 25],
           
           
        # #    ['meta-llama/Meta-Llama-3-8B-Instruct', 1.0, 1.0, 100, 'open_ended_default', 25],
        # #    ['meta-llama/Meta-Llama-3-70B-Instruct', 1.0, 1.0, 100, 'open_ended_default', 25],                
           
        #    # go in increments of .2 until 1.2, then .1 until 2
        #    ['meta-llama/Meta-Llama-3-8B-Instruct', 0.2, 1.0, 100, 'open_ended_default', 25], 
        #       ['meta-llama/Meta-Llama-3-8B-Instruct', 0.4, 1.0, 100, 'open_ended_default', 25],
        #         ['meta-llama/Meta-Llama-3-8B-Instruct', 0.6, 1.0, 100, 'open_ended_default', 25],
        #             ['meta-llama/Meta-Llama-3-8B-Instruct', 0.8, 1.0, 100, 'open_ended_default', 25],
        #                 # ['meta-llama/Meta-Llama-3-8B-Instruct', 1.0, 1.0, 100, 'open_ended_default', 25],
            
            
        #     ['meta-llama/Meta-Llama-3-8B-Instruct', 1.1, 1.0, 100, 'open_ended_default', 25],
        #         ['meta-llama/Meta-Llama-3-8B-Instruct', 1.2, 1.0, 100, 'open_ended_default', 25],
        #             ['meta-llama/Meta-Llama-3-8B-Instruct', 1.3, 1.0, 100, 'open_ended_default', 25],
        #                 ['meta-llama/Meta-Llama-3-8B-Instruct', 1.4, 1.0, 100, 'open_ended_default', 25],
        #                     ['meta-llama/Meta-Llama-3-8B-Instruct', 1.5, 1.0, 100, 'open_ended_default', 25],
        #                         ['meta-llama/Meta-Llama-3-8B-Instruct', 1.6, 1.0, 100, 'open_ended_default', 25],
        #                             ['meta-llama/Meta-Llama-3-8B-Instruct', 1.7, 1.0, 100, 'open_ended_default', 25],
        #                                 ['meta-llama/Meta-Llama-3-8B-Instruct', 1.8, 1.0, 100, 'open_ended_default', 25],
        #                                     ['meta-llama/Meta-Llama-3-8B-Instruct', 1.9, 1.0, 100, 'open_ended_default', 25],
        #                                         ['meta-llama/Meta-Llama-3-8B-Instruct', 2.0, 1.0, 100, 'open_ended_default', 25],
                                        
                                        
        #     ['meta-llama/Meta-Llama-3-70B-Instruct', 0.2, 1.0, 100, 'open_ended_default', 25], 
        #         ['meta-llama/Meta-Llama-3-70B-Instruct', 0.4, 1.0, 100, 'open_ended_default', 25],
        #             ['meta-llama/Meta-Llama-3-70B-Instruct', 0.6, 1.0, 100, 'open_ended_default', 25],
        #                 ['meta-llama/Meta-Llama-3-70B-Instruct', 0.8, 1.0, 100, 'open_ended_default', 25],  
        #                     # ['meta-llama/Meta-Llama-3-70B-Instruct', 1.0, 1.0, 100, 'open_ended_default', 25],                
                            
        #     ['meta-llama/Meta-Llama-3-70B-Instruct', 1.1, 1.0, 100, 'open_ended_default', 25],
        #         ['meta-llama/Meta-Llama-3-70B-Instruct', 1.2, 1.0, 100, 'open_ended_default', 25],
        #             ['meta-llama/Meta-Llama-3-70B-Instruct', 1.3, 1.0, 100, 'open_ended_default', 25],
        #                 ['meta-llama/Meta-Llama-3-70B-Instruct', 1.4, 1.0, 100, 'open_ended_default', 25],
        #                     ['meta-llama/Meta-Llama-3-70B-Instruct', 1.5, 1.0, 100, 'open_ended_default', 25],
        #                         ['meta-llama/Meta-Llama-3-70B-Instruct', 1.6, 1.0, 100, 'open_ended_default', 25],
        #                             ['meta-llama/Meta-Llama-3-70B-Instruct', 1.7, 1.0, 100, 'open_ended_default', 25],  
        #                                 ['meta-llama/Meta-Llama-3-70B-Instruct', 1.8, 1.0, 100, 'open_ended_default', 25],
        #                                     ['meta-llama/Meta-Llama-3-70B-Instruct', 1.9, 1.0, 100, 'open_ended_default', 25],  
        #                                         ['meta-llama/Meta-Llama-3-70B-Instruct', 2.0, 1.0, 100, 'open_ended_default', 25],  
           
           
           
        #    ['codellama/CodeLlama-70b-Python-hf', 1.0, 1.0, 100, 'open_ended_default', 4],
        #    ['codellama/CodeLlama-70b-Python-hf', 1.0, 1.0, 100, 'open_ended_two_shot', 4], 
        #    ['codellama/CodeLlama-70b-Python-hf', 1.0, 1.0, 100, 'open_ended_two_shot_cot', 4],
            
        #    ['codellama/CodeLlama-70b-Instruct-hf', 1.0, 1.0, 100, 'open_ended_default', 4], 
        #    ['codellama/CodeLlama-70b-Instruct-hf', 1.0, 1.0, 100, 'open_ended_two_shot', 4], 
        #    ['codellama/CodeLlama-70b-Instruct-hf', 1.0, 1.0, 100, 'open_ended_two_shot_cot', 4]
           
    ]


@dataclass
class Arguments:
    experiment_name: str
    experiment_id: str 
    experiment_output_dir: str
    experiment_output_root: str 
    path_to_dataset: str = '../data/open_ended/open_ended_final/dataset.jsonl'
    model: str = 'gpt-3.5-turbo'
    template: str = 'open_ended_default'
    temperature: float = 1.0
    top_p: float = 1.0
    max_length: int = 768
    num_return_sequences: int = 10
    repetition_penalty: float = 1.0
    parallel_samples: int = 5
    port: int = 9999
    devices_list: str = '4,5,6,7'
    startup_timeout: int = 600
    generation_timeout: int = 100
    volume: str = 'saved_models'
    path_to_hf_token: str = None
    batch_size: int = None
    max_programs: int = -1
    eval_workers: int = 10
    eval_timeout: int = 60
    docker_communication_timeout: int = 2000
    reformat_results: bool = True
    is_directed: bool = False
    
    

def load_arguments_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        args_dict = yaml.safe_load(file)
    return Arguments(**args_dict)

template_dir = os.path.join(os.path.dirname(__file__), "../prompt_templates")
templates = [f for f in os.listdir(template_dir) if f.endswith('.txt')]
template_names = [f.split('.')[0] for f in templates] + [None]


def validate_config(config):
    assert len(config) == 5 or len(config) == 6, f"Configuration must have 5 elements or 6 elements, got {len(config)}."
    assert isinstance(config[0], str), f"Model must be a string, got {type(config[0])}."
    assert isinstance(config[1], float), f"Temperature must be a float, got {type(config[1])}."
    assert isinstance(config[2], float), f"Top-p must be a float, got {type(config[2])}."
    assert isinstance(config[3], int), f"Number of return sequences must be an integer, got {type(config[3])}."
    assert isinstance(config[4], str), f"Template must be a string, got {type(config[4])}."
    assert config[1] > 0, f"Temperature must be greater than 0, got {config[1]}."
    assert config[2] > 0, f"Top-p must be greater than 0, got {config[2]}."
    assert config[3] > 0, f"Number of return sequences must be greater than 0, got {config[3]}."
    assert config[4] in template_names, f"Template must be one of {' '.join(template_names)}, got {config[4]}."
    if len(config) == 6:
        assert isinstance(config[5], int), f"Batch size must be an integer, got {type(config[5])}."
    return True

def create_config(model, temperature, top_p, num_return_sequences, template, batch_size, driver_root): 
    """Create YAML configuration file."""
    # experiment_string = f"{model_name_clean}_temp_{args.temperature}_top_p_{args.top_p}_max_length_{args.max_length}_num_return_sequences_{args.num_return_sequences}_repetition_penalty_{args.repetition_penalty}_{args.template}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    model_name_clean = model.replace('/', '-')
    experiment_name = f"{model_name_clean}_temp_{temperature}_top_p_{top_p}_num_return_sequences_{num_return_sequences}_{template}" 
    experiment_id = experiment_name + f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    config = {
        'path_to_dataset': PATH_TO_DATASET, 
        'experiment_name': experiment_name,
        'experiment_id': experiment_id,
        "experiment_output_root": driver_root,
        'experiment_output_dir': os.path.join(driver_root, experiment_name),
        'model': model,
        'template': template,
        'temperature': temperature,
        'top_p': top_p,
        'max_length': MAX_LENGTH,
        'num_return_sequences': num_return_sequences,
        'batch_size': batch_size,
        'repetition_penalty': REPITITION_PENALTY,
        'parallel_samples': PARALLEL_MODEL_SAMPLES, # effecitve will be max of this and batch_size
        'port': PORT, 
        'devices_list': DEVICES,
        'startup_timeout': STARTUP_TIMEOUT,
        'volume': VOLUME,
        'generation_timeout': GENERATION_TIMEOUT,
        'path_to_hf_token': PATH_TO_HF_TOKEN, 
        'eval_workers': EVAL_WORKERS,
        'eval_timeout': EVAl_TIMEOUT,
        'docker_communication_timeout': DOCKER_COMMUNICATION_TIMEOUT, 
        'max_programs': MAX_PROGRAMS, 
        'is_directed': IS_DIRECTED
        
    }
    return config 
    

def reformat_config(config):
    if len(config) == 5:
        return_seqs = config[3]
        config = config + [return_seqs]
    return config
    

def main(configurations):

    # this_driver_root = f"{ALL_EXPERIMENT_OUTPUT_ROOT}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_name = "Run" if RUN_NAME == "" else RUN_NAME
    this_driver_root = os.path.join(ALL_EXPERIMENT_OUTPUT_ROOT, f"{run_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    logs_dir = os.path.join(this_driver_root, 'driver_logs')
    
    os.makedirs(this_driver_root, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Created directory {this_driver_root}/")

    assert all([validate_config(config) for config in configurations]), "Invalid configuration."
    # check that there are no duplicates
    assert len(set(tuple(config) for config in configurations)) == len(configurations), "Duplicate configurations."
    
    configurations = [reformat_config(config) for config in configurations]

    config_path_list = []
    for config in configurations:
        full_config = create_config(*config, driver_root=this_driver_root)
        experiment_name = full_config['experiment_name']
        config_path = os.path.join(this_driver_root, f'{experiment_name}.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(full_config, f)
        config_path_list.append(config_path)
        
    with open(os.path.join(this_driver_root, 'all_configs_list.txt'), 'w') as f:
        f.write('\n'.join(config_path_list))
    
    print(f"Created configuration files in {this_driver_root}.")
    print(f"Executing 'python eval_driver.py {this_driver_root} &'")
    # os.system(" ".join(['python', 'eval_driver.py', this_driver_root, "&"]))
    eval_driver = subprocess.Popen(["python", "eval_driver.py", this_driver_root])
    print(f"Executing 'python generation_driver.py {this_driver_root}'")
    os.system(" ".join(['python', 'generation_driver.py', this_driver_root]))
    print("Done Generating.")
    eval_driver.wait()
    print("Done Evaluating.")

if __name__ == "__main__":
    main(CONFIGS)
    
        