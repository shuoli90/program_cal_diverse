import os
import sys
import yaml
import subprocess
from datetime import datetime

# EXPERIMENT_OUTPUT_ROOT = "/home/data1/cal_diverse/open_ended_results/"
ALL_EXPERIMENT_OUTPUT_ROOT = "/data1/shypula/prog_diversity/all_experiments/"

if not os.path.exists(ALL_EXPERIMENT_OUTPUT_ROOT):
    os.makedirs(ALL_EXPERIMENT_OUTPUT_ROOT, exist_ok=True)
    print(f"Created directory {ALL_EXPERIMENT_OUTPUT_ROOT}.")
    
PATH_TO_HF_TOKEN="/home/shypula/hf_token.txt"

DEVICES="0,1,2,3,4,5,6,7"

MAX_LENGTH=1500
REPITITION_PENALTY=1.0
PARALLEL_MODEL_SAMPLES=5
PORT=9999
STARTUP_TIMEOUT=2000
VOLUME="saved_models"
GENERATION_TIMEOUT=1000
EVAL_WORKERS=10
EVAl_TIMEOUT=60
DOCKER_MAX_WORKERS=10
DOCKER_COMMUNICATION_TIMEOUT=2000


CONFIGS = [  
           # params: model, temperature, top_p, num_return_sequences, template, batch_size
           ['meta-llama/Meta-Llama-3-8B', 1.0, 1.0, 100, 'open_ended_default', 25], 
           ['meta-llama/Meta-Llama-3-8B', 1.0, 1.0, 100, 'open_ended_two_shot', 25],
           ['meta-llama/Meta-Llama-3-8B', 1.0, 1.0, 100, 'open_ended_two_shot_cot', 25],
           
    ]

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
    assert config[4] in ['open_ended_default', 'open_ended_two_shot', 'open_ended_two_shot_cot'], f"Template must be one of 'open_ended_default', 'open_ended_two_shot', 'open_ended_two_shot_cot', got {config[4]}."
    if len(config) == 6:
        assert isinstance(config[5], int), f"Batch size must be an integer, got {type(config[5])}."
    return True

def create_config(model, temperature, top_p, num_return_sequences, template, batch_size): 
    """Create YAML configuration file."""
    # experiment_string = f"{model_name_clean}_temp_{args.temperature}_top_p_{args.top_p}_max_length_{args.max_length}_num_return_sequences_{args.num_return_sequences}_repetition_penalty_{args.repetition_penalty}_{args.template}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    model_name_clean = model.replace('/', '-')
    experiment_name = f"{model_name_clean}_temp_{temperature}_top_p_{top_p}_num_return_sequences_{num_return_sequences}_{template}" 
    experiment_id = experiment_name + f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    config = {
        'path_to_dataset': '../data/open_ended_final/dataset_update.jsonl',
        'experiment_name': f"{model.replace('/', '-')}_temp_{temperature}_top_p_{top_p}_num_return_sequences_{num_return_sequences}_{template}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        'experiment_id': experiment_id,
        "experiment_output_root": ALL_EXPERIMENT_OUTPUT_ROOT,
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
        'docker_max_workers': DOCKER_MAX_WORKERS,
        'docker_communication_timeout': DOCKER_COMMUNICATION_TIMEOUT
        
    }
    return config 
    
    
    # file_path = os.path.join(config_dir, f'config_{model_name_clean}_{temperature}_{top_p}_{num_return_sequences}_{template}.yaml')
    # with open(file_path, 'w') as file:
    #     yaml.dump(config, file)
    # return file_path

# def run_experiment(config_path, log_file_path):
#     """Run the script with the given configuration file and stream logs."""
#     command = ['python', 'gen_eval_open_ended.py', config_path]
#     print(f"Running command: {' '.join(command)}")
    
#     command = command + ["2>&1 | tee", log_file_path]
#     os.system(" ".join(command))
            
#     # read back in config
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)
#     model = config['model'].replace('/', '-')
#     temperature = config['temperature']
#     top_p = config['top_p']
#     num_return_sequences = config['num_return_sequences']
    
#     dirs = [d for d in os.listdir(ALL_EXPERIMENT_OUTPUT_ROOT) if os.path.isdir(os.path.join(ALL_EXPERIMENT_OUTPUT_ROOT, d))]
#     if len(dirs) == 0:
#         print("No directories found.")
#         # import pdb; pdb.set_trace()
#         return None
#     time_sorted_dirs = sorted(dirs, key=lambda x: datetime.strptime(x[-19:], '%Y-%m-%d_%H-%M-%S'), reverse=True)
#     latest_dir = time_sorted_dirs[0]
#     results_file = os.path.join(ALL_EXPERIMENT_OUTPUT_ROOT, latest_dir, 'results_stats_mean.tsv')
#     # read in as dict
#     try: 
#         with open(results_file, 'r') as f:
#             lines = f.readlines()
#         results = {}
#         for line in lines:
#             k, v = line.strip().split('\t')
#             results[k] = v
#         results.update(config)
            
#         # coherence	semantic_count	distinct_1	distinct_2	distinct_3	distinct_4	distinct_5	distinct_6	corpus_self_bleu	plain_subtrees_3	plain_subtrees_4	plain_subtrees_5	plain_subtrees_6	stripped_subtrees_3	stripped_subtrees_4	stripped_subtrees_5	stripped_subtrees_6	obfuscated_subtrees_3	obfuscated_subtrees_4	obfuscated_subtrees_5	obfuscated_subtrees_6
#         results["model"] = model
#         print(f"Results for experiment {model}_temp_{temperature}_top_p_{top_p}_num_return_sequences_{num_return_sequences}:")
#         print(results)
#         return results
#     except Exception as e:
#         print(f"Error reading in results file: {e}")
#         return None
    
def reformat_config(config):
    if len(config) == 5:
        return_seqs = config[3]
        config = config + [return_seqs]
    return config
    

def main(configurations):
    # config_dir = '../configs'
    # if not os.path.exists(config_dir):
    #     raise FileNotFoundError(f"Configuration directory {config_dir} not found.")
    # os.makedirs(config_dir, exist_ok=True)

    this_driver_root = f"{ALL_EXPERIMENT_OUTPUT_ROOT}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    logs_dir = os.path.join(this_driver_root, 'driver_logs')
    os.makedirs(this_driver_root, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    
    # driver_stats_file = os.path.join(this_driver_root, f"driver_stats.tsv")
    # driver_pretty_stats_file = os.path.join(this_driver_root, f"driver_stats_pretty.txt")
    
    # keys = ['model', 'template', 'temperature', 'top_p', 'num_return_sequences']  
    # addtl_keys = ['coherence', 'semantic_count', 'semantic_proportion', 'distinct_1', 'distinct_2', 'distinct_3', 'distinct_4', 'distinct_5', 'distinct_6']
    # addtl_keys = addtl_keys + [f"{key}_{height}" for key in ['plain_subtrees', 'stripped_subtrees', 'obfuscated_subtrees'] for height in [3,4,5,6]]
    # addtl_keys = [f"{recordtype}_{key}" for recordtype in ['all', 'coh', 'err'] for key in addtl_keys]
    # keys = keys + addtl_keys
    
    # string_keys = ['model', 'template']
    # param_keys = ['temperature', 'top_p', 'num_return_sequences']
    # result_keys = [k for k in keys if k not in string_keys + param_keys and "semantic_count" not in k]
    
    
    # with open(driver_stats_file, 'w') as f:
    #     f.write('\t'.join(keys) + '\n')
    
    # pretty_column_widths = [46, 46] + [27] * (len(keys) - 2)
    # pretty_column_widths = [46, 15] + [max(len(k) + 2, 6) for k in keys[2:]]
    # with open(driver_pretty_stats_file, 'w') as f:
    # # Writing column headers with fixed width formatting
    #     f.write(''.join([f"{k.ljust(pretty_column_widths[i])}" for i, k in enumerate(keys)]) + '\n')
        
    assert all([validate_config(config) for config in configurations]), "Invalid configuration."
    
    configurations = [reformat_config(config) for config in configurations]

    config_path_list = []
    for config in configurations:
        full_config = create_config(*config)
        experiment_name = full_config['experiment_name']
        config_path = os.path.join(this_driver_root, f'{experiment_name}.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(full_config, f)
        config_path_list.append(config_path)
        
    with open(os.path.join(this_driver_root, 'all_configs_list.txt'), 'w') as f:
        f.write('\n'.join(config_path_list))
    
    
    os.system(" ".join(['python', 'eval_driver.py', this_driver_root, "&"]))
    os.system(" ".join(['python', 'generation_driver.py', this_driver_root]))
    
        
    
        
#         print(f"Running experiment with configuration {config}")
#         # results = run_experiment(yaml_path, os.path.join(logs_dir, f'log_{config[0]}_{config[1]}_{config[2]}_{config[3]}_{config[4]}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'))
#         results = run_experiment(yaml_path, os.path.join(logs_dir, f'log_{config[0].replace("/", "-")}_{config[1]}_{config[2]}_{config[3]}_{config[4]}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'))
#         if results is not None:
            
#             formatted_results = [] 
#             for k in keys:
#                 if k in result_keys:
#                     formatted_results.append(round(float(results[k]) * 100, 2))
#                 elif "semantic_count" in k:
#                     formatted_results.append(round(float(results[k]), 2))
#                 elif k == "template": 
#                     formatted_results.append(results[k].replace("open_ended_", ""))
#                 else:
#                     formatted_results.append(results[k])
                    
#         else: 
#             # prepare an error message
#             # formatted_results = [str(config[0]), str(config[1]), str(config[2]), str(config[3]), str(config[4])] + ['ERROR']*(len(keys) - 5)
#             formatted_results = [str(config[0]), str(config[4]), str(config[1]), str(config[2]), str(config[3])] + ['ERROR']*(len(keys) - 5)
            
#         with open(driver_stats_file, 'a') as f:
#                 f.write('\t'.join([str(k) for k in formatted_results]) + '\n')
            
#         with open(driver_pretty_stats_file, 'a') as f:
#             f.write(''.join([f"{str(k).ljust(pretty_column_widths[i])}" for i, k in enumerate(formatted_results)]) + '\n')
                
#         print(f"Experiment {config} completed.")
        
#     print("All experiments completed.")

# if __name__ == '__main__':
    
#     main(CONFIGS)
