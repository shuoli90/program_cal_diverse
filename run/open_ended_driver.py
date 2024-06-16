import os
import sys
import yaml
import subprocess
from datetime import datetime

# EXPERIMENT_OUTPUT_ROOT = "/home/data1/cal_diverse/open_ended_results/"
EXPERIMENT_OUTPUT_ROOT = "/data1/shypula/prog_diversity/open_ended_debug/"
if not os.path.exists(EXPERIMENT_OUTPUT_ROOT):
    os.makedirs(EXPERIMENT_OUTPUT_ROOT, exist_ok=True)
    print(f"Created directory {EXPERIMENT_OUTPUT_ROOT}.")
    
PATH_TO_HF_TOKEN="/home/shypula/hf_token.txt"

# params: model, temperature, top_p, num_return_sequences, template
CONFIGS = [  
           
           ['meta-llama/Meta-Llama-3-8B-Instruct', 1.0, 1.0, 10, 'open_ended_default'],
           ['meta-llama/Meta-Llama-3-8B-Instruct', 1.0, 1.0, 10, 'open_ended_two_shot'],
           
        #    ['meta-llama/Meta-Llama-3-8B', 1.0, 1.0, 30, 'open_ended_default'],
        #    ['meta-llama/Meta-Llama-3-8B', 1.0, 1.0, 30, 'open_ended_two_shot'],
        #    ['meta-llama/Meta-Llama-3-8B', 1.0, 1.0, 30, 'open_ended_two_shot_cot'],
           
        #    ['meta-llama/Meta-Llama-3-8B-Instruct', 1.0, 1.0, 30, 'open_ended_default'],
        #    ['meta-llama/Meta-Llama-3-8B-Instruct', 1.0, 1.0, 30, 'open_ended_two_shot'],
        #    ['meta-llama/Meta-Llama-3-8B-Instruct', 1.0, 1.0, 30, 'open_ended_two_shot_cot'],
           
        #    ['meta-llama/Meta-Llama-3-70B', 1.0, 1.0, 30, 'open_ended_default'],
        #     ['meta-llama/Meta-Llama-3-70B', 1.0, 1.0, 30, 'open_ended_two_shot'],
        #     ['meta-llama/Meta-Llama-3-70B', 1.0, 1.0, 30, 'open_ended_two_shot_cot'],
            
        #     ['meta-llama/Meta-Llama-3-70B-Instruct', 1.0, 1.0, 30, 'open_ended_default'],
        #     ['meta-llama/Meta-Llama-3-70B-Instruct', 1.0, 1.0, 30, 'open_ended_two_shot'],
        #     ['meta-llama/Meta-Llama-3-70B-Instruct', 1.0, 1.0, 30, 'open_ended_two_shot_cot'],
            
        #     ## grid search over temperature and top_p, use 8B-Instruct and 70B-Instruct
            
        #     ['meta-llama/Meta-Llama-3-8B-Instruct', 0.75, 1.0, 30, 'open_ended_default'],
        #     ['meta-llama/Meta-Llama-3-8B-Instruct', 1.0, 1.0, 30, 'open_ended_default'],
        #     ['meta-llama/Meta-Llama-3-8B-Instruct', 1.25, 1.0, 30, 'open_ended_default'],
        #     ['meta-llama/Meta-Llama-3-8B-Instruct', 1.5, 1.0, 30, 'open_ended_default'],
        #     ['meta-llama/Meta-Llama-3-8B-Instruct', 1.75, 1.0, 30, 'open_ended_default'],
        #     ['meta-llama/Meta-Llama-3-8B-Instruct', 2.0, 1.0, 30, 'open_ended_default'],
            
            # ['meta-llama/Meta-Llama-3-8B-Instruct', 0.75, 0.9, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-8B-Instruct', 1.0, 0.9, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-8B-Instruct', 1.25, 0.9, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-8B-Instruct', 1.5, 0.9, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-8B-Instruct', 1.75, 0.9, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-8B-Instruct', 2.0, 0.9, 30, 'open_ended_default'],
            
            # ['meta-llama/Meta-Llama-3-8B-Instruct', 0.75, 0.8, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-8B-Instruct', 1.0, 0.8, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-8B-Instruct', 1.25, 0.8, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-8B-Instruct', 1.5, 0.8, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-8B-Instruct', 1.75, 0.8, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-8B-Instruct', 2.0, 0.8, 30, 'open_ended_default'],
            
            # ['meta-llama/Meta-Llama-3-8B-Instruct', 0.75, 0.7, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-8B-Instruct', 1.0, 0.7, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-8B-Instruct', 1.25, 0.7, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-8B-Instruct', 1.5, 0.7, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-8B-Instruct', 1.75, 0.7, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-8B-Instruct', 2.0, 0.7, 30, 'open_ended_default'],
            
            # # repeat for 70B-Instruct
            
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 0.75, 1.0, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 1.0, 1.0, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 1.25, 1.0, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 1.5, 1.0, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 1.75, 1.0, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 2.0, 1.0, 30, 'open_ended_default'],
            
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 0.75, 0.9, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 1.0, 0.9, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 1.25, 0.9, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 1.5, 0.9, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 1.75, 0.9, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 2.0, 0.9, 30, 'open_ended_default'],
            
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 0.75, 0.8, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 1.0, 0.8, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 1.25, 0.8, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 1.5, 0.8, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 1.75, 0.8, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 2.0, 0.8, 30, 'open_ended_default'],
            
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 0.75, 0.7, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 1.0, 0.7, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 1.25, 0.7, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 1.5, 0.7, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 1.75, 0.7, 30, 'open_ended_default'],
            # ['meta-llama/Meta-Llama-3-70B-Instruct', 2.0, 0.7, 30, 'open_ended_default'],
            
            
            # ['codellama/CodeLlama-34b-Instruct-hf', 1.0, 1.0, 30, 'open_ended_default'],
            # ['codellama/CodeLlama-34b-Instruct-hf', 1.0, 1.0, 30, 'open_ended_two_shot'],
            # ['codellama/CodeLlama-34b-Instruct-hf', 1.0, 1.0, 30, 'open_ended_two_shot_cot'],
            
            # ['codellama/CodeLlama-70b-Python-hf', 1.0, 1.0, 30, 'open_ended_default', 4],
            # ['codellama/CodeLlama-70b-Python-hf', 1.0, 1.0, 30, 'open_ended_two_shot', 4], 
            # ['codellama/CodeLlama-70b-Python-hf', 1.0, 1.0, 30, 'open_ended_two_shot_cot', 4],
            
            # ['codellama/CodeLlama-70b-Instruct-hf', 1.0, 1.0, 30, 'open_ended_default', 4], 
            # ['codellama/CodeLlama-70b-Instruct-hf', 1.0, 1.0, 30, 'open_ended_two_shot', 4], 
            # ['codellama/CodeLlama-70b-Instruct-hf', 1.0, 1.0, 30, 'open_ended_two_shot_cot', 4]
            
            # ['meta-llama/CodeLlama-70b-Python-hf', 1.0, 1.0, 30, 'open_ended_two_shot', 30], 
            # ['meta-llama/CodeLlama-70b-Python-hf', 1.0, 1.0, 30, 'open_ended_two_shot_cot', 30],
            
            
    
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

def create_yaml_config(model, temperature, top_p, num_return_sequences, template, batch_size, config_dir):
    """Create YAML configuration file."""
    config = {
        'path_to_dataset': '../data/open_ended_final/dataset.jsonl',
        "experiment_output_root": EXPERIMENT_OUTPUT_ROOT,
        'model': model,
        'template': template,
        'temperature': temperature,
        'top_p': top_p,
        'max_length': 1500,
        'num_return_sequences': num_return_sequences,
        'batch_size': batch_size,
        'repetition_penalty': 1.0, 
        'parallel_samples': 5, 
        'port': 9999, 
        'devices_list': '0,1,2,3',
        'startup_timeout': 2000,
        'volume': 'saved_models',
        'generation_timeout': 1000,
        'path_to_hf_token': PATH_TO_HF_TOKEN
    }
    
    model_name_clean = model.replace('/', '-')
    file_path = os.path.join(config_dir, f'config_{model_name_clean}_{temperature}_{top_p}_{num_return_sequences}_{template}.yaml')
    with open(file_path, 'w') as file:
        yaml.dump(config, file)
    return file_path

def run_experiment(config_path, log_file_path):
    """Run the script with the given configuration file and stream logs."""
    command = ['python', 'gen_eval_open_ended.py', config_path]
    print(f"Running command: {' '.join(command)}")
    
    command = command + ["2>&1 | tee", log_file_path]
    os.system(" ".join(command))
            
    
    # read back in config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model = config['model'].replace('/', '-')
    temperature = config['temperature']
    top_p = config['top_p']
    num_return_sequences = config['num_return_sequences']
    
    
    dirs = [d for d in os.listdir(EXPERIMENT_OUTPUT_ROOT) if os.path.isdir(os.path.join(EXPERIMENT_OUTPUT_ROOT, d))]
    if len(dirs) == 0:
        print("No directories found.")
        # import pdb; pdb.set_trace()
        return None
    time_sorted_dirs = sorted(dirs, key=lambda x: datetime.strptime(x[-19:], '%Y-%m-%d_%H-%M-%S'), reverse=True)
    latest_dir = time_sorted_dirs[0]
    results_file = os.path.join(EXPERIMENT_OUTPUT_ROOT, latest_dir, 'results_stats_mean.tsv')
    # read in as dict
    try: 
        with open(results_file, 'r') as f:
            lines = f.readlines()
        results = {}
        for line in lines:
            k, v = line.strip().split('\t')
            results[k] = v
        results.update(config)
            
        # coherence	semantic_count	distinct_1	distinct_2	distinct_3	distinct_4	distinct_5	distinct_6	corpus_self_bleu	plain_subtrees_3	plain_subtrees_4	plain_subtrees_5	plain_subtrees_6	stripped_subtrees_3	stripped_subtrees_4	stripped_subtrees_5	stripped_subtrees_6	obfuscated_subtrees_3	obfuscated_subtrees_4	obfuscated_subtrees_5	obfuscated_subtrees_6
        results["model"] = model
        print(f"Results for experiment {model}_temp_{temperature}_top_p_{top_p}_num_return_sequences_{num_return_sequences}:")
        print(results)
        return results
    except Exception as e:
        print(f"Error reading in results file: {e}")
        return None
    
def reformat_config(config):
    if len(config) == 5:
        return_seqs = config[3]
        config = config + [return_seqs]
    return config
    

def main(configurations):
    config_dir = '../configs'
    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Configuration directory {config_dir} not found.")
    # os.makedirs(config_dir, exist_ok=True)

    logs_dir = '/data1/shypula/prog_diversity/open_ended_logs/'
    os.makedirs(logs_dir, exist_ok=True)
    
    driver_stats_file = os.path.join(EXPERIMENT_OUTPUT_ROOT, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_driver_stats.tsv")
    driver_pretty_stats_file = os.path.join(EXPERIMENT_OUTPUT_ROOT, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_driver_stats_pretty.txt")
    
    # keys = ['model', 'temperature', 'top_p', 'num_return_sequences', 'template', 'coherence', 'semantic_count', 'distinct_1', 'distinct_2', 'distinct_3', 'distinct_4', 'distinct_5', 'distinct_6', 'corpus_self_bleu', 'plain_subtrees_3', 'plain_subtrees_4', 'plain_subtrees_5', 'plain_subtrees_6', 'stripped_subtrees_3', 'stripped_subtrees_4', 'stripped_subtrees_5', 'stripped_subtrees_6', 'obfuscated_subtrees_3', 'obfuscated_subtrees_4', 'obfuscated_subtrees_5', 'obfuscated_subtrees_6']
    # results_stats_keys = ['coherence', 'semantic_count', 'distinct_1', 'distinct_2', 'distinct_3', 'distinct_4', 'distinct_5', 'distinct_6']
    #     results_stats_keys = results_stats_keys + [f"{key}_{height}" for key in ['plain_subtrees', 'stripped_subtrees', 'obfuscated_subtrees'] for height in [3,4,5,6]]
    #     results_stats_keys = [f"{recordtype}_{key}" for recordtype in ['all', 'coherent', 'incoherent'] for key in results_stats_keys]
    
    keys = ['model', 'template', 'temperature', 'top_p', 'num_return_sequences']  
    addtl_keys = ['coherence', 'semantic_count', 'semantic_proportion', 'distinct_1', 'distinct_2', 'distinct_3', 'distinct_4', 'distinct_5', 'distinct_6']
    addtl_keys = addtl_keys + [f"{key}_{height}" for key in ['plain_subtrees', 'stripped_subtrees', 'obfuscated_subtrees'] for height in [3,4,5,6]]
    addtl_keys = [f"{recordtype}_{key}" for recordtype in ['all', 'coh', 'err'] for key in addtl_keys]
    keys = keys + addtl_keys
    
    string_keys = ['model', 'template']
    param_keys = ['temperature', 'top_p', 'num_return_sequences']
    result_keys = [k for k in keys if k not in string_keys + param_keys and "semantic_count" not in k]
    
    
    with open(driver_stats_file, 'w') as f:
        f.write('\t'.join(keys) + '\n')
    
    # pretty_column_widths = [46, 46] + [27] * (len(keys) - 2)
    pretty_column_widths = [46, 15] + [max(len(k) + 2, 6) for k in keys[2:]]
    with open(driver_pretty_stats_file, 'w') as f:
    # Writing column headers with fixed width formatting
        f.write(''.join([f"{k.ljust(pretty_column_widths[i])}" for i, k in enumerate(keys)]) + '\n')
        
    assert all([validate_config(config) for config in configurations]), "Invalid configuration."
    
    configurations = [reformat_config(config) for config in configurations]

    for config in configurations:
        yaml_path = create_yaml_config(*config, config_dir)
        print(f"Running experiment with configuration {config}")
        # results = run_experiment(yaml_path, os.path.join(logs_dir, f'log_{config[0]}_{config[1]}_{config[2]}_{config[3]}_{config[4]}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'))
        results = run_experiment(yaml_path, os.path.join(logs_dir, f'log_{config[0].replace("/", "-")}_{config[1]}_{config[2]}_{config[3]}_{config[4]}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'))
        if results is not None:
            
            formatted_results = [] 
            for k in keys:
                if k in result_keys:
                    formatted_results.append(round(float(results[k]) * 100, 2))
                elif "semantic_count" in k:
                    formatted_results.append(round(float(results[k]), 2))
                elif k == "template": 
                    formatted_results.append(results[k].replace("open_ended_", ""))
                else:
                    formatted_results.append(results[k])
                    
        else: 
            # prepare an error message
            # formatted_results = [str(config[0]), str(config[1]), str(config[2]), str(config[3]), str(config[4])] + ['ERROR']*(len(keys) - 5)
            formatted_results = [str(config[0]), str(config[4]), str(config[1]), str(config[2]), str(config[3])] + ['ERROR']*(len(keys) - 5)
            
        with open(driver_stats_file, 'a') as f:
                f.write('\t'.join([str(k) for k in formatted_results]) + '\n')
            
        with open(driver_pretty_stats_file, 'a') as f:
            f.write(''.join([f"{str(k).ljust(pretty_column_widths[i])}" for i, k in enumerate(formatted_results)]) + '\n')
                
        print(f"Experiment {config} completed.")
        
    print("All experiments completed.")

if __name__ == '__main__':
    
    main(CONFIGS)
