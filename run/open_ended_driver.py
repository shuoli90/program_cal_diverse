import os
import sys
import yaml
import subprocess
from datetime import datetime

# EXPERIMENT_OUTPUT_ROOT = "/home/data1/cal_diverse/open_ended_results/"
EXPERIMENT_OUTPUT_ROOT = "/data1/shypula/prog_diversity/open_ended/"
if not os.path.exists(EXPERIMENT_OUTPUT_ROOT):
    os.makedirs(EXPERIMENT_OUTPUT_ROOT, exist_ok=True)
    print(f"Created directory {EXPERIMENT_OUTPUT_ROOT}.")
    
PATH_TO_HF_TOKEN="/home/shypula/hf_token.txt"

# params: model, temperature, top_p, num_return_sequences, template
CONFIGS = [
            ['meta-llama/Meta-Llama-3-8B-Instruct', 0.5, 1.0, 30, 'open_ended_default'],
            ['meta-llama/Meta-Llama-3-8B', 0.5, 1.0, 30, 'open_ended_default'],
            
            ['meta-llama/Meta-Llama-3-8B-Instruct', 1.0, 1.0, 30, 'open_ended_default'],
            ['meta-llama/Meta-Llama-3-8B', 1.0, 1.0, 30, 'open_ended_default'],
            
            ['meta-llama/Meta-Llama-3-8B-Instruct', 1.5, 1.0, 30, 'open_ended_default'],
            ['meta-llama/Meta-Llama-3-8B', 1.5, 1.0, 30, 'open_ended_default'],
            
            ['meta-llama/Meta-Llama-3-8B-Instruct', 2.0, 1.0, 30, 'open_ended_default'],
            ['meta-llama/Meta-Llama-3-8B', 2.0, 1.0, 30, 'open_ended_default'],
            
            ['meta-llama/Meta-Llama-3-8B-Instruct', 2.5, 1.0, 30, 'open_ended_default'],
            ['meta-llama/Meta-Llama-3-8B', 2.5, 1.0, 30, 'open_ended_default'],
            
        
        # ['davinci-002', 1.0, 1.0, 20, 'open_ended_default'],
        # ['gpt-3.5-turbo-instruct', 1.0, 1.0, 20, 'open_ended_default'],
        
        # `['gpt-3.5-turbo-instruct', 0.4, 1.0, 20, 'open_ended_default'],`
        # ['gpt-3.5-turbo-instruct', 0.7, 1.0, 20, 'open_ended_default'],
        # ['gpt-3.5-turbo-instruct', 1.0, 1.0, 20, 'open_ended_default'],
        # ['gpt-3.5-turbo-instruct', 1.3, 1.0, 20, 'open_ended_default']
        
        # ['davinci-002', 0.4, 1.0, 20, 'open_ended_default'],
        # ['davinci-002', 0.7, 1.0, 20, 'open_ended_default'],
        # ['davinci-002', 1.0, 1.0, 20, 'open_ended_default'],
        # ['davinci-002', 1.3, 1.0, 20, 'open_ended_default']
        
        
    ]

def create_yaml_config(model, temperature, top_p, num_return_sequences, template, config_dir): 
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
        'repetition_penalty': 1.0, 
        'parallel_samples': 5, 
        'port': 9999, 
        'devices_list': '4,5,6,7',
        'startup_timeout': 600,
        'volume': 'saved_models',
        'generation_timeout': 100,
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
    # with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace', encoding='utf-8') as proc, \
    #      open(log_file_path, 'w', encoding='utf-8', errors='replace') as log_file:
    #     for line in proc.stdout:  # Process output line by line
    #         sys.stdout.write(line)  # Print to console
    #         log_file.write(line)  # Write to log file
    
    # with os.popen(" ".join(command), 'r', ) as proc, open(log_file_path, 'w', encoding='utf-8', errors='ignore') as log_file:
    #     for line in proc:
    #         sys.stdout.write(line)  # Print to console
    #         log_file.write(line)  # Write to log file
    command = command + ["2>&1 | tee", log_file_path]
    os.system(" ".join(command))
            
    # from inside the experiment
    #experiment_string = f"{args.model}_temp_{args.temperature}_top_p_{args.top_p}_max_length_{args.max_length}_num_return_sequences_{args.num_return_sequences}_repetition_penalty_{args.repetition_penalty}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    # experiment_output_dir = os.path.join(args.experiment_output_root, experiment_string)        
    # mean = described.loc['mean']
    # with open(os.path.join(experiment_output_dir, 'results_stats_mean.tsv'), 'w') as f:
    #     for k, v in mean.items():
    #         f.write(f"{k}\t{v}\n")
    # find the directory, the latest one (because the time will be different)
    # then read in the results_stats_mean.tsv file
    
    # read back in config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model = config['model'].replace('/', '-')
    temperature = config['temperature']
    top_p = config['top_p']
    num_return_sequences = config['num_return_sequences']
    
    
    # matching_dirs = [d for d in os.listdir(EXPERIMENT_OUTPUT_ROOT) if os.path.isdir(os.path.join(EXPERIMENT_OUTPUT_ROOT, d)) and d.startswith(f"{model}_temp_{temperature}_top_p_{top_p}_num_return_sequences_{num_return_sequences}_repetition_penalty_")]
    # if len(matching_dirs) == 0:
    #     print("No matching directories found.")
    #     return None
    dirs = [d for d in os.listdir(EXPERIMENT_OUTPUT_ROOT) if os.path.isdir(os.path.join(EXPERIMENT_OUTPUT_ROOT, d))]
    if len(dirs) == 0:
        print("No directories found.")
        # import pdb; pdb.set_trace()
        return None
    time_sorted_dirs = sorted(dirs, key=lambda x: datetime.strptime(x[-19:], '%Y-%m-%d_%H-%M-%S'), reverse=True)
    latest_dir = time_sorted_dirs[0]
    results_file = os.path.join(EXPERIMENT_OUTPUT_ROOT, latest_dir, 'results_stats_mean.tsv')
    # read in as dict
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
    
    
    

def main(configurations):
    config_dir = '../configs'
    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Configuration directory {config_dir} not found.")
    # os.makedirs(config_dir, exist_ok=True)

    logs_dir = '../logs'
    os.makedirs(logs_dir, exist_ok=True)
    
    driver_stats_file = os.path.join(EXPERIMENT_OUTPUT_ROOT, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_driver_stats.tsv")
    keys = ['model', 'temperature', 'top_p', 'num_return_sequences', 'template', 'coherence', 'semantic_count', 'distinct_1', 'distinct_2', 'distinct_3', 'distinct_4', 'distinct_5', 'distinct_6', 'corpus_self_bleu', 'plain_subtrees_3', 'plain_subtrees_4', 'plain_subtrees_5', 'plain_subtrees_6', 'stripped_subtrees_3', 'stripped_subtrees_4', 'stripped_subtrees_5', 'stripped_subtrees_6', 'obfuscated_subtrees_3', 'obfuscated_subtrees_4', 'obfuscated_subtrees_5', 'obfuscated_subtrees_6']
    with open(driver_stats_file, 'w') as f:
        f.write('\t'.join(keys) + '\n')

    for config in configurations:
        yaml_path = create_yaml_config(*config, config_dir)
        print(f"Running experiment with configuration {config}")
        results = run_experiment(yaml_path, os.path.join(logs_dir, f'log_{config[0]}_{config[1]}_{config[2]}_{config[3]}_{config[4]}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'))
        if results is not None:
            with open(driver_stats_file, 'a') as f:
                f.write('\t'.join([str(results[k]) for k in keys]) + '\n')
        

        print(f"Experiment {config} completed.")
        
    print("All experiments completed.")

if __name__ == '__main__':
    
    main(CONFIGS)
