### takes in the directory of the experiment 
### looks for the file with paths


import os
import sys
import time 
import logging 
import datetime
sys.path.append(os.path.dirname(__file__))
# import the arguments class
from async_driver import Arguments, load_arguments_from_yaml
from typing import List


# make the keys for the results
base_keys = ['model', 'template', 'temperature', 'top_p', 'num_return_sequences']  

results_stats_keys = ['coherence', 'semantic_count', 'semantic_proportion', 'distinct_1', 'distinct_2', 'distinct_3', 'distinct_4', 'distinct_5', 'distinct_6']
results_stats_keys = results_stats_keys + [f"{key}_{height}" for key in ['plain_subtrees', 'stripped_subtrees'] for height in [3,4,5,6]]
results_stats_keys = [f"{recordtype}_{key}" for recordtype in ['all', 'coh', 'err'] for key in results_stats_keys]
results_stats_keys.insert(4, 'coh_semantic_proportion_of_all')

all_keys = base_keys + results_stats_keys

pretty_column_widths = [46, 15] + [(len(k) + 2) for k in all_keys[2:]]


def parse_results(results_dir: str): 
    results_file = os.path.join(results_dir, 'results_stats_mean.tsv')
    # read in as dict
    try: 
        with open(results_file, 'r') as f:
            lines = f.readlines()
        results = {}
        for line in lines:
            k, v = line.strip().split('\t')
            if k in results_stats_keys:
                results[k] = round(float(v) * 100, 2)
            elif "semantic_count" in k:
                results[k] = round(float(v), 2)
            else:
                results[k] = v
            results[k] = v
        return results
    except Exception as e:
        print(f"Error reading in results file: {e}")
        return None
    
        # # coherence	semantic_count	distinct_1	distinct_2	distinct_3	distinct_4	distinct_5	distinct_6	corpus_self_bleu	plain_subtrees_3	plain_subtrees_4	plain_subtrees_5	plain_subtrees_6	stripped_subtrees_3	stripped_subtrees_4	stripped_subtrees_5	stripped_subtrees_6	obfuscated_subtrees_3	obfuscated_subtrees_4	obfuscated_subtrees_5	obfuscated_subtrees_6
        # results["model"] = model
        # print(f"Results for experiment {model}_temp_{temperature}_top_p_{top_p}_num_return_sequences_{num_return_sequences}:")
        # print(results)
        
def init_results_file(results_dir: str):
    stats_file = os.path.join(results_dir, 'driver_stats.tsv')
    stats_pretty_file = os.path.join(results_dir, 'driver_stats_pretty.tsv')
    pretty_column_widths = [46, 15] + [max(len(k) + 2, 6) for k in keys[2:]]
    
    with open(stats_file, 'w') as f:
        f.write('\t'.join(all_keys) + '\n')
    
    with open(stats_pretty_file, 'w') as f:
    # Writing column headers with fixed width formatting
        f.write(''.join([f"{k.ljust(pretty_column_widths[i])}" for i, k in enumerate(all_keys)]) + '\n')
    return stats_file, stats_pretty_file
        
        
def write_out_results(results: dict, config: Arguments, stats_file: str, stats_pretty_file: str):
    results["model"] = config.model
    results["template"] = config.template.replace("open_ended_", "")
    results["temperature"] = config.temperature
    results["top_p"] = config.top_p
    results["num_return_sequences"] = config.num_return_sequences
    
    for key in all_keys:
        if key not in results:
            results[key] = "NOT_FOUND"
            
    with open(stats_file, 'a') as f:
        f.write('\t'.join([str(results[k]) for k in all_keys]) + '\n')
        
    with open(stats_pretty_file, 'a') as f:
        f.write(''.join([f"{str(results[k]).ljust(pretty_column_widths[i])}" for i, k in enumerate(all_keys)]) + '\n')


def monitor_directories_and_run(configs: List[Arguments]): 
    """Monitor a list of directories for the existence of results.jsonl."""
    # TODO: error handling (what if the experiment failed)
    stats_file, stats_pretty_file = init_results_file(configs[0].experiment_output_root)
    consecutive_sleeps = 0
    while configs:
        for config in configs: 
            directory = config.experiment_output_dir
            
            if os.path.exists(os.path.join(directory, 'results.jsonl')):
                logging.info(f"Running eval.py on {directory}")
                os.system(f"python experiment_eval.py {directory}")
                result = parse_results(directory)
                write_out_results(result, config, stats_file, stats_pretty_file)
                configs.remove(config)
                
            elif os.path.exists(os.path.join(directory, 'error.txt')):
                logging.info(f"Error in {directory}")
                with open(os.path.join(directory, 'error.txt'), 'r') as f:
                    error = f.read()
                logging.error(f"Error in {directory}: {error}")
                configs.remove(config)
                
        if configs:  # Only sleep if there are still directories to check
            time.sleep(15)
            if (consecutive_sleeps % 20) == 0:
                logging.info(f"Waiting for {len(configs)} directories to finish...")
            consecutive_sleeps += 1
            


# Example usage
if __name__ == "__main__":
    experiment_directory = sys.argv[1]
    logging.basicConfig(level=logging.INFO)
    # log to experiment_directory/driver_logs   
    log_file = os.path.join(experiment_directory, "driver_logs", f"eval_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=False)
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info(f"Starting evaluation driver for {experiment_directory}")

    with open(os.path.join(experiment_directory, "all_configs_list.txt"), "r") as f:
        all_configs_paths = f.readlines()
    assert isinstance(all_configs_paths, list), "all_configs must be a list"
    assert all(os.path.exists(config_path) for config_path in all_configs_paths), "all paths must exist"
    all_configs = [load_arguments_from_yaml(config_path) for config_path in all_configs_paths]
    # load all the configs
    ## TODO: summarize results to the summary file as per previous driver
    monitor_directories_and_run(all_configs)
    