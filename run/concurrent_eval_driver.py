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
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import threading


# make the keys for the results
base_keys = ['model', 'template', 'temperature', 'top_p', 'num_return_sequences']  
# result[f'{recordtype}_distinct_{i}_raw'] = distinct_n_raw

results_stats_keys = ['coherence', 'semantic_count', 'semantic_proportion', 'accuracy']
results_stats_keys = results_stats_keys + ['semantic_count_wcoh', 'semantic_proportion_wcoh', 'semantic_count_wcoh_nonempty', 'semantic_proportion_wcoh_nonempty', 'semantic_count_wcoh_nonempty_woutput', 'semantic_proportion_wcoh_nonempty_woutput']
results_stats_keys += ['pairwise_semantic_prop', 'pairwise_semantic_prop_wcoh', 'pairwise_semantic_prop_wcoh_nonempty', 'pairwise_semantic_prop_wcoh_nonempty_woutput']
# result[f'{recordtype}_average_cosine_distance_programs_zero_null'] = average_cosine_distance_zero_null
results_stats_keys = results_stats_keys + ['average_cosine_distance_programs', 'average_cosine_distance_raw'] 
results_stats_keys = results_stats_keys + ['average_cosine_distance_programs_zero_null', 'average_cosine_distance_raw_zero_null', 'average_cosine_distance_programs_one_null', 'average_cosine_distance_raw_one_null']
results_stats_keys = results_stats_keys + [f"distinct_{i}" for i in range(1, 7)] + [f"distinct_{i}_no_comments" for i in range(1, 7)] + [f"distinct_{i}_raw" for i in range(1, 7)] 
results_stats_keys = results_stats_keys + [f"distinct_{i}_bootstrap" for i in range(1, 7)] + [f"distinct_{i}_no_comments_bootstrap" for i in range(1, 7)] + [f"distinct_{i}_raw_bootstrap" for i in range(1, 7)]
results_stats_keys = results_stats_keys + [f"distinct_{i}_jaccard" for i in range(1, 7)]
results_stats_keys = results_stats_keys + [f"{key}_{height}" for key in ['plain_subtrees', 'stripped_subtrees'] for height in [3,4,5,6]]
# boostrap keys
results_stats_keys = results_stats_keys + [f"{key}_{height}_bootstrap" for key in ['plain_subtrees', 'stripped_subtrees'] for height in [3,4,5,6]]
results_stats_keys = results_stats_keys + [f"{key}_{height}_jaccard" for key in ['plain_subtrees', 'stripped_subtrees'] for height in [3,4,5,6]]
# results_stats_keys = [f"{recordtype}_{key}" for recordtype in ['all', 'coh', 'err', 'acc', 'inacc'] for key in results_stats_keys]
results_stats_keys = [f"{recordtype}_{key}" for recordtype in ['all', 'coh', 'syn', 'err', 'acc', 'inacc'] for key in results_stats_keys]
results_stats_keys.insert(4, 'coh_semantic_proportion_of_all')

all_keys = base_keys + results_stats_keys

pretty_column_widths = [46, 15] + [(len(k) + 2) for k in all_keys[2:]]

MAX_WORKERS=5


def parse_results(results_dir: str): 
    results_file = os.path.join(results_dir, 'results_stats_mean.tsv')
    # read in as dict
    try: 
        with open(results_file, 'r') as f:
            lines = f.readlines()
        results = {}
        for line in lines:
            k, v = line.strip().split('\t')
            if "semantic_count" in k:
                results[k] = str(round(float(v), 2))
            elif k in results_stats_keys:
                results[k] = str(round(float(v) * 100, 2))
            else:
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
    pretty_column_widths = [46, 15] + [max(len(k) + 2, 6) for k in all_keys[2:]]
    
    with open(stats_file, 'w') as f:
        f.write('\t'.join(all_keys) + '\n')
    
    with open(stats_pretty_file, 'w') as f:
    # Writing column headers with fixed width formatting
        f.write(''.join([f"{k.ljust(pretty_column_widths[i])}" for i, k in enumerate(all_keys)]) + '\n')
    return stats_file, stats_pretty_file

        
def write_out_results(results: dict, config: Arguments, stats_file: str, stats_pretty_file: str, is_error=False):
    results["model"] = config.model
    results["template"] = config.template.replace("open_ended_", "").replace("directed_", "")
    results["temperature"] = config.temperature
    results["top_p"] = config.top_p
    results["num_return_sequences"] = config.num_return_sequences
    
    if is_error: 
        for key in all_keys:
            if key not in results:
                results[key] = "ERROR"
    
    for key in all_keys:
        if key not in results:
            results[key] = "NOT_FOUND"
            
    with open(stats_file, 'a') as f:
        f.write('\t'.join([str(results[k]) for k in all_keys]) + '\n')
        
    with open(stats_pretty_file, 'a') as f:
        f.write(''.join([f"{str(results[k]).ljust(pretty_column_widths[i])}" for i, k in enumerate(all_keys)]) + '\n')


def run_experiment_and_log_results(config_path, config, stats_file, stats_pretty_file, lock):
    directory = config.experiment_output_dir
    error_path = os.path.join(directory, 'error.txt')
    result_path = os.path.join(directory, 'results.jsonl')

    if os.path.exists(result_path):
        # os.system(f"python experiment_eval.py {config_path}")
        # subprocess.Popen(["python", "experiment_eval.py", config_path])
        subprocess.call(["python", "experiment_eval.py", config_path])
        result = parse_results(directory)
        with lock:
            write_out_results(result, config, stats_file, stats_pretty_file, is_error=False)
    elif os.path.exists(error_path):
        with open(error_path, 'r') as f:
            error = f.read()
        logging.error(f"Error in {directory}: {error}")
        with lock:
            write_out_results({}, config, stats_file, stats_pretty_file, is_error=True)


def monitor_directories_and_run(configs_paths, experiment_directory):
    config_path_to_config = {config_path: load_arguments_from_yaml(config_path) for config_path in configs_paths}
    stats_file, stats_pretty_file = init_results_file(experiment_directory)
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while configs_paths:
            for config_path in list(configs_paths):  # Use list to avoid modification during iteration
                config = config_path_to_config[config_path]
                directory = config.experiment_output_dir
                result_path = os.path.join(directory, 'results.jsonl')
                error_path = os.path.join(directory, 'error.txt')

                if os.path.exists(result_path) or os.path.exists(error_path):
                    executor.submit(run_experiment_and_log_results, config_path, config, stats_file, stats_pretty_file, lock)
                    configs_paths.remove(config_path)
            
            time.sleep(10)  # Adjust sleep time based on expected update frequency
    
# def monitor_directories_and_run(configs_paths: List[str], experiment_directory): 
#     """Monitor a list of directories for the existence of results.jsonl."""
#     # TODO: error handling (what if the experiment failed)
#     config_path_to_config = {config_path: load_arguments_from_yaml(config_path) for config_path in configs_paths}
#     stats_file, stats_pretty_file = init_results_file(experiment_directory)
#     consecutive_sleeps = 0
#     while configs_paths: # terminates when configs_paths  == []
#         for config_path in configs_paths: 
#             config = config_path_to_config[config_path]
            
#             directory = config.experiment_output_dir
            
#             if os.path.exists(os.path.join(directory, 'results.jsonl')):
#                 # logging.info(f"Running `python experiment_eval.py {config_path}`")
#                 os.system(f"python experiment_eval.py {config_path}")
                
                
#                 result = parse_results(directory)
#                 write_out_results(result, config, stats_file, stats_pretty_file, is_error=False)
#                 configs_paths.remove(config_path)
#                 logging.info(f"Finished {directory}, {len(configs_paths)} remaining.")
                
#             elif os.path.exists(os.path.join(directory, 'error.txt')):
#                 logging.info(f"Error in {directory}")
#                 with open(os.path.join(directory, 'error.txt'), 'r') as f:
#                     error = f.read()
#                 logging.error(f"Error in {directory}: {error}")
#                 write_out_results({}, config, stats_file, stats_pretty_file, is_error=True)
#                 configs_paths.remove(config_path)
                
#         if configs_paths:  # Only sleep if there are still directories to check
#             time.sleep(15)
#             if (consecutive_sleeps % 20) == 0:
#                 logging.info(f"Waiting for {len(configs_paths)} directories to finish...")
#             consecutive_sleeps += 1
            
# Example usage
if __name__ == "__main__":
    experiment_directory = sys.argv[1]
    logging.basicConfig(level=logging.INFO)
    # log to experiment_directory/driver_logs   
    log_file = os.path.join(experiment_directory, "driver_logs", f"eval_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(level=logging.INFO, 
                        handlers=[
                            logging.FileHandler(log_file),  # File handler
                            logging.StreamHandler(sys.stdout)  # Console handler
                        ], 
                        force=True)
    logging.info(f"Starting evaluation driver for {experiment_directory}")

    with open(os.path.join(experiment_directory, "all_configs_list.txt"), "r") as f:
        all_configs_paths = f.readlines()
    all_configs_paths = [p.strip() for p in all_configs_paths]
    
    assert isinstance(all_configs_paths, list), "all_configs must be a list"
    newline = "\n"
    assert all(os.path.exists(config_path) for config_path in all_configs_paths), f"all paths must exist, {newline.join([p for p in all_configs_paths if not os.path.exists(p)])}\ndidn't exist"

    # load all the configs
    ## TODO: summarize results to the summary file as per previous driver
    monitor_directories_and_run(all_configs_paths, experiment_directory)
    # 