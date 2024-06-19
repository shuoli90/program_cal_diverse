### takes in the directory of the experiment 
### looks for the file with paths


import os
import sys
import time 
import logging 
import datetime

def monitor_directories_and_run(directories):
    """Monitor a list of directories for the existence of results.jsonl."""
    consecutive_sleeps = 0
    while directories:
        for directory in directories:
            if os.path.exists(os.path.join(directory, 'results.jsonl')):
                logging.info(f"Running eval.py on {directory}")
                os.system(f"python experiment_eval.py {directory}")
                directories.remove(directory)
                
        if directories:  # Only sleep if there are still directories to check
            time.sleep(15)
            if (consecutive_sleeps % 20) == 0:
                logging.info(f"Waiting for {len(directories)} directories to finish...")
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
        all_configs = f.readlines()
    assert isinstance(all_configs, list), "all_configs must be a list"
    assert all(os.path.exists(config_path) for config_path in all_configs), "all paths must exist"
    ## TODO: summarize results to the summary file as per previous driver
    monitor_directories_and_run(all_configs)
    