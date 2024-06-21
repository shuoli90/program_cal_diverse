import os
import sys
import yaml
import subprocess
from datetime import datetime
import sys 
import logging


def main(experiment_driver_root):
    assert os.path.exists(os.path.join(experiment_driver_root, 'all_configs_list.txt')), f"all_configs_list.txt not found in {experiment_driver_root}"
    with open(os.path.join(experiment_driver_root, 'all_configs_list.txt'), 'r') as f:
        config_paths = [line.strip() for line in f.readlines()]
        
    logging.basicConfig(level=logging.INFO)
    log_file = os.path.join(experiment_driver_root, "driver_logs", f"generation_driver_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info(f"Starting generation driver for {experiment_driver_root}")
    
    for config_path in config_paths:
        logging.info(f"Running experiment with configuration {config_path}")
        p = subprocess.run(["python", "experiment_generate.py", config_path])
        if p.returncode != 0:
            assert os.path.exists(os.path.join(experiment_driver_root, 'error.txt')), f"error.txt not found in {experiment_driver_root} despite return code {p.returncode}"
            logging.error(f"Error in {config_path}")
            logging.error(f"Error code: {p.returncode}")
        else:
            assert os.path.exists(os.path.join(experiment_driver_root, 'results.jsonl')), f"results.jsonl not found in {experiment_driver_root}, despite return code 0"
            logging.info(f"Experiment {config_path} completed.")
        

if __name__ == '__main__':
    experiment_driver_root = sys.argv[1]
    main(experiment_driver_root)
