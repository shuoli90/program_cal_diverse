import os
import sys
import yaml
import subprocess
from datetime import datetime
import sys 
import logging

import sys 
from async_driver import Arguments, load_arguments_from_yaml

RUN_DIR = os.path.dirname(os.path.abspath(__file__))


def main(experiment_driver_root):
    assert os.path.exists(os.path.join(experiment_driver_root, 'all_configs_list.txt')), f"all_configs_list.txt not found in {experiment_driver_root}"
    with open(os.path.join(experiment_driver_root, 'all_configs_list.txt'), 'r') as f:
        config_paths = [line.strip() for line in f.readlines()]
        
    logging.basicConfig(level=logging.INFO)
    log_file = os.path.join(experiment_driver_root, "driver_logs", f"generation_driver_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    # logging.basicConfig(filename=log_file, level=logging.INFO, force=True)
    
    logging.basicConfig(level=logging.INFO, 
                        handlers=[
                                logging.FileHandler(log_file),  # File handler
                                logging.StreamHandler(sys.stdout)  # Console handler
                            ], 
                        force=True)
    logging.info(f"Starting generation driver for {experiment_driver_root}")
    
    for config_path in config_paths:
        arguments = load_arguments_from_yaml(config_path)
        experiment_output_dir = arguments.experiment_output_dir
        logging.info(f"Running experiment with configuration {config_path} at {experiment_output_dir}")
        logging.info(f"Running 'python experiment_generate.py {config_path}'")
        p = subprocess.run(["python", "experiment_generate.py", config_path], cwd=RUN_DIR)
        if p.returncode != 0:
            logging.error(f"Error in {config_path}, return code {p.returncode}")
            assert os.path.exists(os.path.join(experiment_output_dir, 'error.txt')), f"error.txt not found in {experiment_output_dir} despite return code {p.returncode}"
            logging.error(f"Error in {config_path}")
            logging.error(f"Error code: {p.returncode}")
        else:
            assert os.path.exists(os.path.join(experiment_output_dir, 'results.jsonl')), f"results.jsonl not found in {experiment_output_dir}, despite return code 0"
            logging.info(f"Experiment {config_path} completed with return code 0")
        

if __name__ == '__main__':
    experiment_driver_root = sys.argv[1]
    main(experiment_driver_root)
