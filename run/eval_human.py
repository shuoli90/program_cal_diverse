import os
import sys
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))
import argparse
import pandas as pd
import json
import numpy as np
from models import gpt, opensource, hf_inference
from utils import textprocessing
from utils.clustering import clustering
from utils.clustering.clustering import tqdm_joblib
import joblib
from joblib import Parallel, delayed
from utils.clustering import lexical_diversity
from utils.clustering.ast_processing import AllSubtreeAnalysis, AstSubTree, parallel_subtree_analysis
from dataclasses import dataclass
import yaml
from tqdm import tqdm
from functools import partial
from datetime import datetime
import traceback
import copy 
import shutil   

import signal
import traceback

from async_driver import Arguments, load_arguments_from_yaml, create_config, reformat_config, validate_config
from eval_driver import results_stats_keys
import logging 
import glob 
import subprocess

logging.basicConfig(level=logging.INFO)


RUN_NAME="human_directed_eval_debug_new"

MAX_PROGRAMS=-1


PATH_TO_HUMAN_RESULTS = "../data/high_solve_rate_problems/reprocessed_problem_descriptions_v9_solve_rate_0.4_n_testcases_15_sampled_100.jsonl"
# RUN_NAME="directed_debug"

ALL_EXPERIMENT_OUTPUT_ROOT = "/data1/shypula/prog_diversity/all_experiments/"

if not os.path.exists(ALL_EXPERIMENT_OUTPUT_ROOT):
    os.makedirs(ALL_EXPERIMENT_OUTPUT_ROOT, exist_ok=True)
    print(f"Created directory {ALL_EXPERIMENT_OUTPUT_ROOT}.")

MOCK_CONFIG = ['human_outputs/human_outputs', 1.0, 1.0, 100, 'open_ended_default', 25]


def main(config): 

    ### Copied from async_driver.py
    run_name = "Run" if RUN_NAME == "" else RUN_NAME
    this_driver_root = os.path.join(ALL_EXPERIMENT_OUTPUT_ROOT, f"{run_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    logs_dir = os.path.join(this_driver_root, 'driver_logs')
    
    os.makedirs(this_driver_root, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Created directory {this_driver_root}/")
    
    assert validate_config(config), "Invalid configuration."


    full_config = create_config(*config, driver_root=this_driver_root)
    full_config["is_directed"] = True
    full_config["eval_workers"] = 30
    full_config["max_programs"] = MAX_PROGRAMS
    
    experiment_name = full_config['experiment_name']
    config_path = os.path.join(this_driver_root, f'{experiment_name}.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(full_config, f)
    
    with open(os.path.join(this_driver_root, 'all_configs_list.txt'), 'w') as f:
        f.write(config_path)
        
    ### End of copy from async_driver.py
    experiment_output_dir = full_config["experiment_output_dir"]
    os.makedirs(experiment_output_dir, exist_ok=True)
    # copy the human df to results.jsonl 
    shutil.copy(PATH_TO_HUMAN_RESULTS, os.path.join(experiment_output_dir, 'results.jsonl'))
    
    logging.info(f"Running eval_driver for {this_driver_root}")
    logging.info(f"Running 'python eval_driver.py {this_driver_root}'")
    p = subprocess.run(["python", "eval_driver.py", this_driver_root])
    if p.returncode != 0:
        logging.error(f"Error in {this_driver_root}, return code {p.returncode}")    
    else:
        logging.info(f"Experiment {this_driver_root} completed with return code 0")
    
    
if __name__ == "__main__":
    main(MOCK_CONFIG)
        