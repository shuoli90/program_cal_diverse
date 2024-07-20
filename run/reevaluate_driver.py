
import os
import sys
import yaml
import subprocess
from datetime import datetime
import time
from dataclasses import dataclass
import subprocess
import shutil 
import logging 
import glob 
import re
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


### we will have a list of the directories we want to merge for re-evaluation 

ALL_EXPERIMENT_OUTPUT_ROOT = "/data1/shypula/prog_diversity/all_experiments/"

RUN_NAME = "Open_Ended_Reevaluation_Cosine_Bootstrap"

DIRECTORY_PATHS=[
    # "/data1/shypula/prog_diversity/all_experiments/Model_Prompt_Temp_Sweep_2024-06-21_22-18-19", 
    # "/data1/shypula/prog_diversity/all_experiments/Model_Prompt_Temp_Sweep_2024-06-22_09-57-15", 
    # "/data1/shypula/prog_diversity/all_experiments/Model_Prompt_Temp_Sweep_2024-06-25_11-46-30", 
    # "/data1/shypula/prog_diversity/all_experiments/Model_Prompt_Temp_Sweep_2024-06-27_01-07-28", 
    # "/data1/shypula/prog_diversity/all_experiments/Model_Prompt_Temp_Sweep_2024-06-27_01-21-33"
    "/data1/shypula/prog_diversity/all_experiments/Model_Prompt_Temp_Sweep_2024-07-08_18-50-16", 
    "/data1/shypula/prog_diversity/all_experiments/Open_Ended_Reevaluation_2024-07-05_00-53-22"
    
]

# REEXECUTE = False
# use_previous_executions
USE_PREVIOUS_EXECUTIONS = False
REFORMAT_RESULTS = True

EXTRA_VERBOSE = False

FIX_OLD_FORMAT = False

EVAL_WORKERS = 20

def capture_i_and_coh_j(text):
    # Pattern to capture 'i' and the 'coh_j...' part as separate groups
    pattern = r"output_record_(\d+)_(.*)"
    
    # Find all matches in the text
    matches = re.findall(pattern, text)
    
    # If matches found, return the list of tuples (i, 'coh_j...')
    if matches and len(matches) == 1 and len(matches[0]) == 2:
        i, coh_j = matches[0]
        return int(i), coh_j
    else:
        logging.critical(f"Error in parsing the output_record: {text}")
        return 99, "coh_99"


def main(DIRECTORY_PATHS): 
    
    # Create the directory for the re-evaluation
    experiment_name = f"{RUN_NAME}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    experiment_output_root = os.path.join(ALL_EXPERIMENT_OUTPUT_ROOT, experiment_name)
    
    ### Step 1: Copy in 
    
    # initialize a list of the .yaml paths 
    
    all_dir_yaml_paths = [line.strip() for directory_path in DIRECTORY_PATHS for line in open(os.path.join(directory_path, 'all_configs_list.txt'), 'r').readlines()]
    pbar = tqdm(total=len(all_dir_yaml_paths))
    
    yaml_paths = []
    for directory_path in DIRECTORY_PATHS:
        
        # we can merge 1.1 and 1.2 together
        ## Step 1.1 
        # go through the directory, and get all the sub-directories 
        # except driver_logs; and copy those in 
        
        ## Step 1.2 
        # copy in also each yaml file; but also we need to change the following fields
        # experiment_output_dir: str
        # experiment_output_root: str 
        
        ## we can begin with the all_configs_list.txt file
        
        with open(os.path.join(directory_path, 'all_configs_list.txt'), 'r') as f:
            this_dir_yaml_paths = [line.strip() for line in f.readlines()]
            
        for yaml_path in this_dir_yaml_paths: 
            ## open the yaml file
            with open(yaml_path, 'r') as f: 
                yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
                
            this_experiment_output_dir = yaml_dict["experiment_output_dir"]
            this_experiment_output_root = yaml_dict["experiment_output_root"]
            this_experiment_id = os.path.basename(this_experiment_output_dir)
            assert this_experiment_output_root == directory_path, f"experiment_output_root is not the same as the directory path, something weird is going on: {this_experiment_output_root} != {directory_path}"
            if not this_experiment_output_dir == os.path.join(directory_path, this_experiment_id): 
                import pdb; pdb.set_trace()
                # find the string differenc 
                for i, (c1, c2) in enumerate(zip(this_experiment_output_dir, os.path.join(directory_path, this_experiment_id))): 
                    if c1 != c2: 
                        print(f"Index {i}: {c1} != {c2}")
                pdb.set_trace()
            assert this_experiment_output_dir == os.path.join(directory_path, this_experiment_id), f"experiment_output_dir is not the same as the directory path, something weird is going on: {this_experiment_output_dir} != {os.path.join(directory_path, this_experiment_id)}"
            
            ## only copy over if there is a results.jsonl file
            if not os.path.exists(os.path.join(this_experiment_output_dir, 'results.jsonl')): 
                logging.info(f"Skipping {yaml_path} because results.jsonl does not exist")
                pbar.update(1)
                continue
            
            # copy the directory over
            if os.path.exists(os.path.join(experiment_output_root, this_experiment_id)): 
                logging.critical(f"Experiment {this_experiment_id} already exists in {experiment_output_root}")
                pbar.update(1)
                continue
                # raise ValueError(f"Experiment {this_experiment_id} already exists in {experiment_output_root}")
            
            new_experiment_output_dir = os.path.join(experiment_output_root, this_experiment_id)
            shutil.copytree(this_experiment_output_dir, new_experiment_output_dir)
            
            yaml_dict["experiment_output_dir"] = new_experiment_output_dir
            yaml_dict["experiment_output_root"] = experiment_output_root
            
            ## override the old re-execute and reformat_results
            yaml_dict["use_previous_executions"] = USE_PREVIOUS_EXECUTIONS
            yaml_dict["reformat_results"] = REFORMAT_RESULTS
            yaml_dict["eval_workers"] = EVAL_WORKERS
            
            # remove the old results + config 
            
            # old_config_path = os.path.join(new_experiment_output_dir, 'config.yaml')
            # old_results_stats_path = os.path.join(new_experiment_output_dir, 'results_stats.tsv')
            # old_results_stats_mean_path = os.path.join(new_experiment_output_dir, 'results_stats_mean.tsv')
            # old_eval_log_path = os.path.join(new_experiment_output_dir, 'eval.log')
            old_paths = [os.path.join(new_experiment_output_dir, basename) for basename in ['config.yaml', 'results_stats.tsv', 'results_stats_mean.tsv', 'eval.log']]
            for old_path in old_paths:
                if os.path.exists(old_path):
                    if EXTRA_VERBOSE:
                        logging.info(f"Removing {old_path}")
                    os.remove(old_path)
                else: 
                    if EXTRA_VERBOSE:
                        logging.warning(f"Did not find {old_path}")
            
            ## Optional TODO: we could also replace other yaml params here if we wanted
                        
            # save the new config file to the experiment directory and the root, add to the yaml paths 
            new_yaml_path = os.path.join(new_experiment_output_dir, 'config.yaml')
            new_root_yaml_path = os.path.join(experiment_output_root, f"{this_experiment_id}.yaml")
            
            with open(new_yaml_path, 'w') as f:
                yaml.dump(yaml_dict, f)
        
            with open(new_root_yaml_path, 'w') as f:
                yaml.dump(yaml_dict, f)
            
            yaml_paths.append(new_yaml_path)

            ## remove results.tsv of each directory
            
            ## if we have old-format, create new format -> save the output_records in a generation-specific directory
            
            new_experiment_problem_id_dirs = glob.glob(os.path.join(new_experiment_output_dir, "problem_*"))
            for new_experiment_problem_id_dir in new_experiment_problem_id_dirs: 
                if os.path.exists(os.path.join(new_experiment_problem_id_dir, 'results.tsv')): 
                    if EXTRA_VERBOSE:
                        logging.info(f"Removing {os.path.join(new_experiment_problem_id_dir, 'results.tsv')}")
                    os.remove(os.path.join(new_experiment_problem_id_dir, 'results.tsv'))
                
                if FIX_OLD_FORMAT: 
                    new_experiments_records = glob.glob(os.path.join(new_experiment_problem_id_dir, "output_record_*"))
                    ## if any of these exist it is old format
                    for new_experiment_record in new_experiments_records:
                        basename = os.path.splitext(os.path.basename(new_experiment_record))[0]
                        # output_record_{i}_{suffix} 
                        gen_index, suffix = capture_i_and_coh_j(basename)
                        # create new directory generation_3_coh_0.0
                        new_experiment_dir = os.path.join(new_experiment_problem_id_dir, f"generation_{gen_index}_{suffix}")
                        os.makedirs(new_experiment_dir, exist_ok=True)
                        # copy to output_record.json
                        shutil.copy(new_experiment_record, os.path.join(new_experiment_dir, "output_record.json"))
                        
            # update the progress bar
            pbar.update(1)
            
    with open(os.path.join(experiment_output_root, 'all_configs_list.txt'), 'w') as f:
        f.write('\n'.join(yaml_paths))
    
    # let's do a check on the configs
    
    # invariants: 
    # 1. for each yaml path 
        # i. the yaml exists
        # ii. the experiment_output_dir exists
        # iii. the experiment_output_root exists
        # iv. The experiment output_dir is a subdirectory of the experiment_output_root
        # v. the experiment_output_dir contains a results.jsonl file
    logging.info("Checking invariants")
    for yaml_path in yaml_paths: 
        with open(yaml_path, 'r') as f: 
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        
        this_experiment_output_dir = yaml_dict["experiment_output_dir"]
        this_experiment_output_root = yaml_dict["experiment_output_root"]
        
        assert os.path.exists(yaml_path), f"yaml_path {yaml_path} does not exist"
        assert yaml_path.startswith(experiment_output_root), f"yaml_path {yaml_path} does not start with experiment_output_root {experiment_output_root}"
        assert os.path.exists(this_experiment_output_dir), f"experiment_output_dir {this_experiment_output_dir} does not exist"
        assert os.path.exists(this_experiment_output_root), f"experiment_output_root {this_experiment_output_root} does not exist"
        assert os.path.exists(os.path.join(this_experiment_output_dir, 'results.jsonl')), f"results.jsonl does not exist in {this_experiment_output_dir}"
        assert this_experiment_output_dir.startswith(this_experiment_output_root), f"experiment_output_dir {this_experiment_output_dir} does not start with experiment_output_root {this_experiment_output_root}"
        assert this_experiment_output_root == experiment_output_root, f"experiment_output_root {this_experiment_output_root} does not match experiment_output_root {experiment_output_root}"
        
    logging.info("All invariants satisfied")
        
    pbar.close()
    
    p = subprocess.run(["du", '-h', experiment_output_root, '--max-depth=0'], capture_output=True)
    logging.info(f"Experiment output root size: {p.stdout.decode().strip()}")
    
    
        
                                   
    
    ## Step 2: re-run the eval_driver
    logging.info(f"Running eval_driver for {experiment_output_root}")
    logging.info(f"Running 'python eval_driver.py {experiment_output_root}'")
    p = subprocess.run(["python", "eval_driver.py", experiment_output_root])
    if p.returncode != 0:
        logging.error(f"Error in {experiment_output_root}, return code {p.returncode}")    
    else:
        logging.info(f"Experiment {experiment_output_root} completed with return code 0")
    
if __name__ == '__main__':
    main(DIRECTORY_PATHS)