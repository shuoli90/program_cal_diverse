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
import datetime
import traceback
import logging
from transformers import AutoTokenizer

import signal
import traceback



def handler(pipe, experiment_output_dir, signum, frame):
    print("Signal received, handling cleanup...")
    try:
        print("Stopping and removing the service...")
        with open(os.path.join(experiment_output_dir, 'error.txt'), 'w') as f:
            f.write("Signal received, stopping and removing the service.")
        pipe.stop_service()
        pipe.remove_service()
        print("Cleanup complete, exiting...")
    except Exception as e:
        print("Error during cleanup:", e)
        traceback.print_exc()
    finally:
        sys.exit(0)  # Ensure the process exits

from functools import partial
import sys 
# add dirname of this file to the path
sys.path.append(os.path.dirname(__file__))
# import the arguments class
from async_driver import Arguments, load_arguments_from_yaml

# def partial_handler(pipe, args): 
#     return partial(handler, pipe, args)


# @dataclass
# class Arguments:
#     experiment_id: str 
#     experiment_output_dir: str
#     experiment_output_root: str 
#     path_to_dataset: str = '../data/open_ended/open_ended_final/dataset.jsonl'
#     model: str = 'gpt-3.5-turbo'
#     template: str = 'open_ended_default'
#     temperature: float = 1.0
#     top_p: float = 1.0
#     max_length: int = 768
#     num_return_sequences: int = 10
#     repetition_penalty: float = 1.0
#     parallel_samples: int = 5
#     port: int = 9999
#     devices_list: str = '4,5,6,7'
#     startup_timeout: int = 600
#     generation_timeout: int = 100
#     volume: str = 'saved_models'
#     path_to_hf_token: str = None
#     batch_size: int = None
#     max_programs: int = -1
    
    
template_dir = os.path.join(os.path.dirname(__file__), "../prompt_templates")
templates = [f for f in os.listdir(template_dir) if f.endswith('.txt')]
template_names = [f.split('.')[0] for f in templates] + [None]
    
def readin_template(template_arg): 
    assert template_arg in template_names, f"Template {template_arg} not found. Available templates: {template_names}"
    if template_arg == None: 
        return "## DESCRIPTION\n"
    else: 
        path_to_template = os.path.join(os.path.dirname(__file__), f"../prompt_templates/{template_arg}.txt")
        with open(path_to_template, 'r') as file:
            template = file.read()
        return template
    

def _format_template(prompt, template: str): 
    formatted_prompt = template.replace("## DESCRIPTION", prompt)
    assert "## DESCRIPTION" not in formatted_prompt, "Template not formatted correctly"
    return formatted_prompt


# def load_arguments_from_yaml(yaml_file):
#     with open(yaml_file, 'r') as file:
#         args_dict = yaml.safe_load(file)
#     return Arguments(**args_dict)


if __name__ == '__main__':
    path_to_yaml = sys.argv[1]
    args = load_arguments_from_yaml(path_to_yaml)
    
    # make sure the template is valid
    prompt_template = readin_template(args.template)
    format_template_fun = partial(_format_template, template=prompt_template)
    
    model_name_clean = args.model.replace("/", "-")
    # experiment_string = f"{model_name_clean}_temp_{args.temperature}_top_p_{args.top_p}_max_length_{args.max_length}_num_return_sequences_{args.num_return_sequences}_repetition_penalty_{args.repetition_penalty}_{args.template}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    # experiment_id= args.experiment_id
    experiment_name = args.experiment_name
    experiment_output_dir = args.experiment_output_dir
    is_directed = args.is_directed
    
    os.makedirs(experiment_output_dir, exist_ok=True) # there should be no existing directory, but maybe for re-running
    
    logs_file = os.path.join(experiment_output_dir, 'generation_log.txt')
    # logging.basicConfig(filename=logs_file, level=logging.INFO, force=True)
    logging.basicConfig(level=logging.INFO, 
                        handlers=[
                            logging.FileHandler(logs_file),  # File handler
                            logging.StreamHandler(sys.stdout)  # Console handler
                        ], 
                        force=True
    )
    logging.info(f"Starting generations for {experiment_name}")
    
    # save config
    with open(os.path.join(experiment_output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)
        
    if "tatsu" or "codellama" in args.model.lower():
        with open(args.path_to_hf_token, "r") as f:
            hf_key = f.read().strip()
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_key)
    else: 
        tokenizer = None
        
    pipe = None # for error handling
    try:                                                
        # Setup generation pipeline
        if 'gpt' in args.model or 'davinci' in args.model:
            pipe = gpt.GPTModel(model_name=args.model)
        else:
            # pipe = opensource.OpensourceModel(model_name=args.model)
            with open(args.path_to_hf_token, "r") as f:
                hf_key = f.read().strip()
            logging.info(f"Starting HF Inference Service with model {args.model}")
            pipe = hf_inference.HFInferenceService(model_name=args.model, 
                                                    parallel_samples=max(args.parallel_samples,args.num_return_sequences),
                                                    port=args.port,
                                                    devices_list=args.devices_list,
                                                    volume=args.volume,
                                                    startup_timeout=args.startup_timeout,
                                                    generation_timeout=args.generation_timeout,
                                                    hf_key=hf_key)
            # sigint_handler = partial_handler(pipe)
            sigint_handler = partial(handler, pipe, experiment_output_dir)
            signal.signal(signal.SIGINT, sigint_handler)
                                             
        # Read in data
        print(f'reading in data from {args.path_to_dataset}')
        df = pd.read_json(args.path_to_dataset, lines=True, orient='records')
        # if is_directed and "val" in args.path_to_dataset:
        #     # data/high_solve_rate_problems/val_longer_code_problem_ids.txt
        #     path_to_problem_ids = os.path.join(os.path.dirname(args.path_to_dataset), "val_longer_code_problem_ids.txt")
        #     with open(path_to_problem_ids, 'r') as f:
        #         problem_ids = [p.strip() for p in f.readlines()]
        #     df = df[df["codenet_problem_id"].isin(problem_ids)]
        #     df = df.reset_index(drop=True)
            # assert len(df["codenet_problem_id"].unique()) == 15, f"Expected 15 problems, got {len(df['codenet_problem_id'].unique())}"
        # get first 5 rows
        if args.max_programs > 0:
            logging.info(f"Limiting to {args.max_programs} programs")
            df = df.iloc[:args.max_programs]

        results = []
        count = 0
        times = []
        start = datetime.datetime.now()
        # iterate over the dataframe
        for index, row in tqdm(df.iterrows()):
            this_start = datetime.datetime.now()
            logging.info(f"Generating for index {index}")
            # try:
            result = {}
            result['model'] = args.model
            result['index'] = index
            
            prompt = row['description_string']
            problem_id = row['problem_id'] if not is_directed else row["codenet_problem_id"]
            # problem_id = row['problem_id'] 
            extract_arguments_fun = row["extract_args_fun"] if not is_directed else None
            
            # store the row info into result
            result.update(row.to_dict())
            
            # format the prompt
            formatted_prompt = format_template_fun(prompt)
            result['formatted_prompt'] = formatted_prompt
            if "tatsu" in args.model:
                # alpaca-from max total-tokens = 2048
                n_prompt_tokens = len(tokenizer(formatted_prompt)['input_ids'])
                max_tokens = min(2000 - n_prompt_tokens, args.max_length)
                
            elif "codellama" in args.model.lower():
                n_prompt_tokens = len(tokenizer(formatted_prompt)['input_ids'])
                max_tokens = min(4000 - n_prompt_tokens, args.max_length)
                
            else: 
                max_tokens = args.max_length
            
            
            # Generate text
            raw_generations = pipe.generate(
                formatted_prompt, 
                max_new_tokens=max_tokens,
                num_samples=args.num_return_sequences,
                temperature=args.temperature,
                do_sample=True, 
                top_p=args.top_p,
                top_k=None,
                return_dict_in_generate=True, 
                batch_size=args.batch_size,
            )
            
            programs = [textprocessing.extract_python_code(g) for g in raw_generations]
            if is_directed: 
                formatted_programs = programs
            else: 
                formatted_programs = [clustering.format_open_ended_code(program, extract_arguments_fun) for program in programs]
    
            result["raw_generations"] = raw_generations
            result['description'] = prompt
            result['programs'] = programs
            result['formatted_programs'] = formatted_programs
            testcase_inputs = row['input_testcases']
            result['input_testcases'] = testcase_inputs
            result['problem_id'] = problem_id
            result['extract_args_fun'] = extract_arguments_fun
            result['original_description_string'] = prompt
                                  
            problem_id_dir = os.path.join(experiment_output_dir, f'problem_{problem_id}')   
            problem_id_gen_dir = os.path.join(problem_id_dir, 'generated')
            os.makedirs(problem_id_gen_dir, exist_ok=True)  # may want to set this to False                 
            for i, (generation, program, formatted_program) in enumerate(zip(raw_generations, programs, formatted_programs)):
                with open(os.path.join(problem_id_gen_dir, f'gen_{i}_coh_.txt'), 'w') as f:
                    f.write(generation)
                with open(os.path.join(problem_id_gen_dir, f'prog_{i}_coh.txt'), 'w') as f:
                    f.write(program)
                with open(os.path.join(problem_id_gen_dir, f'formatted_prog_{i}.txt'), 'w') as f:
                    f.write(formatted_program)
                    
            results.append(result)
            count += 1
            this_end = datetime.datetime.now()
            run_elapsed = this_end - this_start
            times.append(run_elapsed)
            logging.info(f"Finished index {index} in {run_elapsed}")
            
        end = datetime.datetime.now()
        total_elapsed = end - start
        
        logging.info(f"Finished all in {total_elapsed}")

        # save results to jsonl
        logging.info("Saving results to jsonl")
        # with open(os.path.join(experiment_output_dir, 'results.jsonl'), 'w') as f:
        #     for result in results:
        #         f.write(json.dumps(result) + '\n')
        # import pdb; pdb.set_trace()
        pd.DataFrame(results).to_json(os.path.join(experiment_output_dir, 'results.jsonl'), orient='records', lines=True)
        logging.info("Done saving results to jsonl")
            
        if "gpt" not in args.model:
            pipe.stop_service()
            pipe.remove_service()
        
        print(f"Done generating for {experiment_name} in {total_elapsed}")
        
    except Exception as e:
        # save some file if there is an error to communicate 
        traceback_str = traceback.format_exc()
        with open(os.path.join(experiment_output_dir, 'error.txt'), 'w') as f:
            f.write("Error during generation\n")
            f.write(traceback_str)
        logging.error(f"Error during generation: {traceback_str}")
        if "gpt" not in args.model and pipe is not None:
            logging.info("Stopping and removing the service...")
            pipe.stop_and_remove_if_running()
            logging.info("Cleanup complete, exiting...")
        raise e
