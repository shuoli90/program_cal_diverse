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
import copy 

import signal
import traceback

from async_driver import Arguments, load_arguments_from_yaml
from eval_driver import results_stats_keys
import logging 

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
#     eval_workers: int = 10
#     eval_timeout: int = 60
#     docker_communication_timeout: int = 2000
#     reformat_results: bool = True
    
    
template_dir = os.path.join(os.path.dirname(__file__), "../prompt_templates")
templates = [f for f in os.listdir(template_dir) if f.endswith('.txt')]
template_names = [f.split('.')[0] for f in templates] + [None]

# # make the keys for the results
# results_stats_keys = ['coherence', 'semantic_count', 'semantic_proportion', 'distinct_1', 'distinct_2', 'distinct_3', 'distinct_4', 'distinct_5', 'distinct_6']
# results_stats_keys = results_stats_keys + [f"{key}_{height}" for key in ['plain_subtrees', 'stripped_subtrees'] for height in [3,4,5,6]]
# results_stats_keys = [f"{recordtype}_{key}" for recordtype in ['all', 'coh', 'err'] for key in results_stats_keys]
# results_stats_keys.insert(4, 'coh_semantic_proportion_of_all')

    
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


def load_arguments_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        args_dict = yaml.safe_load(file)
    return Arguments(**args_dict)

if __name__ == '__main__':
    path_to_yaml = sys.argv[1]
    args = load_arguments_from_yaml(path_to_yaml)
    
    # make sure the template is valid
    prompt_template = readin_template(args.template)
    format_template_fun = partial(_format_template, template=prompt_template)
    
    model_name_clean = args.model.replace("/", "-")
    # experiment_string = f"{model_name_clean}_temp_{args.temperature}_top_p_{args.top_p}_max_length_{args.max_length}_num_return_sequences_{args.num_return_sequences}_repetition_penalty_{args.repetition_penalty}_{args.template}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    experiment_id= args.experiment_id
    experiment_output_dir = args.experiment_output_dir
    results_file = os.path.join(experiment_output_dir, 'results.jsonl')
    assert os.path.exists(results_file)
    log_file = os.path.join(experiment_output_dir, 'eval_log.txt')
    logging.basicConfig(level=logging.INFO,
                        handlers=[
                                logging.FileHandler(log_file),  # File handler
                                logging.StreamHandler(sys.stdout)  # Console handler
                            ], 
                        force=True
    )

    # setup docker client                           
    client, image = clustering.build_docker_image(clustering.clustering_abs_dir, max_pool_size=args.eval_workers, timeout=args.docker_communication_timeout)
    results_df = pd.read_json(results_file, lines=True, orient='records') 
    results = results_df.to_dict(orient='records')
   
    if args.reformat_results:
        reformatted_results = []  
        for result in results: 
            new_result = copy.deepcopy(result)
            raw_generations = result['raw_generations']
            extract_arguments_fun = result['extract_args_fun']
            programs = [textprocessing.extract_python_code(g) for g in raw_generations]
            formatted_programs = [clustering.format_open_ended_code(program, extract_arguments_fun) for program in programs]
            new_result['programs'] = programs
            new_result['formatted_programs'] = formatted_programs
            reformatted_results.append(new_result)
            
        results = reformatted_results
        
    
    submit_tuples = []
    from itertools import chain, repeat
    for result in results:
        problem_id = result['problem_id']
        input_testcases = result['input_testcases']
        for generation_id, formatted_program in enumerate(result['formatted_programs']):
            submit_tuples.append((problem_id, generation_id, formatted_program, input_testcases))

    
    logging.info(f"Starting evaluation for {experiment_id}, {len(submit_tuples)} programs")
    with tqdm_joblib(tqdm(desc="Processing Programs", total=len(submit_tuples))) as progress_bar:
        output_records = Parallel(n_jobs=args.eval_workers, backend='threading')(delayed(clustering.instrument_code_docker)(
            generated_code=formatted_program, 
            testcase_inputs=testcase_inputs, 
            orig_testcase_outputs=None, # testcase_outputs is None
            image=image, 
            client=client,
            n_test_cases=-1, 
            indiv_tc_timeout=60, 
            verbose_instrument=False, 
            verbose_docker=True, 
            problem_id=problem_id,
            generation_id=generation_id,
        ) for problem_id, generation_id, formatted_program, testcase_inputs in submit_tuples)
    
    final_results = []
    
    for result in results: 
        problem_id = result['problem_id']
        matching_records = [record for record in output_records if record['problem_id'] == problem_id]
        sorted_records = sorted(matching_records, key=lambda x: x['generation_id'], reverse=False)
        result['output_records'] = sorted_records

        coherent_records = clustering.get_coherent_records(sorted_records)
        incoherent_records = clustering.get_incoherent_records(sorted_records) 
        result['coherent_records'] = coherent_records
        result['incoherent_records'] = incoherent_records
    
        recordtype_2_records = {
                "all": sorted_records,
                "coh": coherent_records, 
                "err": incoherent_records
        }
        problem_id_dir = os.path.join(experiment_output_dir, f'problem_{problem_id}')   
        os.makedirs(problem_id_dir, exist_ok=True) # we can set to false, for debugging                  
            
        for recordtype, records in recordtype_2_records.items():
            # report coherence
            if type(records) is not list:
                records = [records]
            coherences = clustering.get_coherence(records, strict=False)
            avg_coherence = np.mean([coherence == 1.0 for coherence in coherences])
            result[f'{recordtype}_coherence'] = avg_coherence

            # semantic_clustering
            program_2_semantic_string, semantic_strings_2_programs = clustering.make_semantic_strings(records)
            semantic_count = len(semantic_strings_2_programs.keys())
            result[f'{recordtype}_semantic_count'] = semantic_count
            result[f'{recordtype}_semantic_proportion'] = semantic_count / len(records) if len(records) > 0 else np.nan
            
            if recordtype =="coh": 
                result["coh_semantic_proportion_of_all"] = semantic_count / len(recordtype_2_records["all"]) if len(recordtype_2_records["all"]) > 0 else np.nan

            result[f'{recordtype}_program_2_semantic_string'] = program_2_semantic_string
            result[f'{recordtype}_semantic_strings_2_programs'] = semantic_strings_2_programs

            # lexical diversity metrics 
            programs = [record["code"] for record in records if record["code"] is not None]
            
            if len([p for p in programs if len(p) > 0]) > 2:
                import tokenize
                for i in range(1, 7):
                    distinct_n = lexical_diversity.distinct_n(programs, i, lexical_diversity.get_relevant_tokens_parso, remove_comments=False)
                    result[f'{recordtype}_distinct_{i}'] = distinct_n
                    distinct_n_no_comments = lexical_diversity.distinct_n(programs, i, lexical_diversity.get_relevant_tokens_parso, remove_comments=True)
                    result[f'{recordtype}_distinct_{i}_no_comments'] = distinct_n_no_comments
                    
                corpus_self_bleu = lexical_diversity.parallel_corpus_self_bleu(programs, lexical_diversity.get_relevant_tokens_parso, n_jobs=args.eval_workers, normalize=True)
                result[f'{recordtype}_corpus_self_bleu'] = corpus_self_bleu
                parallel_subtree_results = parallel_subtree_analysis(programs, n_jobs=args.eval_workers, heights=[3,4,5,6], verbose=False)
                for key, height_results in parallel_subtree_results.items():
                    for height, v in height_results.items():
                        result[f"{recordtype}_{key}_{height}"] = v
            else:
                for i in range(1, 7):
                    result[f'{recordtype}_distinct_{i}'] = np.nan
                    result[f'{recordtype}_distinct_{i}_no_comments'] = np.nan
                # result[f'{recordtype}_distinct_1'] = np.nan
                # result[f'{recordtype}_distinct_2'] = np.nan
                # result[f'{recordtype}_distinct_3'] = np.nan
                # result[f'{recordtype}_distinct_4'] = np.nan
                # result[f'{recordtype}_distinct_5'] = np.nan
                # result[f'{recordtype}_distinct_6'] = np.nan
                result[f'{recordtype}_corpus_self_bleu'] = np.nan
                for key in ['plain_subtrees', 'stripped_subtrees', 'obfuscated_subtrees']:
                    for height in [3,4,5,6]:
                        result[f"{recordtype}_{key}_{height}"] = np.nan
                                                                                                                 
            # save the results
            if recordtype == 'all':
                logging.info(f"Saving results for problem {problem_id}, coherence: {avg_coherence}, semantic count: {semantic_count}")
                _programs = [record['code'] for record in records]
                raw_generations = result['raw_generations']

                for i, (generation, program, formatted_program, output_record, coherence) in enumerate(zip(raw_generations, _programs, formatted_programs, output_records, coherences)):
                    with open(os.path.join(problem_id_dir, f'gen_{i}_coh_{coherence}.txt'), 'w') as f:
                        f.write(generation)
                    with open(os.path.join(problem_id_dir, f'prog_{i}_coh_{coherence}.txt'), 'w') as f:
                        f.write(program)
                    with open(os.path.join(problem_id_dir, f'formatted_prog_{i}_coh_{coherence}.txt'), 'w') as f:
                        f.write(formatted_program)
                    with open(os.path.join(problem_id_dir, f'output_record_{i}_coh_{coherence}.json'), 'w') as f:
                        f.write(json.dumps(output_record))  
                
        with open(os.path.join(problem_id_dir, f'result.tsv'), 'w') as f:
            for k in results_stats_keys:
                f.write(f"{k}\t{result[k]}\n")
            
        final_results.append(result)
        logging.info(f"Results for problem {problem_id} saved")
        
    # save results to jsonl
    with open(os.path.join(experiment_output_dir, 'results.jsonl'), 'w') as f:
        for result in final_results:
            f.write(json.dumps(result) + '\n')
    
    # concatenate all the results, summarize the statistics
    df_results = pd.DataFrame(final_results)
    
    df_results_stats = df_results[results_stats_keys]
    described = df_results_stats.apply(lambda x: x.dropna().describe())
    print(described)
    # save as tsv
    described.to_csv(os.path.join(experiment_output_dir, 'results_stats.tsv'), sep='\t')
    # save only the mean
    mean = described.loc['mean']
    with open(os.path.join(experiment_output_dir, 'results_stats_mean.tsv'), 'w') as f:
        for k, v in mean.items():
            f.write(f"{k}\t{v}\n")
    
    print('Done')
    
