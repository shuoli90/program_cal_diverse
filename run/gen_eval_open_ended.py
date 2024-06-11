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


@dataclass
class Arguments:
    path_to_dataset: str = '../data/open_ended/open_ended_final/dataset.jsonl'
    experiment_output_root: str = '../collected/'
    model: str = 'gpt-3.5-turbo'
    template: str = 'open_ended_default'
    temperature: float = 1.0
    top_p: float = 1.0
    max_length: int = 768
    num_return_sequences: int = 10
    repetition_penalty: float = 1.0
    parallel_samples: int = 5
    port: int = 9999
    devices_list: str = '4,5,6,7'
    startup_timeout: int = 600
    generation_timeout: int = 100
    volume: str = 'saved_models'
    path_to_hf_token: str = None
    
    
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
    experiment_string = f"{model_name_clean}_temp_{args.temperature}_top_p_{args.top_p}_max_length_{args.max_length}_num_return_sequences_{args.num_return_sequences}_repetition_penalty_{args.repetition_penalty}_{args.template}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    experiment_output_dir = os.path.join(args.experiment_output_root, experiment_string)
    os.makedirs(experiment_output_dir, exist_ok=False) # there should be no existing directory (H-M)
    
    # Setup generation pipeline
    if 'gpt' in args.model or 'davinci' in args.model:
        pipe = gpt.GPTModel(model_name=args.model)
    else:
        # pipe = opensource.OpensourceModel(model_name=args.model)
        with open(args.path_to_hf_token, "r") as f:
            hf_key = f.read().strip()
        pipe = hf_inference.HFInferenceService(model_name=args.model, 
                                                parallel_samples=max(args.parallel_samples,args.num_return_sequences),
                                                port=args.port,
                                                devices_list=args.devices_list,
                                                volume=args.volume,
                                                startup_timeout=args.startup_timeout,
                                                generation_timeout=args.generation_timeout,
                                                hf_key=hf_key)
    try:                                                

        # Read in data
        print(f'reading in data from {args.path_to_dataset}')
        df = pd.read_json(args.path_to_dataset, lines=True, orient='records')
        # get first 5 rows
        # df = df.iloc[:3]

        # setup docker client
        client, image = clustering.build_docker_image(clustering.clustering_abs_dir)

        results = []
        count = 0
        # iterate over the dataframe
        for index, row in tqdm(df.iterrows()):
            if index > 100:
                break
            # try:
            result = {}
            result['model'] = args.model
            result['index'] = index
            # prompt = row["prompt"]
            prompt = row['description_string']
            problem_id = row['problem_id']
            extract_arguments_fun = row["extract_args_fun"]
            
            # format the prompt
            formatted_prompt = format_template_fun(prompt)
            
            # Generate text
            generateds_program = pipe.generate(
                formatted_prompt, 
                temperature=args.temperature,
                num_return_sequences=args.num_return_sequences,
                # TODO: add more args
                max_length=args.max_length,
                do_sample=True, 
                return_dict_in_generate=True
            )
            
            programs = [textprocessing.extract_python_code(g) for g in generateds_program]
            formatted_programs = [clustering.format_open_ended_code(program, extract_arguments_fun) for program in programs]
            
    
            # if all items in programs are None, skip
            # if all([program is None for program in programs]):
            #     continue
            result['description'] = prompt
            result['programs'] = programs
            result['formatted_programs'] = formatted_programs
            testcase_inputs = row['input_testcases']
            result['input_testcases'] = testcase_inputs
            # testcase_outputs = row['output_testcases'] # no output testcases

            # Test
        
            # output_records = [clustering.instrument_code_docker(
            #     formatted_program, 
            #     testcase_inputs, 
            #     None, # testcase_outputs is None
            #     image, 
            #     client,
            #     n_test_cases=-1, 
            #     indiv_tc_timeout=60, 
            #     verbose_docker=True) for formatted_program in formatted_programs if formatted_program is not None]
            with tqdm_joblib(tqdm(desc="Processing Programs", total=len(formatted_programs))) as progress_bar:
                output_records = Parallel(n_jobs=10, backend='threading')(delayed(clustering.instrument_code_docker)(
                    formatted_program, 
                    testcase_inputs, 
                    None, # testcase_outputs is None
                    image, 
                    client,
                    n_test_cases=-1, 
                    indiv_tc_timeout=60, 
                    verbose_docker=True) for formatted_program in formatted_programs if formatted_program is not None)
            
            result['output_records'] = output_records
            # report coherence
            if type(output_records) is not list:
                output_record = [output_records]
            coherences = clustering.get_coherence(output_records, strict=False)
            avg_coherence = np.mean([coherence == 1.0 for coherence in coherences])
            result['coherence'] = avg_coherence
                    
                    
            # coherence, n_outputs, n_coherent = clustering.report_coherence(output_records)
            # result['coherence'] = coherence
            # result['n_outputs'] = n_outputs
            # result['n_coherent'] = n_coherent

            # No accuracy conceptually exists for this task
            # report accuracy
            # accuracy = clustering.report_accuracy(output_records)
            # accuracies = []
            # for program in programs:
            #     if program is not None:
            #         accuracies.append(accuracy[program])
            #     else:
            #         accuracies.append(0.0)
            # result['accuracy'] = accuracies

            # semantic_clustering
            program_2_semantic_string, semantic_strings_2_programs = clustering.make_semantic_strings(output_records)
            semantic_count = len(semantic_strings_2_programs.keys())
            print('semantic count', semantic_count)
            result['semantic_count'] = semantic_count
            result['program_2_semantic_string'] = program_2_semantic_string
            result['semantic_strings_2_programs'] = semantic_strings_2_programs

            # lexical diversity metrics 
            # def distinct_n(corpus: List[str], n: int, ftokenizer: Callable[str]) -> float:
            # def parallel_corpus_self_bleu(sentences: List[str], ftokenizer: Callable[str], n_jobs: int = -1, normalize: bool = True) -> float:
            programs = [program for program in programs if program is not None]

            if len(programs) >= 2:
                distinct_1 = lexical_diversity.distinct_n(programs, 1, lexical_diversity.codebert_tokenizer)
                distinct_2 = lexical_diversity.distinct_n(programs, 2, lexical_diversity.codebert_tokenizer)
                distinct_3 = lexical_diversity.distinct_n(programs, 3, lexical_diversity.codebert_tokenizer)
                distinct_4 = lexical_diversity.distinct_n(programs, 4, lexical_diversity.codebert_tokenizer)
                distinct_5 = lexical_diversity.distinct_n(programs, 5, lexical_diversity.codebert_tokenizer)
                distinct_6 = lexical_diversity.distinct_n(programs, 6, lexical_diversity.codebert_tokenizer)
                corpus_self_bleu = lexical_diversity.parallel_corpus_self_bleu(programs, lexical_diversity.codebert_tokenizer, n_jobs=8, normalize=True)
                result['distinct_1'] = distinct_1
                result['distinct_2'] = distinct_2
                result['distinct_3'] = distinct_3
                result['distinct_4'] = distinct_4
                result['distinct_5'] = distinct_5
                result['distinct_6'] = distinct_6
                result['corpus_self_bleu'] = corpus_self_bleu
                parallel_subtree_results = parallel_subtree_analysis(programs, n_jobs=8, heights=[3,4,5,6])
                for key, height_results in parallel_subtree_results.items():
                    for height, v in height_results.items():
                        result[f"{key}_{height}"] = v
            else:
                result['distinct_1'] = 0.0
                result['distinct_2'] = 0.0
                result['distinct_3'] = 0.0
                result['distinct_4'] = 0.0
                result['distinct_5'] = 0.0
                result['distinct_6'] = 0.0
                result['corpus_self_bleu'] = 0.0
                for key in ['plain_subtrees', 'stripped_subtrees', 'obfuscated_subtrees']:
                    for height in [3,4,5,6]:
                        result[f"{key}_{height}"] = 0.0
            
            problem_id_dir = os.path.join(experiment_output_dir, f'problem_{problem_id}')   
            os.makedirs(problem_id_dir, exist_ok=False)                 
            for i, (generation, program, formatted_program, output_record, coherence) in enumerate(zip(generateds_program, programs, formatted_programs, output_records, coherences)):
                with open(os.path.join(problem_id_dir, f'gen_{i}_coh_{coherence}.txt'), 'w') as f:
                    f.write(generation)
                with open(os.path.join(problem_id_dir, f'prog_{i}_coh_{coherence}.txt'), 'w') as f:
                    f.write(program)
                with open(os.path.join(problem_id_dir, f'formatted_prog_{i}_coh_{coherence}.txt'), 'w') as f:
                    f.write(formatted_program)
                with open(os.path.join(problem_id_dir, f'output_record_{i}_coh_{coherence}.json'), 'w') as f:
                    f.write(json.dumps(output_record))
                    
            with open(os.path.join(problem_id_dir, f'result.tsv'), 'w') as f:
                for k in ['coherence', 'semantic_count', 'distinct_3', 'distinct_4', 'distinct_5', 'plain_subtrees_3', 'plain_subtrees_4', 'plain_subtrees_5', 'plain_subtrees_6', 'stripped_subtrees_3', 'stripped_subtrees_4', 'stripped_subtrees_5', 'stripped_subtrees_6', 'obfuscated_subtrees_3', 'obfuscated_subtrees_4', 'obfuscated_subtrees_5', 'obfuscated_subtrees_6']:
                    f.write(f"{k}\t{result[k]}\n")
            # except ValueError as e:
            #     print('error')
            #     continue
            results.append(result)
            if count % 10 == 0:
                # save results to jsonl
                # with open(f'../collected/open_ended_{args.model}_temp_{args.temperature}_top_p_{args.top_p}_max_length_{args.max_length}_num_return_sequences_{args.num_return_sequences}_repetition_penalty_{args.repetition_penalty}_results.jsonl', 'w') as f:
                with open(os.path.join(experiment_output_dir, 'results.jsonl'), 'w') as f:
                    for result in results:
                        f.write(json.dumps(result) + '\n')
            count += 1
            # except 
            #     continue

        # save results to jsonl
        with open(os.path.join(experiment_output_dir, 'results.jsonl'), 'w') as f:
        # with open(f'../collected/open_ended_{args.model}_temp_{args.temperature}_top_p_{args.top_p}_max_length_{args.max_length}_num_return_sequences_{args.num_return_sequences}_repetition_penalty_{args.repetition_penalty}_results.jsonl', 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        # concatenate all the results, summarize the statistics
        df_results = pd.DataFrame(results)
        results_stats_keys = ['coherence', 'semantic_count', 'distinct_1', 'distinct_2', 'distinct_3', 'distinct_4', 'distinct_5', 'distinct_6', 'corpus_self_bleu']
        results_stats_keys = results_stats_keys + [f"{key}_{height}" for key in ['plain_subtrees', 'stripped_subtrees', 'obfuscated_subtrees'] for height in [3,4,5,6]]
        df_results_stats = df_results[results_stats_keys]
        described = df_results_stats.describe()
        print(described)
        # save the statistics
        # described.to_csv(f'../collected/open_ended_{args.model}_temp_{args.temperature}_top_p_{args.top_p}_max_length_{args.max_length}_num_return_sequences_{args.num_return_sequences}_repetition_penalty_{args.repetition_penalty}_results_stats.csv')
        # described.to_csv(os.path.join(experiment_output_dir, 'results_stats.csv'))
        # save as tsv
        described.to_csv(os.path.join(experiment_output_dir, 'results_stats.tsv'), sep='\t')
        # save only the mean
        mean = described.loc['mean']
        with open(os.path.join(experiment_output_dir, 'results_stats_mean.tsv'), 'w') as f:
            for k, v in mean.items():
                f.write(f"{k}\t{v}\n")
                
        if "gpt" not in args.model:
            pipe.stop_service()
            pipe.remove_service()
        
        print('Done')
        
    except Exception as e:
        traceback.print_exc()
        if "gpt" not in args.model:
            pipe.stop_service()
            pipe.remove_service()
        raise e
