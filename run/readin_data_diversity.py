import os
import sys
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))
import argparse
import pandas as pd
import json
import numpy as np
from models import gpt, opensource
from utils import textprocessing
from utils.clustering import clustering
from utils.clustering import lexical_diversity
from dataclasses import dataclass
import yaml


@dataclass
class Arguments:
    root: str = '../data'
    folder: str = 'codenet'
    split: str = 'val'
    model: str = 'gpt-3.5-turbo'
    prompt: str = 'Once upon a time'
    template: str = 'vanilla'
    temperature: float = 1.0
    max_length: int = 512
    num_return_sequences: int = 1

def load_arguments_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        args_dict = yaml.safe_load(file)
    return Arguments(**args_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/args.yaml')
    parser_args = parser.parse_args()
    args = load_arguments_from_yaml(parser_args.config_path)
    
    # Setup generation pipeline
    if 'gpt' in args.model:
        pipe = gpt.GPTModel(model_name=args.model)
    else:
        pipe = opensource.OpensourceModel(model_name=args.model)

    # if args.template == 'vanilla':
    #     prompt = textprocessing.vanilla_template(args.prompt)
    # else:
    #     raise ValueError("Unknow template")

    # Read in data
    print(f'reading in data from {args.root}/{args.folder}/{args.split}.jsonl')
    path = os.path.join(args.root, args.folder, f'{args.split}.jsonl')
    df = pd.read_json(path, lines=True, orient='records')
    # df['length'] = df['description_string'].apply(len)
    # df = df[df['length'] < 1000]

    results = []
    count = 0
    # iterate over the dataframe
    for index, row in df.iterrows():
        if index > 30:
            break
        result = {}
        prompt = row["prompt"]
        # prompt = row['description_string']
        # replace '\n\n' and '\n\n\n' with '\n
        
        prompt = prompt.replace('\n\n\n', '\n').replace('\n\n', '\n') + 'write the funciton and a main function that reads in the input from stdin and writes the output to stdout.'
        # Generate text
        generateds = pipe.generate(prompt, temperature=args.temperature,
                                   max_length=args.max_length,
                                   do_sample=True if args.num_return_sequences > 1 else False,
                                   num_return_sequences=args.num_return_sequences,
                                   return_full_text=False)
        
        programs = [textprocessing.extract_function(g) for g in generateds]
        result['description'] = prompt
        result['programs'] = programs

        # programs = [textprocessing.code_template(function_body) for function_body in functions]

        testcase_inputs = row['testcase_inputs']
        client, image = clustering.build_docker_image(clustering.clustering_abs_dir)
        result['input_testcases'] = testcase_inputs

        # Test
        output_records = [clustering.instrument_code_docker(
            program, 
            testcase_inputs, 
            None,
            image, 
            client,
            n_test_cases=-1, 
            indiv_tc_timeout=20, 
            verbose_docker=True) for program in programs if program is not None]
        result['output_records'] = output_records

        # report coherence
        if type(output_records) is not list:
            output_record = [output_records]
        coherence, n_outputs, n_coherent = clustering.report_coherence(output_records)
        result['coherence'] = coherence
        result['n_outputs'] = n_outputs
        result['n_coherent'] = n_coherent

        # No accuracy conceptually exists for this task

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
        distinct_1 = lexical_diversity.distinct_n(programs, 1, lexical_diversity.codebert_tokenizer)
        distinct_2 = lexical_diversity.distinct_n(programs, 2, lexical_diversity.codebert_tokenizer)
        distinct_3 = lexical_diversity.distinct_n(programs, 3, lexical_diversity.codebert_tokenizer)
        corpus_self_bleu = lexical_diversity.parallel_corpus_self_bleu(programs, lexical_diversity.codebert_tokenizer, n_jobs=-1, normalize=True)
        result['distinct_1'] = distinct_1
        result['distinct_2'] = distinct_2
        result['distinct_3'] = distinct_3
        result['corpus_self_bleu'] = corpus_self_bleu
        # print(result)
        

        results.append(result)
        if count % 10 == 0:
            # save results to jsonl
            with open(f'../collected/{args.folder}_{args.split}_results.jsonl', 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
        count += 1
    
    # save results to jsonl
    with open(f'../collected/{args.folder}_{args.split}_results.jsonl', 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # concatenate all the results, summarize the statistics
    df_results = pd.DataFrame(results)
    df_results_stats = df_results[['coherence', 'semantic_count', 'n_outputs', 'n_coherent', 'distinct_1', 'distinct_2', 'distinct_3', 'corpus_self_bleu']]
    described = df_results_stats.describe()
    print(described)
    # save the statistics
    described.to_csv(f'../collected/{args.folder}_{args.split}_results_stats.csv')
    
    print('Done')
