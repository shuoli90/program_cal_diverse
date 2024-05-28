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
from tqdm import tqdm
from functools import partial


@dataclass
class Arguments:
    path_to_dataset: str = '../data/codenet/open_ended/open_ended_final/dataset.jsonl'
    model: str = 'gpt-3.5-turbo'
    template: str = 'open_ended_default'
    temperature: float = 1.0
    top_p: float = 1.0
    max_length: int = 768
    num_return_sequences: int = 10
    repetition_penalty: float = 1.0
    
    
def readin_template(template_arg): 
    assert template_arg in ["open_ended_default", None], "Template not supported"
    if template_arg == None: 
        return "## DESCRIPTION\n"
    else: 
        path_to_template = os.path.join(os.path.dirname(__file__), f"../templates/{template_arg}.txt")
        with open(path_to_template, 'r') as file:
            template = file.read()
        return template
    

# def format_null_template(promot): 
#     return prompt 

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
    format_template_fun = partial(_format_template, template=template)

    
    # Setup generation pipeline
    if 'gpt' in args.model or 'davinci' in args.model:
        pipe = gpt.GPTModel(model_name=args.model)
    else:
        pipe = opensource.OpensourceModel(model_name=args.model)

    # Read in data
    print(f'reading in data from {args.path_to_dataset}')
    df = pd.read_json(args.path_to_dataset, lines=True, orient='records')

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
        
        # format the prompt
        formatted_prompt = format_template_fun(prompt)
        
        # Generate text
        generateds_program = pipe.generate(
            formatted_prompt, 
            temperature=args.temperature,
            # TODO: add more args
            max_length=args.max_length,
            do_sample=True, 
            return_dict_in_generate=True
        )
        
        programs = [textprocessing.extract_python_code(g) for g in generateds_program]
 
        # if all items in programs are None, skip
        if all([program is None for program in programs]):
            continue
        result['description'] = prompt
        result['programs'] = programs
        testcase_inputs = row['input_testcases']
        result['input_testcases'] = testcase_inputs
        # testcase_outputs = row['output_testcases'] # no output testcases

        # Test
        output_records = [clustering.instrument_code_docker(
            program, 
            testcase_inputs, 
            testcase_outputs,
            image, 
            client,
            n_test_cases=-1, 
            indiv_tc_timeout=60, 
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
            corpus_self_bleu = lexical_diversity.parallel_corpus_self_bleu(programs, lexical_diversity.codebert_tokenizer, n_jobs=-1, normalize=True)
            result['distinct_1'] = distinct_1
            result['distinct_2'] = distinct_2
            result['distinct_3'] = distinct_3
            result['corpus_self_bleu'] = corpus_self_bleu
        else:
            result['distinct_1'] = 0.0
            result['distinct_2'] = 0.0
            result['distinct_3'] = 0.0
            result['corpus_self_bleu'] = 0.0
        # except ValueError as e:
        #     print('error')
        #     continue
        results.append(result)
        if count % 10 == 0:
            # save results to jsonl
            with open(f'../collected/{args.model}_temp_{args.temperature}_top_p_{args.top_p}_max_length_{args.max_length}_num_return_sequences_{args.num_return_sequences}_repetition_penalty_{args.repetition_penalty}_results.jsonl', 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
            count += 1
        # except 
        #     continue

    # save results to jsonl
    with open(f'../collected/{args.model}_temp_{args.temperature}_top_p_{args.top_p}_max_length_{args.max_length}_num_return_sequences_{args.num_return_sequences}_repetition_penalty_{args.repetition_penalty}_results.jsonl', 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # concatenate all the results, summarize the statistics
    df_results = pd.DataFrame(results)
    df_results_stats = df_results[['coherence', 'semantic_count', 'n_outputs', 'n_coherent', 'distinct_1', 'distinct_2', 'distinct_3', 'corpus_self_bleu']]
    described = df_results_stats.describe()
    print(described)
    # save the statistics
    described.to_csv(f'../collected/{args.model}_temp_{args.temperature}_top_p_{args.top_p}_max_length_{args.max_length}_num_return_sequences_{args.num_return_sequences}_repetition_penalty_{args.repetition_penalty}_results_stats.csv')
    
    print('Done')
