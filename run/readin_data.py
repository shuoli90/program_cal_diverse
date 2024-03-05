import os
import sys
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))
import argparse
import pandas as pd
import json
import numpy as np
import torch
from tqdm import tqdm
from models import gpt, opensource
from utils import textprocessing
from utils.clustering import clustering
from indicators import uncertainty_blackbox, uncertainty_whitebox

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read in data')
    parser.add_argument('--root', type=str, default='../data',
                        help='root directory of data')
    parser.add_argument('--folder', type=str, default='codenet',
                        help='folder name of data')
    parser.add_argument('--split', type=str, default='val',
                        help='split of data')
    parser.add_argument('--model', type=str,
                        default='gpt-3.5-turbo',
                        help='model to use')
    parser.add_argument('--prompt', type=str, default='Once upon a time')
    parser.add_argument('--template', type=str, default='vanilla')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--num_return_sequences', type=int, default=5)
    args = parser.parse_args()

    # Setup generation pipeline
    if 'gpt' in args.model:
        pipe = gpt.GPTModel(model_name=args.model)
    else:
        pipe = opensource.OpensourceModel(model_name=args.model)

    if args.template == 'vanilla':
        prompt = textprocessing.vanilla_template(args.prompt)
    else:
        raise ValueError("Unknow template")

    # Read in data
    path = os.path.join(args.root, args.folder, f'{args.split}_descriptions_and_testcases.jsonl')
    df = pd.read_json(path, lines=True, orient='records')
    df['length'] = df['description_string'].apply(len)
    df = df[df['length'] < 1000]

    ecc = uncertainty_blackbox.Eccentricity()
    degree = uncertainty_blackbox.Degree()
    spectral = uncertainty_blackbox.SpectralEigv()
    se = uncertainty_whitebox.SemanticEntropy()

    results = []
    count = 0
    # iterate over the dataframe
    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            result = {'index':index}
            prompt = row['description_string']
            # replace '\n\n' and '\n\n\n' with '\n
            prompt = prompt.replace('\n\n\n', '\n').replace('\n\n', '\n') + 'write the funciton and a main function to call the function in python, using the form of markdown\n\n'
            # Generate text
            generateds, generated_nlls = pipe.generate(prompt, temperature=args.temperature,
                                    max_length=args.max_length,
                                    do_sample=True if args.num_return_sequences > 1 else False,
                                    num_return_sequences=args.num_return_sequences,
                                    return_full_text=False)
            programs = [textprocessing.extract_function(g) for g in generateds]
            result['description'] = prompt
            result['programs'] = programs

            # programs = [textprocessing.code_template(function_body) for function_body in functions]

            testcase_inputs = row['input_testcases']
            expected_outputs = row['output_testcases']
            client, image = clustering.build_docker_image(clustering.clustering_abs_dir)
            result['input_testcases'] = testcase_inputs
            result['output_testcases'] = expected_outputs

            # Test
            output_records = [clustering.instrument_code_docker(
                program, 
                testcase_inputs, 
                expected_outputs,
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

            # report accuracy
            accuracy = clustering.report_accuracy(output_records)
            # print('accuracy:', accuracy)
            result['accuracy'] = [accuracy[program] for program in programs]

            # semantic_clustering
            program_2_semantic_string, semantic_strings_2_programs = clustering.make_semantic_strings(output_records)
            # print('semantic count', len(semantic_strings_2_programs.keys()))
            result['program_2_semantic_string'] = program_2_semantic_string
            result['semantic_strings_2_programs'] = semantic_strings_2_programs

            # create similarity matrix
            similarity_matrix = np.zeros((len(programs), len(programs)))
            for i in range(len(programs)):
                for j in range(len(programs)):
                    program_i = programs[i]
                    program_j = programs[j]
                    if program_2_semantic_string[program_i] == program_2_semantic_string[program_j]:
                        similarity_matrix[i, j] = 1
            
            # blackbox methods
            ecc_scores = ecc([similarity_matrix])
            degree_scores = degree([similarity_matrix])
            spectral_scores = spectral([similarity_matrix])
            result['ecc_u'] = [num.astype(np.float64).tolist() for num in ecc_scores[0]]
            result['ecc_confidence'] = [num.astype(np.float64).tolist() for num in ecc_scores[1]]
            result['degree_u'] = [num.astype(np.float64) for num in degree_scores[0]]
            result['degree_confidence'] = [num.astype(np.float64).tolist() for num in degree_scores[1]]
            result['spectral_u'] = [num.astype(np.float64).tolist() for num in spectral_scores]
            result['similarity_matrix'] = similarity_matrix.tolist()

            # encode semantic to index
            semantic_ids = {}
            for idx, semantic in enumerate(semantic_strings_2_programs.keys()):
                semantic_ids[semantic] = idx

            # whitebox method
            semantic_set_ids = [semantic_ids[program_2_semantic_string[program]] for program in programs]
            se_scores = se.compute_scores(semantic_set_ids=[torch.tensor(semantic_set_ids)], nlls=generated_nlls)
            result['nll'] = [[np.sum(item) for item in nll] for nll in generated_nlls]
            result['se'] = [num.numpy().astype(np.float64).tolist() for num in se_scores]

            results.append(result)
            if count % 10 == 0:
                # save results to jsonl
                with open(f'../collected/{args.model}_{args.temperature}_{args.folder}_{args.split}.jsonl', 'w') as f:
                    for result in results:
                        f.write(json.dumps(result) + '\n')
            count += 1
        except Exception as e:
            print(e)
            print('Error at index:', index)
            continue

    # save results to jsonl
    with open(f'../collected/{args.model}_{args.temperature}_{args.folder}_{args.split}.jsonl', 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print('Done')
