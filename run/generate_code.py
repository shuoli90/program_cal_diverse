import os
import sys
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))
import argparse
from models import gpt, opensource
from utils import textprocessing
from utils.clustering import clustering

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text from a prompt')
    parser.add_argument('--model', type=str,
                        default='codellama/CodeLlama-7b-hf',
                        help='model to use')
    parser.add_argument('--prompt', type=str, default='Once upon a time')
    parser.add_argument('--template', type=str, default='vanilla')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--num_return_sequences', type=int, default=1)
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

    # Generate text
    generateds = pipe.generate(args.prompt, temperature=args.temperature,
                              max_length=args.max_length,
                              do_sample=True if args.num_return_sequences > 1 else False,
                              num_return_sequences=args.num_return_sequences,
                              return_full_text=False)
    functions = [textprocessing.extract_function(g) for g in generateds]
    breakpoint()
    function_names, function_parameters, function_bodies = [], [], []
    for function in functions:
        if function:
            function_names.append(function[0])
            function_parameters.append(function[1])
            function_bodies.append(function[2])

    programs = [textprocessing.code_template(function_body) for function_body in function_bodies]

    testcase_inputs = {"0": "3", "1": "4"}
    expected_outputs = {"0": "Input: 3\nFactorial: 6", "1": "Input: 4\nFactorial: 24"}
    orig_testcase_outputs = {"0": "6", "1": "24"}
    client, image = clustering.build_docker_image(clustering.clustering_abs_dir)

    # Test
    output_records = [clustering.instrument_code_docker(
        program, 
        testcase_inputs, 
        expected_outputs,
        image, 
        client, 
        n_test_cases=-1, 
        indiv_tc_timeout=20, 
        verbose_docker=True
    ) for program in programs]

    # report coherence
    if type(output_records) is not list:
        output_record = [output_records]
    coherence, n_outputs, n_coherent = clustering.report_coherence(output_records)

    # report accuracy
    accuracy = clustering.report_accuracy(output_records)

    # semantic_clustering
    program_2_semantic_string, semantic_strings_2_programs = clustering.make_semantic_strings(output_records)