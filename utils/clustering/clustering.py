from typing import List, Optional, Union, Dict, Tuple
import tempfile
import os
import shutil
import docker
import logging
import traceback 
import glob 
from collections import defaultdict
from joblib import Parallel, delayed
import contextlib
import yaml
from dataclasses import dataclass
from typing import List, Optional, Dict
import joblib
from tqdm import tqdm
import re
import requests
import warnings
import json 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


clustering_abs_dir = os.path.dirname(os.path.abspath(__file__))
docker_driver_abs_path = os.path.join(clustering_abs_dir, "docker_driver.py")
docker_file_abs_path = os.path.join(clustering_abs_dir, "Dockerfile")
open_ended_wrapper_abs_path = os.path.join(clustering_abs_dir, "open_ended_wrapper.py")

import uuid 


def format_open_ended_code(f_code: str, extract_arguments_code: str) -> str:
    """
    Call this function to format the open-ended code with the f() function and extract_arguments() function
    for the open-ended scenario, to then be evaluated in the Docker container.
    """
    with open(open_ended_wrapper_abs_path, "r") as f:
        wrapper_code = f.read()
    formatted_wrapper = wrapper_code.replace("## REPLACE F", f_code).replace("## REPLACE EXTRACT_ARGUMENTS", extract_arguments_code)
    assert "## REPLACE F" not in formatted_wrapper, "F not replaced in formatted wrapper"
    assert "## REPLACE EXTRACT_ARGUMENTS" not in formatted_wrapper, "extract_arguments not replaced in formatted wrapper"
    return formatted_wrapper


def build_docker_image(path_to_dockerfile, version_tag=None):
    tag = 'python-test-case-runner'
    client = docker.from_env()
    images = client.images.list()
    version_tag = version_tag or "latest"
    for image in images:
        if tag in image.tags or f"{tag}:{version_tag}" in image.tags:
            print(f"Image with tag '{tag}' already exists. Using existing image.")
            return client, image
    # Build the Docker image
    image, build_log = client.images.build(path=path_to_dockerfile, tag=f"{tag}:{version_tag}")
    return client, image


def instrument_code_docker(generated_code: str, testcase_inputs: Dict[str, str], orig_testcase_outputs: Union[Dict[str, str], None],
                           image, client, docker_working_dir = None, n_test_cases=-1, indiv_tc_timeout=5, verbose_docker=True, 
                           open_ended=False):
    
    if docker_working_dir is None: 
        docker_working_dir = tempfile.mkdtemp()
    
    if not os.path.exists(docker_working_dir):
        raise ValueError(f"{docker_working_dir} does not exist.")
    
    code_path = os.path.join(docker_working_dir, "soln.py")
    with open(code_path, "w") as f:
        f.write(generated_code)
        
    shutil.copy(docker_driver_abs_path, os.path.join(docker_working_dir, "driver.py"))
    
    for testcase_id, testcase_input in testcase_inputs.items():
        input_path = os.path.join(docker_working_dir, f"input.{testcase_id}.txt")
        with open(input_path, "w") as f:
            f.write(testcase_input)
    
    volumes = {docker_working_dir: {'bind': '/usr/src/app/tc_dir', 'mode': 'rw'}}
    
    try: 
        logging.info(f"Now running docker container for tc_gen.py with testcase_dir {docker_working_dir} and image {image}.")
        container = client.containers.run(
            image.tags[0],
            detach=True,
            volumes=volumes,
            command=f"python tc_dir/driver.py /usr/src/app/tc_dir {indiv_tc_timeout} {verbose_docker} {n_test_cases} {open_ended}",
        )
        # print the container logs 
        for line in container.logs(stream=True):
            print(line.strip().decode('utf-8'))
        logging.info(f"Done running tc_gen.py, stopping container {container.id}.")
        container.stop()
        logging.info(f"Done stopp container for tc_gen.py, removing container {container.id}.")
        # Remove the container
        container.remove()
        logging.info(f"Done removing container {container.id}")
    except Exception as e:
        traceback_str = traceback.format_exc()
        logging.error(f"Failed to run tc_gen.py with testcase_dir {docker_working_dir} and image {image}.")
        logging.error(traceback_str)

    output_files = glob.glob(os.path.join(docker_working_dir, "output.*.txt"))
    testcase_outputs = {}
    
    for output_file in output_files:
        output_number = re.search(r"output.(\d+).txt", output_file).group(1)
        with open(output_file, "r") as f:
            testcase_outputs[output_number] = f.read().strip()
            
    # make sure the number of outputs is the same as the number of inputs
    if len(testcase_outputs) != len(testcase_inputs):
        raise ValueError(f"Number of outputs ({len(testcase_outputs)}) does not match the number of inputs ({len(testcase_inputs)}); outputs: {testcase_outputs}, inputs: {testcase_inputs}")
            
    output_record = {
        "code": generated_code, 
        "testcase_outputs": testcase_outputs, 
        "testcase_inputs": testcase_inputs, 
        "orig_testcase_outputs": orig_testcase_outputs
    }
    return output_record

def report_coherence(output_records: List[Dict]):
    program_2_coherence = {}
    program_2_n_outputs = {}
    program_2_n_coherent = {}
    for output_record in output_records:
        n_outputs = len(output_record["testcase_outputs"])
        n_coherent = len([output for output in output_record["testcase_outputs"].values() if output not in ["Syntax Error", "Runtime Error", "Timeout"]])
        program_2_n_outputs[output_record["code"]] = n_outputs
        program_2_n_coherent[output_record["code"]] = n_coherent
        program_2_coherence[output_record["code"]] = n_coherent / n_outputs
    return program_2_coherence, program_2_n_outputs, program_2_n_coherent

def avg_coherence(output_records: List[Dict], strict=True):
    n_outputs_list = [len(output_record["testcase_outputs"]) for output_record in output_records]
    n_coherent_list = [len([output for output in output_record["testcase_outputs"].values() if output not in ["Syntax Error", "Runtime Error"]]) for output_record in output_records]
    coherent_list = [n_coherent / n_outputs for n_coherent, n_outputs in zip(n_coherent_list, n_outputs_list)]
    if strict: 
        coherent_list = [coherent for coherent in coherent_list if coherent == 1.0]
    return sum(coherent_list) / len(coherent_list)
    
    
    

def report_accuracy(output_records: List[Dict]):
    program_2_accuracy = {}
    for output_record in output_records:
        n_correct = 0 
        for tc_key, output in output_record["testcase_outputs"].items():
            if output.strip() == output_record["orig_testcase_outputs"][tc_key].strip():
                n_correct += 1
        program_2_accuracy[output_record["code"]] = n_correct / len(output_record["testcase_outputs"])
    return program_2_accuracy

def make_semantic_strings(output_records: List[Dict]):
    program_2_semantic_string = {}
    semantic_strings_2_programs = defaultdict(list)
    for output_record in output_records:
        semantic_string = ""
        for testcase_id, testcase_input in output_record["testcase_inputs"].items():
            testcase_output = output_record["testcase_outputs"][testcase_id]
            semantic_string += f"testcase_input: {testcase_input}, output: {testcase_output}\n"
        program_2_semantic_string[output_record["code"]] = semantic_string
        semantic_strings_2_programs[semantic_string].append(output_record["code"])
    return program_2_semantic_string, semantic_strings_2_programs
    

def make_clusters_iterative(programs: List[str],
                    testcases: Dict[str, str],
                    outputs: Optional[Dict[str, str]] = None,
                    do_report_coherence=False, 
                    do_report_accuracy=False, 
                    n_test_cases=-1, 
                    verbose_docker=True, 
                    open_ended=False):
    if do_report_accuracy:
        if outputs is None:
            raise ValueError("Need outputs to report accuracy.")
        if len(testcases) != len(outputs):
            raise ValueError("Number of testcases and outputs must match.")
        
    client, tcgen_image = build_docker_image(clustering_abs_dir)
    
    output_records = []
    for i, program in enumerate(programs):
        record = instrument_code_docker(program, testcases, outputs, tcgen_image, client, n_test_cases=n_test_cases, verbose_docker=verbose_docker, open_ended=open_ended)
        output_records.append(record)
        
        
    # import pdb; pdb.set_trace()
        
    if do_report_coherence:
        program_2_coherence, program_2_n_outputs, program_2_n_coherent = report_coherence(output_records)
    else: 
        program_2_coherence = program_2_n_outputs = program_2_n_coherent = None
    
    if do_report_accuracy:
        ## TODO: convert inputs and outputs to a dict of index 2 output to avoid any sorting/ordering issues and bugs
        program_2_accuracy = report_accuracy(output_records)
    else:
        program_2_accuracy = None
        
    program_2_semantic_string, semantic_strings_2_programs = make_semantic_strings(output_records)
    
    # cleanup 
    try: 
        client.images.remove(tcgen_image.id)
    # allow httperror to be raised
    except requests.exceptions.HTTPError as e:
        warnings.warn(f"Error removing image: {e}, generally this should be okay in case someone else is using the image")
    
    return program_2_semantic_string, semantic_strings_2_programs, program_2_coherence, program_2_n_outputs, program_2_n_coherent, program_2_accuracy
        
    

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

        
    
def make_clusters_parallel(programs: List[str],
                            testcases: List[str], 
                            outputs: Optional[List[str]] = None, 
                            do_report_coherence=False, 
                            do_report_accuracy=False, 
                            n_test_cases=-1, 
                            n_jobs=-1, 
                            verbose_docker=True, 
                            open_ended=False):
    if do_report_accuracy:
        if outputs is None:
            raise ValueError("Need outputs to report accuracy.")
        if len(testcases) != len(outputs):
            raise ValueError("Number of testcases and outputs must match.")
        
    client, tcgen_image = build_docker_image(clustering_abs_dir)
    
    with tqdm_joblib(tqdm(desc="Processing Programs", total=len(programs))) as progress_bar:
        output_records = Parallel(n_jobs=n_jobs, backend='threading')(delayed(instrument_code_docker)(
            program, testcases, outputs, tcgen_image, client, n_test_cases=n_test_cases, verbose_docker=verbose_docker, open_ended=open_ended
        ) for program in programs)
    
    if do_report_coherence:
        program_2_coherence, program_2_n_outputs, program_2_n_coherent = report_coherence(output_records)
    else: 
        program_2_coherence = program_2_n_outputs = program_2_n_coherent = None
    
    if do_report_accuracy:
        ## TODO: convert inputs and outputs to a dict of index 2 output to avoid any sorting/ordering issues and bugs
        program_2_accuracy = report_accuracy(output_records)
    else:
        program_2_accuracy = None
        
    program_2_semantic_string, semantic_strings_2_programs = make_semantic_strings(output_records)
    
    # cleanup 
    try: 
        client.images.remove(tcgen_image.id)
    # allow httperror to be raised
    except requests.exceptions.HTTPError as e:
        warnings.warn(f"Error removing image: {e}, generally this should be okay in case someone else is using the image")
    
    return program_2_semantic_string, semantic_strings_2_programs, program_2_coherence, program_2_n_outputs, program_2_n_coherent, program_2_accuracy


@dataclass
class Config:
    input_file: str
    output_dir: str
    generations_column: str = "generated_code"
    input_testcases_column: str = "testcases"
    report_coherence: bool = True
    report_accuracy: bool = False
    n_test_cases: int = -1
    n_jobs: int = -1

def read_config(config_path: str) -> Config:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)


## TODO: you should refactor all things to use testcase_id: input, testcase_id: output to be more explicit and to reduce any risks of bugs

def load_data(input_file: str, config: Config) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    import pandas as pd
    df = pd.read_json(input_file, lines=True, orient="records")
    programs = [] 
    testcases = []
    outputs = []
    from tqdm import tqdm
    for i, row in tqdm(df.iterrows(), total=len(df)):
        _testcases = row[config.input_testcases_column]
        _outputs = row["outputs"]
        _generated = row[config.generations_column]
        if isinstance(_generated, list):
            for generated_code in _generated:
                programs.append(generated_code)
                testcases.append(_testcases)
                outputs.append(_outputs)
        else:
            programs.append(_generated)
            testcases.append(_testcases)
            outputs.append(_outputs)
    return programs, testcases, outputs

def save_results(results: Tuple[Dict, Dict, Dict, Dict, Dict, Dict], output_dir: str):
    program_2_semantic_string, semantic_strings_2_programs, program_2_coherence, program_2_n_outputs, program_2_n_coherent, program_2_accuracy = results
    if not os.path.exists(output_dir):
        logging.info(f"Creating output directory {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "program_2_semantic_string.json"), "w") as f:
        json.dump(program_2_semantic_string, f)
    with open(os.path.join(output_dir, "semantic_strings_2_programs.json"), "w") as f:
        json.dump(semantic_strings_2_programs, f)
    with open(os.path.join(output_dir, "program_2_coherence.json"), "w") as f:
        json.dump(program_2_coherence, f)
    with open(os.path.join(output_dir, "program_2_n_outputs.json"), "w") as f:
        json.dump(program_2_n_outputs, f)
    with open(os.path.join(output_dir, "program_2_n_coherent.json"), "w") as f:
        json.dump(program_2_n_coherent, f)
    with open(os.path.join(output_dir, "program_2_accuracy.json"), "w") as f:
        json.dump(program_2_accuracy, f)
    logging.info(f"Saved results to {output_dir}")

def main(config_path: str):
    config = read_config(config_path)
    
    # Load programs, test cases, and expected outputs from the input file
    programs, testcases, outputs = load_data(config.input_file)
    
    # Run the clustering
    results = make_clusters_parallel(
        programs,
        testcases,
        outputs,
        config.report_coherence,
        config.report_accuracy,
        config.n_test_cases,
        config.n_jobs
    )
    
    # Save results to the output directory
    save_results(results, config.output_dir)
    
    
if __name__ == "__main__":
    import sys
    main(sys.argv[1])
    

