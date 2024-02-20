from typing import List, Optional
import tempfile
import os
import shutil
import docker
import logging
import traceback 
import glob 
from collections import defaultdict
import joblib
from joblib import Parallel, delayed
import contextlib
import yaml
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


clustering_abs_dir = os.path.dirname(os.path.abspath(__file__))
docker_driver_abs_path = os.path.join(clustering_abs_dir, "docker_driver.py")
docker_file_abs_path = os.path.join(clustering_abs_dir, "Dockerfile")


def build_docker_image(path_to_dockerfile):
    tag = 'python-test-case-runner'
    client = docker.from_env()
    images = client.images.list()
    for image in images:
        if tag in image.tags or f"{tag}:latest" in image.tags:
            print(f"Image with tag '{tag}' already exists. Using existing image.")
            return client, image
    # Build the Docker image
    image, build_log = client.images.build(path=path_to_dockerfile, tag=tag)
    return client, image


def instrument_code_docker(generated_code: str, testcase_inputs: List[str], image, client, 
                           docker_working_dir = None, n_test_cases=-1, indiv_tc_timeout=5, verbose_docker=False):
    
    if docker_working_dir is None: 
        docker_working_dir = tempfile.mkdtemp()
        
    if not os.path.exists(docker_working_dir):
        raise ValueError(f"{docker_working_dir} does not exist.")
    
    code_path = os.path.join(docker_working_dir, "soln.py")
    with open(code_path, "w") as f:
        f.write(generated_code)
        
    shutil.copy(docker_driver_abs_path, os.path.join(docker_working_dir, "driver.py"))
    
    for i, testcase_input in enumerate(testcase_inputs):
        input_path = os.path.join(docker_working_dir, f"input.{i}.txt")
        with open(input_path, "w") as f:
            f.write(testcase_input)
    
    volumes = {docker_working_dir: {'bind': '/usr/src/app/tc_dir', 'mode': 'rw'}}
    
    try: 
        logging.info(f"Now running docker container for tc_gen.py with testcase_dir {docker_working_dir} and image {image}.")
        container = client.containers.run(
            image.tags[0],
            detach=True,
            volumes=volumes,
            command=f"python tc_dir/driver.py /usr/src/app/tc_dir {indiv_tc_timeout} {verbose_docker} {n_test_cases}",
        )
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
    outputs = []
    for output_file in output_files:
        with open(output_file, "r") as f:
            outputs.append(f.read())
    return outputs


def report_coherence(program_2_testcase_2_output):
    # if syntax or runtime error, then it is not coherent
    program_2_coherence = {}
    program_2_n_outputs = {}
    program_2_n_coherent = {}
    for program, testcase_2_output in program_2_testcase_2_output.items():
        n_outputs = len(testcase_2_output)
        n_coherent = len([output for output in testcase_2_output if output not in ["Syntax Error", "Runtime Error"]] )
        program_2_n_outputs[program] = n_outputs
        program_2_n_coherent[program] = n_coherent
        program_2_coherence[program] = n_coherent / n_outputs
    return program_2_coherence, program_2_n_outputs, program_2_n_coherent


# TODO: need to pay attention here
def report_accuracy(program_2_testcase_2_output, expected_outputs: List[List[str]]):
    program_2_accuracy = {}
    for (program, testcase_2_output), expected_output in zip(program_2_testcase_2_output.items(), expected_outputs):
        n_correct = len([output for output, expected_output in zip(testcase_2_output, expected_output) if output.strip() == expected_output.strip()])
        program_2_accuracy[program] = n_correct / len(testcase_2_output)
    return program_2_accuracy


def make_semantic_string(program_2_testcase_2_output, testcase_inputs): 
    program_2_semantic_string = {}
    for program, testcase_2_output in program_2_testcase_2_output.items():
        s = ""
        for i, testcase_input in enumerate(testcase_inputs):
            s += f"testcase_input: {testcase_input}, output: {testcase_2_output[i]}\n"
        program_2_semantic_string[program] = s
    semantic_strings_2_programs = defaultdict(list)
    for program, semantic_string in program_2_semantic_string.items():
        semantic_strings_2_programs[semantic_string].append(program)
    return program_2_semantic_string, semantic_strings_2_programs
    

def make_clusters_iterative(programs: List[str],
                    testcases: List[str], 
                    outputs: Optional[List[str]] = None, 
                    report_coherence=False, 
                    report_accuracy=False, 
                    n_test_cases=-1):
    if report_accuracy:
        if outputs is None:
            raise ValueError("Need outputs to report accuracy.")
        if len(testcases) != len(outputs):
            raise ValueError("Number of testcases and outputs must match.")
        
    client, tcgen_image = build_docker_image(clustering_abs_dir)
    
    program_2_testcase_2_output = {}
    for i, program in enumerate(programs):
        outputs = instrument_code_docker(program, testcases, tcgen_image, client, n_test_cases=n_test_cases)
        program_2_testcase_2_output[program] = outputs
        
    if report_coherence:
        program_2_coherence, program_2_n_outputs, program_2_n_coherent = report_coherence(program_2_testcase_2_output)
    else: 
        program_2_coherence = program_2_n_outputs = program_2_n_coherent = None
    
    if report_accuracy:
        ## TODO: convert inputs and outputs to a dict of index 2 output to avoid any sorting/ordering issues and bugs
        program_2_accuracy = report_accuracy(program_2_testcase_2_output, outputs)
    else:
        program_2_accuracy = None
        
    program_2_semantic_string, semantic_strings_2_programs = make_semantic_string(program_2_testcase_2_output, testcases)
    
    # cleanup 
    client.images.remove(tcgen_image.id)
    
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

        
    
from typing import List, Optional
import joblib
from tqdm import tqdm

def process_program(program, testcases, tcgen_image, client, n_test_cases):
    outputs = instrument_code_docker(program, testcases, tcgen_image, client, n_test_cases=n_test_cases)
    return program, outputs

def make_clusters_parallel(programs: List[str],
                           testcases_list: List[List[str]],
                           outputs_list: List[List[str]],
                           report_coherence=False,
                           report_accuracy=False,
                           n_test_cases=-1,
                           n_jobs=-1):
    if report_accuracy:
        if outputs is None:
            raise ValueError("Need outputs to report accuracy.")
        if len(testcases_list) != len(outputs_list):
            raise ValueError("Number of testcases and outputs must match.")

    client, tcgen_image = build_docker_image(clustering_abs_dir)

    program_2_testcase_2_output = {}

    with tqdm_joblib(tqdm(desc="Processing programs", total=len(programs))):
        results = joblib.Parallel(n_jobs=n_jobs, backend='threading')(
            joblib.delayed(process_program)(program, testcases, tcgen_image, client, n_test_cases)
            for program, testcases in zip(programs, testcases_list)
        )

    for program, outputs in results:
        program_2_testcase_2_output[program] = outputs

    if report_coherence:
        program_2_coherence, program_2_n_outputs, program_2_n_coherent = report_coherence(program_2_testcase_2_output)
    else:
        program_2_coherence = program_2_n_outputs = program_2_n_coherent = None

    if report_accuracy:
        program_2_accuracy = report_accuracy(program_2_testcase_2_output, ...) 
    else:
        program_2_accuracy = None

    program_2_semantic_string, semantic_strings_2_programs = make_semantic_string(program_2_testcase_2_output, testcases)
    
    # cleanup 
    client.images.remove(tcgen_image.id)

    return program_2_semantic_string, semantic_strings_2_programs, program_2_coherence, program_2_n_outputs, program_2_n_coherent, program_2_accuracy


@dataclass
class Config:
    input_file: str
    output_dir: str
    report_coherence: bool
    report_accuracy: bool
    n_test_cases: int
    n_jobs: int

def read_config(config_path: str) -> Config:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)


## TODO: you should refactor all things to use testcase_id: input, testcase_id: output to be more explicit and to reduce any risks of bugs

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
    

