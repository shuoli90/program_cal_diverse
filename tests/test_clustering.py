import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.clustering import clustering
import tempfile
import pytest
import shutil
import numpy as np
from collections import defaultdict

## helper
def dicts_are_close(dict1, dict2, atol=1e-8):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if not np.isclose(dict1[key], dict2[key], atol=atol):
            return False
    return True

def test_lite_build_docker_image():
    # Setup temporary directory and Dockerfile
    temp_dir = tempfile.mkdtemp()
    temp_dockerfile_path = os.path.join(temp_dir, 'Dockerfile')
    dockerfile_content = """
    FROM python:3.8-slim
    RUN pip install --no-cache-dir pytest
    """
    with open(temp_dockerfile_path, 'w') as f:
        f.write(dockerfile_content)

    # Test
    client, image = clustering.build_docker_image(temp_dir)
    assert image.tags[0] == 'python-test-case-runner:latest'

    # Cleanup
    client.images.remove(image.id)
    shutil.rmtree(temp_dir)  # Remove temporary directory

    
def test_real_build_docker_image():
    client, image = clustering.build_docker_image(clustering.clustering_abs_dir)
    assert client is not None
    assert image is not None

    # Check if the image has the correct tag
    assert 'python-test-case-runner:latest' in image.tags

    client.images.remove(image.id)
    

generated_code_squared = """
import sys

def main():
    for line in sys.stdin:
        print(f'Input: {line.strip()}')
        num = int(line.strip())
        print(f'Squared: {num ** 2}')

if __name__ == '__main__':
    main()
"""

def test_instrument_generated_code_squared_docker():
    testcase_inputs = ["3", "4"]
    expected_outputs = [
        "Input: 3\nSquared: 9",
        "Input: 4\nSquared: 16"
    ]
    client, image = clustering.build_docker_image(clustering.clustering_abs_dir)
    docker_working_dir = tempfile.mkdtemp()

    # Test
    outputs = clustering.instrument_code_docker(
        generated_code_squared, 
        testcase_inputs, 
        image, 
        client, 
        docker_working_dir, 
        n_test_cases=-1, 
        indiv_tc_timeout=20, 
        verbose_docker=True
    )

    # Check the outputs
    assert len(outputs) == len(testcase_inputs)
    for output, expected_output in zip(outputs, expected_outputs):
        assert output.strip() == expected_output

    # Cleanup
    shutil.rmtree(docker_working_dir)
    client.images.remove(image.id)


def test_report_coherence():
    program_2_testcase_2_output = {
        'prog1': ['Output 1', 'Syntax Error', 'Output 2'],
        'prog2': ['Runtime Error', 'Output 3', 'Output 4']
    }
    expected_coherence = {'prog1': 2/3, 'prog2': 2/3}
    expected_n_outputs = {'prog1': 3, 'prog2': 3}
    expected_n_coherent = {'prog1': 2, 'prog2': 2}

    coherence, n_outputs, n_coherent = clustering.report_coherence(program_2_testcase_2_output)

    assert dicts_are_close(coherence, expected_coherence)
    assert n_outputs == expected_n_outputs
    assert n_coherent == expected_n_coherent


def test_make_semantic_string():
    program_2_testcase_2_output = {
        'prog1': ['Output 1', 'Output 2'],
        'prog2': ['Output 3', 'Output 4']
    }
    testcase_inputs = ['Input 1', 'Input 2']
    expected_semantic_string = {
        'prog1': "testcase_input: Input 1, output: Output 1\ntestcase_input: Input 2, output: Output 2\n",
        'prog2': "testcase_input: Input 1, output: Output 3\ntestcase_input: Input 2, output: Output 4\n"
    }
    expected_semantic_strings_2_programs = defaultdict(list, {
        "testcase_input: Input 1, output: Output 1\ntestcase_input: Input 2, output: Output 2\n": ['prog1'],
        "testcase_input: Input 1, output: Output 3\ntestcase_input: Input 2, output: Output 4\n": ['prog2']
    })

    semantic_string, semantic_strings_2_programs = clustering.make_semantic_string(program_2_testcase_2_output, testcase_inputs)

    assert semantic_string == expected_semantic_string
    assert dict(semantic_strings_2_programs) == dict(expected_semantic_strings_2_programs)


def test_report_accuracy():
    program_2_testcase_2_output = {
        'prog1': ['Output 1', 'Output 2', 'Output 3'],
        'prog2': ['Output 1', 'Wrong Output', 'Output 3']
    }
    expected_outputs = ['Output 1', 'Output 2', 'Output 3']
    expected_accuracy = {'prog1': 1.0, 'prog2': 2/3}

    accuracy_output = clustering.report_accuracy(program_2_testcase_2_output, expected_outputs)

    assert dicts_are_close(accuracy_output, expected_accuracy)

    
def test_make_clusters_iterative():
    # Setup
    programs = ["print(input())", "print(int(input()) ** 2)"]
    testcases = ["3", "4"]
    generations = ["gen1", "gen2"]
    expected_outputs = ["3", "16"]
    expected_semantic_strings = {
        programs[0]: "testcase_input: 3, output: 3\ntestcase_input: 4, output: 4\n",
        programs[1]: "testcase_input: 3, output: 9\ntestcase_input: 4, output: 16\n"
    }

    # Test
    results = clustering.make_clusters_iterative(
        programs, testcases, generations, report_accuracy=True, outputs=expected_outputs
    )
    program_2_semantic_string, semantic_strings_2_programs, program_2_coherence, program_2_n_outputs, program_2_n_coherent, program_2_accuracy = results

    # Check the results
    assert program_2_semantic_string == expected_semantic_strings
    assert program_2_accuracy[programs[0]] == 1.0
    assert program_2_accuracy[programs[1]] == 0.5
    assert dicts_are_close(program_2_coherence, {'prog1': 1.0, 'prog2': 1.0})
    assert program_2_n_outputs == {'prog1': 2, 'prog2': 2}
    assert program_2_n_coherent == {'prog1': 2, 'prog2': 2}

def test_make_clusters_parallel():
    # Setup
    programs = ["print(input())", "print(int(input()) ** 2)"]
    testcases = ["3", "4"]
    generations = ["gen1", "gen2"]
    expected_outputs = ["3", "16"]
    expected_semantic_strings = {
        programs[0]: "testcase_input: 3, output: 3\ntestcase_input: 4, output: 4\n",
        programs[1]: "testcase_input: 3, output: 9\ntestcase_input: 4, output: 16\n"
    }

    # Test
    results = clustering.make_clusters_parallel(
        programs, testcases, generations, report_accuracy=True, outputs=expected_outputs, n_jobs=2
    )
    program_2_semantic_string, semantic_strings_2_programs, program_2_coherence, program_2_n_outputs, program_2_n_coherent, program_2_accuracy = results

    # Check the results
    assert program_2_semantic_string == expected_semantic_strings
    assert program_2_accuracy[programs[0]] == 1.0
    assert program_2_accuracy[programs[1]] == 0.5
    assert dicts_are_close(program_2_coherence, {'prog1': 1.0, 'prog2': 1.0})
    assert program_2_n_outputs == {'prog1': 2, 'prog2': 2}
    assert program_2_n_coherent == {'prog1': 2, 'prog2': 2}

    

if __name__ == '__main__':
    pytest.main()

    
    