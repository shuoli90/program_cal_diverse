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
import requests
import warnings

if __name__ == "__main__":

    def test_format_open_ended_code():
        f_code = """
def f(N, A):
    if len(A) != N:
        print("Length of A is not N")
    else: 
        print(sum(A))
"""
        extract_arguments_code = """
def extract_arguments(fh):
    N = int(fh.readline().strip())
    A = list(map(int, fh.readline().strip().split()))
    return N, A
"""


        # Assuming the function already reads from the predefined path
        result = clustering.format_open_ended_code(f_code, extract_arguments_code)
        assert f_code in result, "f_code not in result"
        assert extract_arguments_code in result, "extract_arguments_code not in result"

        # Print out the formatted code for inspection
        print("Formatted Code:\n", result)

        print("test_format_open_ended_code passed")
        return result

    results = test_format_open_ended_code()

    
    testcases = {
        "0": "3\n1 2 3\n",
        "1": "2\n1 2\n",
        "2": "4\n1 2 3\n"
    }
    expected_outputs = {
        "0": "6",
        "1": "3",
        "2": "Length of A is not N"
    }
    results = clustering.make_clusters_iterative(
        [results], testcases, expected_outputs, do_report_coherence=True, do_report_accuracy=True, n_test_cases=-1, verbose_docker=True, 
    )
    program_2_semantic_string, semantic_strings_2_programs, program_2_coherence, program_2_n_outputs, program_2_n_coherent, program_2_accuracy = results
    from pprint import pprint
    print("program_2_semantic_string:")
    for k, v in program_2_semantic_string.items():
        print(f"program: {k}\nsemantic_string: {v}")
    print("program_2_n_outputs:")
    for k, v in program_2_n_outputs.items():
        print(f"program: {k}\nn_outputs: {v}")
    print("program_2_n_coherent:")
    for k, v in program_2_n_coherent.items():
        print(f"program: {k}\nn_coherent: {v}")
    print("program_2_accuracy:")
    for k, v in program_2_accuracy.items():
        print(f"program: {k}\naccuracy: {v}")
    
    
        
    
    # pprint("program_2_n_outputs:\n", program_2_n_outputs)
    # pprint("program_2_n_coherent:\n", program_2_n_coherent)
    # pprint("program_2_accuracy:\n", program_2_accuracy)
    
        