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

    programs = ["print(input())", "print(int(input()) ** 2)"]
    testcases = {"0": "3", "1": "4"}
    expected_outputs = {"0": "9", "1": "16"}
    expected_semantic_strings = {
        programs[0]: "testcase_input: 3, output: 3\ntestcase_input: 4, output: 4\n",
        programs[1]: "testcase_input: 3, output: 9\ntestcase_input: 4, output: 16\n"
    }

    # Test
    results = clustering.make_clusters_iterative(
        programs, testcases, expected_outputs, do_report_coherence=True, do_report_accuracy=True, n_test_cases=-1, verbose_docker=True
    )
    program_2_semantic_string, semantic_strings_2_programs, program_2_coherence, program_2_n_outputs, program_2_n_coherent, program_2_accuracy = results