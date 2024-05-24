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
    
test_format_open_ended_code()