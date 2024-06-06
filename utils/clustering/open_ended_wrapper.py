## TEMPLATE FOR OPEN-ENDED EXECUTION
from typing import * 
import sys
import os

from typing import TextIO

"""
high level idea for f and extract_arguments: 

def f(...):
    pass
    
def extract_arguments(fh: TextIO) -> Tuple:
    N = int(fh.readline().strip())
    A = list(map(int, fh.readline().strip().split()))
    C = list(map(int, fh.readline().strip().split()))
    return N, A, C
"""

## REPLACE F 

## REPLACE EXTRACT_ARGUMENTS



if __name__ == "__main__":
    input_path = sys.argv[1]
    with open(input_path, 'r') as fh: 
        inp = extract_arguments(fh)
    if type(inp) == tuple:
        f(*inp)
    else:
        f(inp)
    # exit
    
        