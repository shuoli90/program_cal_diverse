import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import clustering


if __name__ == '__main__':
    outputs = [[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]]
    results = clustering.cluster(outputs)

    print("Results:")
    for unique, count in zip(*results):
        print(f"Unique: {unique} Count: {count}")
    