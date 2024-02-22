import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text from a prompt')
    parser.add_argument('--data_dir', type=str, default='../data', help='data directory')
    parser.add_argument('--data_file', type=str, default='codenet', help='data file')
    parser.add_argument('--split', type=str, default='test', help='data split')
    args = parser.parse_args()

    description_dir = os.path.join(args.data_dir, args.data_file, f'{args.split}_descriptions_and_testcases.jsonl')
    with open(description_dir, 'r') as f:
        data = [json.loads(line) for line in f]
    breakpoint()


