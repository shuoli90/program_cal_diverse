# program_cal_diverse
This is the repo for evaluating calibration and diversity for program generation.

## Structure
We structure the implementation by functionality. Here is a brief description of folders:
- tasks: this folder contains scripts for pre-processing nlp datasets. Each script contains preprocessing steps for datasets under the same nlp task. The list of nlp tasks and corresponding datasets are listed in README file.
- models: this folder prepare opensource models and OpenAI APIs. The preparation operations include: 1, load in models or setup API; 2, setup generation pipeline. Users are supposed to bring their own OpenAI account.
- metrics: this folder contains implementations for generation correctness, calibration and diversity metrics.
- utils: this folder contains miscellaneous functions.
- run: this folder contains scripts that are supposed to be run by users. Functions include generating responses, calibrating LLM using a specific indicator, comparing multiple indicators.

## Simple utilization
```
cd run
python generate_code.py --prompt 'Generate a function to calculate the factorial of a number.' [--temperature TEMPERATURE] [--max_length LENGTH] [--num_return_sequences NUM_GENERATIONS]
```

## Dataset

The dataset is available [here](https://drive.google.com/drive/folders/1b7s_PKRbTgjWL45fhD3zRAceognosOWM?usp=sharing). The directory has three kinds of files 

1. `{train, valid, test}_descriptions_and_testcases.jsonl`
2. `{train, valid, test}_examples.jsonl`
3. `{train, valid, test}_examples_cpp.jsonl`
4. `{train, valid, test}_examples_big.jsonl`
5. `{train, valid, test}_examples_big_cpp.jsonl`

[1] contains the competitive programming problem description and the test cases that ship for that problem in codenet
[2 - 5] contain examples of solutions for each of these problems. If there is no `_cpp` then the examples are in Python, otherwise they are in C++. The `_big` files contain the same data as the non `_big` files, but with more examples per problem.