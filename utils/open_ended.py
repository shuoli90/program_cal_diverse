import openai 
from tenacity import * 
import json 

import openai 
PATH_TO_KEY="/home/data1/pie_data_sept_2023/openai_key.txt"
with open(PATH_TO_KEY, 'r') as f:
    key = f.read().strip()
PATH_TO_ORG="/home/data1/pie_data_sept_2023/openai_org.txt"
with open(PATH_TO_ORG, 'r') as f:
    org = f.read().strip()
openai.organization = org
openai.api_key = key

default_model = "gpt-3.5-turbo-0125"

@retry(
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type((openai.error.APIConnectionError, openai.error.APIError, json.JSONDecodeError)),
)
def get_general_response(prompt, model=default_model, temperature=1.0, n=1, top_p=1.0): 
  completion = openai.ChatCompletion.create(
    model = model,
    messages = [
      {'role': 'system', 'content': 'You are a helpful assistant for software engineering and programming tasks.'},
      {'role': 'user', 'content': f'{prompt}'}
    ],
    temperature = temperature,
    n=n,
    top_p=top_p
  )
  return [c["message"]["content"] for c in completion["choices"]]
#   return completion["choices"][0]["message"]["content"]
    


example_problem = """
### Problem Statement: Count and Modify

**Description**

You are building a system that processes data based on a character command. The system takes in an array of integers and a character, which specifies a type of operation to perform. Each integer in the array represents a count of operations performed by a device in a network of devices.

**Input**

1. An integer \(N\) (1 ≤ \(N\) ≤ 10^3), representing the number of devices in a network.
2. A list of integers \(A\) of size \(N\), where each integer \(A[i]\) (0 ≤ \(A[i]\) ≤ 10^5) represents the number of operations performed by the \(i\)-th device.
3. A character \(C\) from the set {'a', 'b', 'c', 'd', 'e'}, which determines the type of operation to perform on the array.

**Operations**

- 'a': Increment all the counts by 1.
- 'b': Decrement all the counts by 1 (no count should go below zero).
- 'c': Double the counts of all devices.
- 'd': Halve the counts of all devices (use integer division).
- 'e': Reset all counts to zero.

**Output**

Print the modified list of integers after applying the operation specified by \(C\).

**Examples**

_Input_
```
5
3 10 0 7 15
b
```

_Output_
```
2 9 0 6 14
```

_Input_
```
3
4 5 6
c
```

_Output_
```
8 10 12
```

**Explanation**

In the first example, the operation 'b' decrements each count by 1. Since the third device already has a count of 0, it remains 0 after the operation.

In the second example, the operation 'c' doubles the count for each device, resulting in [8, 10, 12].

**Note**

This problem is designed to test basic array manipulation and conditional operations based on character input, suitable for beginner level competitive programmers.
"""

example_input = """
An integer N (1 ≤ N ≤ 10^3), representing some quantity or size.
A list of integers A of size N, where each integer is between 0 and 1e5.
A character C from the set {'a', 'b', 'c', 'd', 'e'}.

### Example Input:

```
5
3 10 0 7 15
b
```

### Function Signature:
Write a function f(N, A, C) that takes in the input. 
def f(N: int, A: List[int], C: str): 
    ''' 
    N: an integer 
    A: a list of integers
    C: a character
    '''
    

"""

example_tcgen = """
def tcgen(): 
    N = random.randint(1, 10**3)
    
    A = [random.randint(0, 10**5) for _ in range(N)]

    C = random.choice(['a', 'b', 'c', 'd', 'e'])
    
    return N, A, C
"""

example_extract_arguments = """
def f(N, A, C):
    ....

def extract_arguments(fh):
    N = int(fh.readline().strip())
    A = list(map(int, fh.readline().strip().split()))
    C = fh.readline().strip()
    return N, A, C

if __name__ == "__main__":
    input_path = sys.argv[1]
    with open(input_path, 'r') as fh: 
        N, A, C = extract_arguments(fh)
    f(N, A, C)
    
"""


def prompt_gpt_to_generate_input_description(new_problem_statement, orig_problem_statement, orig_input_description, model=default_model, temperature=0.5, n=1, top_p=1.0):
    prompt = f"""
    Can you extract and also canonicalize the input specification for this competitive programming problem? 
    To canonicalize, make the input specification broad to the input types and ranges, and disassociate it from the 
    problem description itself. Ideally, include a short example of simple inputs as a string that will be read in from a file / stdin.
    And also include a function signature that processes all the inputs (do not specify the output type, just the inputs).
    If the inputs can be a tuple: for example, (N, A, C), then please include a function signature that processes the tuple, f(N, A, C). 
    If the inputs are a list of such tuples, then include a function signature that processes the list of tuples, f(inputs). 
    Here is an example of a problem statement and input description:
    Problem Statement:
    {orig_problem_statement}
    Canonicalized Input Description: 
    {orig_input_description}
    
    Here is a the problem statement that you need to extract the input description from:
    Problem Statement:
    {new_problem_statement}
    Now, please extract and canonicalize the input description, please try to anonymize and obfuscate away anything related to the problem statement
    and only mention the ranges/types of the input. 
    """
    return get_general_response(prompt, model=model, temperature=temperature, n=n, top_p=top_p)


def prompt_gpt_to_write_testcase_generator(new_problem_statement, orig_problem_statement, orig_tcgen, model=default_model, temperature=0.5, n=1, top_p=1.0):
    prompt = f"""
    Can you write a test case generator for this competitive programming problem? 
    Here is an example of a problem statement and test case generator:
    Problem Statement:
    {orig_problem_statement}
    Test Case Generator:
    {orig_tcgen}
    
    Here is a the problem statement that you need to write a test case generator for:
    Problem Statement:
    {new_problem_statement}
    """
    return get_general_response(prompt, model=model, temperature=temperature, n=n, top_p=top_p)

def prompt_gpt_to_write_extract_arguments(new_problem_statement, orig_problem_statement, orig_extract_arguments, model=default_model, temperature=0.5, n=1, top_p=1.0):
    prompt = f"""
    Can you write a function to extract the arguments from the input file for this competitive programming problem? 
    Then also write the function that processes the arguments. 
    If the inputs can be a tuple: for example, (N, A, C), then please include a function signature that processes the tuple, f(N, A, C).
    If the inputs are a list of such tuples, then include a function signature that processes the list of tuples, f(inputs).
    Here is an example of a problem statement and extract arguments function:
    Problem Statement:
    {orig_problem_statement}
    Extract Arguments Function:
    {orig_extract_arguments}
    
    Here is a the problem statement that you need to write an extract arguments function for:
    Problem Statement:
    {new_problem_statement}
    """
    return get_general_response(prompt, model=model, temperature=temperature, n=n, top_p=top_p)
  
def format_html(orig_description, specification, tcgen, extract_args, problem_no="NA", generation="1"): 
    html_template = """
    <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Problem {problem_no} - Generation {generation}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f4f4f4;
                }}
                .code-container {{
                    display: flex;
                    flex-wrap: wrap; /* Allows wrapping if the total width exceeds the viewport */
                    justify-content: space-between;
                    margin-bottom: 20px;
                }}
                .code-box {{
                    width: 22%; /* Adjusted width to fit all four */
                    min-width: 200px; /* Ensures that boxes don't get too small */
                    background-color: #ffffff;
                    border: 1px solid #e0e0e0;
                    padding: 10px;
                    box-shadow: 1px 1px 5px rgba(0, 0, 0, 0.1);
                    margin: 0px; /* Adds some spacing between boxes */
                }}
                .code-box pre {{
                    white-space: pre-wrap;
                    background-color: #eee;
                    padding: 10px;
                    border-radius: 4px;
                    overflow-x: auto;
                }}
                .explanation {{
                    background-color: #ffffff;
                    border: 1px solid #e0e0e0;
                    padding: 10px;
                    box-shadow: 1px 1px 5px rgba(0, 0, 0, 0.1);
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <h1>Problem {problem_no} - Generation {generation}</h1>
            <div class="code-container">
                <!-- Original Description -->
                <div class="code-box">
                    <h2>Orig Description</h2>
                    <pre>{orig_description}</pre>
                </div>
                <!-- Extracted Specification -->
                <div class="code-box">
                    <h2>Extracted Specification</h2>
                    <pre>{specification}</pre>
                </div>
                <!-- TC Generator -->
                <div class="code-box">
                    <h2>Test Case Generator</h2>
                    <pre>{tcgen}</pre>
                </div>
                <!-- Extract Arguments -->
                <div class="code-box">
                    <h2>Extract Arguments</h2>
                    <pre>{extract_args}</pre>
                </div>
            </div>
        </body>
        </html>
    """

    return html_template.format(orig_description=orig_description, specification=specification, tcgen=tcgen, 
                                problem_no=problem_no, generation=generation, extract_args=extract_args)

# def make_html_from_programs(src_program, tgt_program, generated_program, speedup_tgt, speedup_generated, good_model="gpt-4", use_gpt=False):
#   if use_gpt:
#     print("Using GPT to generate src vs tgt explanation")
#     nl_explanation = get_natural_language_description_of_code(src_program, model=good_model)
#     src_tgt_explanation = prompt_gpt_to_understand_program(src_program, tgt_program, speedup_tgt, model=good_model)
#     src_gen_explanation = prompt_gpt_to_understand_program(src_program, generated_program, speedup_generated, model=good_model)
#     if speedup_generated > speedup_tgt:
#         tgt_gen_explanation = prompt_gpt_to_understand_program(tgt_program, generated_program, speedup_generated, model=good_model)
#     else: 
#         tgt_gen_explanation = f"tgt was not faster than src, so no explanation generated with speedup {speedup_generated} less than {speedup_tgt}"
#   else: 
#     nl_explanation = "not enough funds; only done for top programs"
#     src_tgt_explanation = "not enough funds; only done for top programs"
#     src_gen_explanation = "not enough funds; only done for top programs"
#     tgt_gen_explanation = "not enough funds; only done for top programs"
#   src_program = src_program.replace("\n", "<br>")
#   tgt_program = tgt_program.replace("\n", "<br>")
#   generated_program = generated_program.replace("\n", "<br>")
#   return format_html(generated_program, tgt_program, src_program, nl_explanation,
#                      src_tgt_explanation, src_gen_explanation, tgt_gen_explanation, speedup_tgt, speedup_generated)

# def make_n_html_from_programs(new_description, n, temp, top_p, model=default_model, example_problem=example_problem, example_input=example_input, example_tcgen=example_tcgen, problem_no="NA"): 
#     n_input_descriptions = prompt_gpt_to_generate_input_description(new_description, example_problem, example_input, model=model, temperature=temp, n=n, top_p=top_p)
#     n_tc_gens = prompt_gpt_to_write_testcase_generator(new_description, example_problem, example_tcgen, model=model, temperature=temp, n=n, top_p=top_p)
#     return [format_html(new_description, n_input_descriptions[i], n_tc_gens[i], problem_no=problem_no, generation=str(i+1)) for i in range(n)]

def make_n_html_from_programs(new_description, n, temp, top_p, model=default_model, example_problem=example_problem, example_input=example_input, example_tcgen=example_tcgen, problem_no="NA"):
    n_input_descriptions = prompt_gpt_to_generate_input_description(new_description, example_problem, example_input, model=model, temperature=temp, n=n, top_p=top_p)
    n_tc_gens = prompt_gpt_to_write_testcase_generator(new_description, example_problem, example_tcgen, model=model, temperature=temp, n=n, top_p=top_p)
    n_extract_args = prompt_gpt_to_write_extract_arguments(new_description, example_problem, example_extract_arguments, model=model, temperature=temp, n=n, top_p=top_p)
    # html_outputs = [format_html(new_description, n_input_descriptions[i], n_tc_gens[i], problem_no=problem_no, generation=str(i+1)) for i in range(n)]
    html_outputs = [format_html(new_description, n_input_descriptions[i], n_tc_gens[i], n_extract_args[i], problem_no=problem_no, generation=str(i+1)) for i in range(n)]
    return html_outputs, n_input_descriptions, n_tc_gens, n_extract_args
  
  
def print_model_list(): 
    response = openai.Model.list()

    # Extract and print the model names
    for model in response["data"]:
        print(model["id"])