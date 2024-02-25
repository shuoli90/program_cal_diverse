import re

def vanilla_template(description):
    verbalized = f"Generate a program completion in python for the given function description: {description}"
    return verbalized

def code_template(program):
    generated_code_squared = f"""
import sys

{program} """ + """

def main():
    for line in sys.stdin:
        print(f'Input: {line.strip()}')
        num = int(line.strip())
        print(f'Factorial: {factorial(num)}')

if __name__ == '__main__':
    main()
    """
    return generated_code_squared

# def extract_function(text):
#     # extract text in ''' or """
#     pattern = r'```[^\n]*\n(.*?)\n```'
#     text = re.findall(pattern, text, re.DOTALL)
#     if len(text) == 0:
#         return None
#     else:
#         text = text[0]
#     text = text.split('\n\n')[0]
#     # Regular expression pattern to match a function definition
#     pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)\s*:\s*(.*?)\n\s*'
#     match = re.search(pattern, text, re.DOTALL)
#     if match:
#         function_name = match.group(1)
#         function_parameters = match.group(2)
#         function_body = text
#         return function_name, function_parameters, function_body
#     else:
#         return None

def extract_function(text):
    pattern = r'```[^\n]*\n(.*?)\n```'
    text_matched = re.findall(pattern, text, re.DOTALL)
    if len(text_matched) == 0:
        return None
    else:
        text = text_matched[0]
        return text
    
# def extract_function(text):
#     # Regular expression pattern to match Python code
#     pattern = r'```python(.*?)```'
#     # Find all matches
#     matches = re.findall(pattern, text, re.DOTALL)
#     # Extract and return the code
#     return [match.strip() for match in matches]