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

# def extract_function(text):
#     pattern = r'```[^\n]*\n(.*?)\n```'
#     text_matched = re.findall(pattern, text, re.DOTALL)
#     if len(text_matched) == 0:
#         return None
#     else:
#         text = text_matched[0]
#         return text
    
def extract_python_code(text):
    # Define a regular expression pattern to capture Python code blocks
    pattern = re.compile(r'def\s+.*?if\s+__name__\s+==\s+"__main__":\s+main\(\)', re.DOTALL)
    # Search for the pattern in the provided text
    match = pattern.search(text)
    if match:
        # If a match is found, return the code block
        return match.group()
    else:
        # If no match is found, return a message indicating no code was found
        # return "No Python code found."
        pass