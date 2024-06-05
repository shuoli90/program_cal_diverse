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
    
# def extract_python_code(text):
    # # Define a regular expression pattern to capture Python code blocks
    # pattern = re.compile(r'def\s+.*?if\s+__name__\s+==\s+"__main__":\s+main\(\)', re.DOTALL)
    # # Search for the pattern in the provided text
    # match = pattern.search(text)
    # if match:
    #     # If a match is found, return the code block
    #     return match.group()
    # else:
    #     # If no match is found, return a message indicating no code was found
    #     # return "No Python code found."
    #     pass
    
def extract_python_code(text):
    import re

    lines = text.splitlines()
    python_code = ""
    block = ""
    stack = []
    block_active = False
    previous_indent = None

    for line in lines:
        stripped_line = line.lstrip()
        leading_spaces = len(line) - len(stripped_line)

        # Check if the line starts a new block or is a continuation of a block
        if re.match(r"(def\s+\w+\s*\(|if\s+__name__\s*==\s*\"__main__\":)", stripped_line) and not block_active:
            if not block_active or leading_spaces > previous_indent:
                if block_active:
                    python_code += block.strip() + "\n\n"
                    block = ""
                block_active = True
                previous_indent = leading_spaces
                stack.append(leading_spaces)
            block += line + "\n"
            continue

        # Handle import statements outside of block checks (ie. at the beginning of the file)
        if re.match(r"(from\s+[\w\.]+\s+import\s+[\w\.,\s*]+|import\s+[\w\.,\s*]+)", stripped_line) and not block_active:
            python_code += stripped_line + "\n"
            continue

        # If currently active in a block and the line is part of it
        if block_active and leading_spaces > previous_indent:
            block += line + "\n"
        else:
            # If the line is not part of an active block, close off any active block
            if block_active:
                # we should actually take the previous-indent and de-dent the block
                lines = block.splitlines()
                block = "\n".join([line[previous_indent:] for line in lines])
                # add the block to the python code
                python_code += block.strip() + "\n\n"
                block = ""
                block_active = False
                stack.clear()

    # Add the last block if there's any (ie EOF I think w/out any dedent)
    if block_active and block:
        lines = block.splitlines()
        block = "\n".join([line[previous_indent:] for line in lines])
        python_code += block.strip() + "\n"

    return python_code.strip()
