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

# def extract_formatted_code(text):
#     # Updated regex to handle optional language specifiers and varying whitespace
#     text = re.sub(r"```Python\n", "```", text)
    
#     pattern = r"```(.*?)```"
    
#     # re.DOTALL allows dot (.) to match across multiple lines
#     matches = re.findall(pattern, text, re.DOTALL)
    
#     if matches:
#         # import pdb; pdb.set_trace()
#         # Remove all matched code blocks from the text
#         # text = re.sub(pattern, "", text)
#         for match in matches:
#             text = re.sub(match, "", text, count=1)
#         # Join all code blocks with a newline and ensure a newline at the end of each block
#         combined_code_blocks = '\n'.join(match.strip() for match in matches)
#         return combined_code_blocks, text
#     return "", text  # Return an empty string if no matches are found

# def extract_formatted_code(text):
#     text = re.sub(r"```Python\n", "```\n", text)
#     lines = text.split('\n')  # Split the input text into lines
#     inside_code_block = False  # State to keep track of whether we're inside a code block
#     code_blocks = []  # List to hold all extracted code blocks
#     current_code = []  # Buffer to hold the current code block content
    
#     non_matching_code = []

#     for line in lines:
#         if line.strip().startswith('```'):  # Check if the line starts with ```
#             if inside_code_block:
#                 # If already inside a block, this ``` signifies the end
#                 code_blocks.append('\n'.join(current_code).strip())
#                 current_code = []  # Reset current code block buffer
#                 inside_code_block = False  # Toggle the state to 'not in a block'
#             else:
#                 # If not inside a block, this ``` signifies the start of a new block
#                 inside_code_block = True
#         elif inside_code_block:
#             # If we're inside a code block, add this line to the current code block buffer
#             current_code.append(line)
#         else:
#             non_matching_code.append(line)
    
#     if current_code:
#         code_blocks.append('\n'.join(current_code).strip())

#     return '\n\n'.join(code_blocks), '\n'.join(non_matching_code)


statement_patterns = re.compile(r"""
                                def\s+\w+\s*\(|
                                class\s+\w+|
                                if\s+__name__\s*==\s*\"__main__\":|
                                while\s+.*:\s*$|           # while statements, ends with colon and optional whitespace until end of line    
                                for\s+\w+\s+in\s+.*:\s*$|  # for statements
                                if\s+.*:\s*$|              # if statements
                                elif\s+.*:\s*$|            # elif statements
                                else\s*:\s*$|              # else statements
                                try\s*:\s*$|               # try statements
                                except\s+.*:\s*$|          # except statements
                                finally\s*:\s*$|           # finally statements
                                with\s+.*:\s*$|            # with statements
                                async\s+def\s+\w+\s*\(  # async def statements
                                """, re.VERBOSE)
    
def extract_python_code(text):
    import ast 
    import re
    
    # formatted_code, text = extract_formatted_code(text)
    # this is to handle markdown formatted code blocks
    text = re.sub(r"```Python\n", "```\n", text)
    text = re.sub(r"```", "```\n", text)
    lines = text.splitlines()
    python_code = ""
    block = ""
    stack = []
    block_active = False
    previous_indent = 0

    for line in lines:
        # if "while True:" in line:   
        #     import pdb; pdb.set_trace()
        stripped_line = line.lstrip()
        leading_spaces = len(line) - len(stripped_line)

        # Check if the line starts a new block or is a continuation of a block
        
        if statement_patterns.match(stripped_line):
            # import pdb; pdb.set_trace()
            if not block_active or leading_spaces > previous_indent:
                # if block_active:
                #     python_code += block.strip() + "\n\n"
                #     block = ""
                if not block_active: 
                    previous_indent = leading_spaces
                block_active = True
                # stack.append(leading_spaces)
            block += line + "\n"
            continue

        # Handle import statements outside of block checks (ie. at the beginning of the file), decorators
        # if re.match(r"(from\s+[\w\.]+\s+import\s+[\w\.,\s*]+|import\s+[\w\.,\s*]+)", stripped_line) and not block_active:
        if not block_active and stripped_line and not stripped_line.startswith('@'):
            try: 
                parsed_node = ast.parse(stripped_line).body
                if parsed_node and isinstance(parsed_node[0], ast.stmt) and not isinstance(parsed_node[0], ast.Expr): 
                    python_code += stripped_line + "\n"
                    continue
            except SyntaxError:
                pass
        
        if re.match(r"(from\s+[\w\.]+\s+import\s+[\w\.,\s*]+|import\s+[\w\.,\s*]+|@)", stripped_line) and not block_active:
            python_code += stripped_line + "\n"
            continue

        # If currently active in a block and the line is part of it, or empty line
        if block_active and (leading_spaces > previous_indent or not stripped_line):
            block += line + "\n"
            continue
        
        # if empty line, skip
        # if not stripped_line and block_active:
            
        #     continue
        
        else:
            # If the line is not part of an active block, close off any active block
            if block_active:
                # import pdb; pdb.set_trace()
                # we should actually take the previous-indent and de-dent the block
                lines = block.splitlines()
                block = "\n".join([line[previous_indent:] for line in lines])
                # add the block to the python code
                python_code += block.strip() + "\n\n"
                block = ""
                block_active = False
                # stack.clear()

    # Add the last block if there's any (ie EOF I think w/out any dedent)
    if block_active and block:
        lines = block.splitlines()
        block = "\n".join([line[previous_indent:] for line in lines])
        python_code += block.strip() + "\n"

    return python_code.strip()
    
    
    
    
    
# def extract_python_code(text):
#     import re

#     lines = text.splitlines()
#     python_code = ""
#     block = ""
#     # stack = []
#     block_active = False
#     previous_indent = None
    
#     current_indent = 0

#     for line in lines:
#         stripped_line = line.lstrip()
#         leading_spaces = len(line) - len(stripped_line)
        
#         # Detect decorators and handle them as part of the upcoming block
#         if stripped_line.startswith('@'):
#             block += line + "\n"
#             continue
        
#         # skip if line is empty
#         if not stripped_line:
#             continue

#         # Check if the line starts a new block or is a continuation of a block
#         # if re.match(r"(def\s+\w+\s*\(|if\s+__name__\s*==\s*\"__main__\":)", stripped_line) and not block_active:
#         if re.match(r"(def\s+\w+\s*\(|class\s+\w+|if\s+__name__\s*==\s*\"__main__\":)", stripped_line):
#             if not block_active or leading_spaces > previous_indent:
#                 # if block_active:
#                 #     lines = block.splitlines()
#                 #     block = "\n".join([line[previous_indent:] for line in lines])
#                 #     python_code += block.strip() + "\n\n"
#                 #     block = ""
#                 block_active = True
#                 previous_indent = leading_spaces
#                 # stack.append(leading_spaces)
#             else: 
#                 pass
#                 # lines = block.splitlines()
#                 # block = "\n".join([line[previous_indent:] for line in lines])
#                 # python_code += block.strip() + "\n\n"
                
#             block += line + "\n"
#             continue

#         # Handle import statements outside of block checks (ie. at the beginning of the file)
#         if re.match(r"(from\s+[\w\.]+\s+import\s+[\w\.,\s*]+|import\s+[\w\.,\s*]+)", stripped_line) and not block_active:
#             python_code += stripped_line + "\n"
#             # hoist 
#             continue

#         # If currently active in a block and the line is part of it
#         if block_active and leading_spaces > previous_indent:
#             block += line + "\n"
#         else:
#             # If the line is not part of an active block, close off any active block
#             if block_active:
#                 # we should actually take the previous-indent and de-dent the block
#                 lines = block.splitlines()
#                 block = "\n".join([line[previous_indent:] for line in lines])
#                 # add the block to the python code
#                 python_code += block.strip() + "\n\n"
#                 block = ""
#                 block_active = False
#                 # stack.clear()

#     # Add the last block if there's any (ie EOF I think w/out any dedent)
#     if block_active and block:
#         lines = block.splitlines()
#         block = "\n".join([line[previous_indent:] for line in lines])
#         python_code += block.strip() + "\n"

#     return python_code.strip()

