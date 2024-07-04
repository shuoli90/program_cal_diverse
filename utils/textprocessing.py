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

def extract_and_remove_multiline_comments(code):
    # Define patterns for """ and '''
    triple_double_quotes_pattern = r'\"\"\"(.*?)\"\"\"'
    triple_single_quotes_pattern = r"\'\'\'(.*?)\'\'\'"

    # Initialize an empty list to store extracted comments
    extracted_comments = []

    # Extract and remove triple-double-quoted comments
    extracted_comments.extend(re.findall(triple_double_quotes_pattern, code, flags=re.DOTALL))
    cleaned_code = re.sub(triple_double_quotes_pattern, '', code, flags=re.DOTALL)

    # Extract and remove triple-single-quoted comments
    extracted_comments.extend(re.findall(triple_single_quotes_pattern, cleaned_code, flags=re.DOTALL))
    cleaned_code = re.sub(triple_single_quotes_pattern, '', cleaned_code, flags=re.DOTALL)

    extracted_comments = '"""\n' + "\n".join(extracted_comments) + '\n"""' if extracted_comments else ""
    
    return extracted_comments, cleaned_code


block_patterns = re.compile(r"""
                                def\s+\w+\s*\(|
                                class\s+\w+|
                                if\s+__name__\s*==\s*\"__main__\":|
                                while\s*.*:\s*$|           # while statements, ends with colon and optional whitespace until end of line    
                                for\s+.*in\s+.*:\s*$|  # for statements
                                # for\s+\w+\s+in\s+.*:\s*$|  # for statements
                                if\s*.*:\s*$|              # if statements
                                elif\s*.*:\s*$|            # elif statements
                                else\s*:\s*$|              # else statements
                                try\s*:\s*$|               # try statements
                                except\s*.*:\s*$|          # except statements
                                finally\s*:\s*$|           # finally statements
                                with\s*.*:\s*$|            # with statements
                                async\s+def\s+\w+\s*\(  # async def statements
                                """, re.VERBOSE)


block_patterns_allow_addtl = re.compile(r"""
                        # while\s+.*:.*|           # while statements, ends with colon and optional whitespace until end of line
                        # for\s+.*in\s+.*:.*|  # for statements
                        if\s+.*:.*|              # if statements
                        elif\s+.*:.*|            # elif statements
                        else\s*:.*|              # else statements
                        # try\s*:.*|               # try statements
                        # except\s+.*:.*|          # except statements
                        # finally\s*:.*|           # finally statements
                        # with\s+.*:.*|            # with statements
                        """, re.VERBOSE)

# def clean_control_structures(code):
#     # Regex that matches 'if', 'elif', 'else' at the start of a line followed by anything up to the first colon
#     pattern = r'^(if|elif|else).*?:'
    
#     # Substitute matched patterns with the keyword followed by a colon
#     cleaned_code = re.sub(pattern, r'\1:', code, flags=re.MULTILINE)
#     return cleaned_code
                                
    
def extract_python_code(text):
    import ast 
    import re
    
    
    # formatted_code, text = extract_formatted_code(text)
    # this is to handle markdown formatted code blocks
    multiline_comments, text = extract_and_remove_multiline_comments(text)
    text = re.sub(r"```Python\n", "```\n", text)
    text = re.sub(r"```", "```\n", text)
    lines = text.splitlines()
    # python_code = ""
    python_code = multiline_comments + "\n"
    block = ""
    stack = []
    block_active = False
    previous_indent = 0

    for line_idx, line in enumerate(lines):
        
        
        stripped_line = re.sub(r'#.*$', '', line)  # Remove comments
        stripped_line = stripped_line.lstrip() 
        leading_spaces = len(line) - len(stripped_line)

        # Check if the line starts a new block or is a continuation of a block
        
        if not block_active and not stripped_line:
            python_code += line + "\n"
            continue
        
        if block_patterns.match(stripped_line):
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
        
        # If currently active in a block and the line is part of it, or empty line
        if block_active and (leading_spaces > previous_indent or not stripped_line):
            block += line + "\n"
            continue
        
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
        

        # Handle import statements outside of block checks (ie. at the beginning of the file), decorators
        if re.match(r"(from\s+[\w\.]+\s+import\s+[\w\.,\s*]+|import\s+[\w\.,\s*]+|@)", stripped_line) and not block_active:
            python_code += stripped_line + "\n"
            continue        

        
        if not block_active and stripped_line and not stripped_line.startswith('@'):
        # if stripped_line and not stripped_line.startswith('@'):
            try: 
                parsed_node = ast.parse(stripped_line).body
                if parsed_node: # and isinstance(parsed_node[0], ast.stmt) and not isinstance(parsed_node[0], ast.Expr): 
                    if not (isinstance(parsed_node[0], ast.Expr) and isinstance(parsed_node[0].value, ast.Name)):
                        python_code += line + "\n"
                        continue
            except SyntaxError:
                pass
            
            # if "else:print(y + (1867,1911,1925,1988)[e - 1])" in line: 
            #     import pdb; pdb.set_trace()
            try: 
                # if 'else:print("no")' in line:
                #     import pdb; pdb.set_trace()
                if block_patterns_allow_addtl.match(stripped_line):
                    block_pattern_sub = re.sub(".*:", "", stripped_line)
                    parsed_node = ast.parse(block_pattern_sub).body
                    if (isinstance(parsed_node[0], ast.Expr) and not isinstance(parsed_node[0].value, ast.Name)):
                        # see that the next line is not indented 
                        if line_idx + 1 >= len(lines):
                            python_code += line + "\n"
                        else: 
                            next_line = lines[line_idx + 1]
                            next_stripped_line = re.sub(r'#.*$', '', next_line).lstrip()
                            next_leading_spaces = len(next_line) - len(next_stripped_line)
                            if next_leading_spaces <= leading_spaces:
                                python_code += line + "\n"
                                continue
                        
            except (SyntaxError, IndexError): 
                pass
                        
            
            
        
            
        

        

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

