def vanilla_template(description):
    verbalized = f"Generate a program completion for the given function description: {description}"
    return verbalized

def extract_functions(text):
    lines = text.split('\n')
    functions = []
    current_function = None
    function_indent = None

    for line in lines:
        stripped_line = line.lstrip()
        indent = len(line) - len(stripped_line)

        if stripped_line.startswith('def '):
            if current_function is not None:
                functions.append('\n'.join(current_function))
            current_function = [line]
            function_indent = indent
        elif current_function is not None:
            if indent > function_indent:
                current_function.append(line)
            else:
                functions.append('\n'.join(current_function))
                current_function = None
                function_indent = None

    if current_function is not None:
        functions.append('\n'.join(current_function))

    return functions