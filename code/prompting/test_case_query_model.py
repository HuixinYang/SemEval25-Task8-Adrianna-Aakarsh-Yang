# Assume openai>=1.0.0
from openai import OpenAI
import re
import ast
import os

def get_api_prompt_completion(prompt, model="Qwen/Qwen2.5-Coder-32B-Instruct", max_tokens=1024, api_key=None):
    """
    Get the completion from the API 
    """
    if api_key is not None:
        api_key = os.getenv('DEEPINFRA_TOKEN')
        
    # Create an OpenAI client with your deepinfra token and endpoint
    openai = OpenAI(
        api_key=f"{api_key}",
        base_url="https://api.deepinfra.com/v1/openai",
    )

    chat_completion = openai.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", 
                   "content": prompt}],
    )
    return chat_completion.choices[0].message.content


def extract_code_from_response(response, idx=0):
  """
  """
  code_blocks = re.findall(r"```python(.*?)```", response, re.DOTALL)
  return code_blocks[idx]

def extract_functions_and_imports(code):
    """
    Extract all import statements and function definitions from the code.
    Returns a tuple:
    - List of import statements as strings.
    - Dictionary of function names and their evaluable strings.
    """
    # Parse the code into an AST
    parsed_code = ast.parse(code)

    # List to store import statements
    imports = []

    # Dictionary to store function names and their strings
    functions_map = {}

    for node in parsed_code.body:
        # Check for import statements
        if isinstance(node, ast.Import):
            imports.append(ast.unparse(node))
        elif isinstance(node, ast.ImportFrom):
            imports.append(ast.unparse(node))
        # Check for function definitions
        elif isinstance(node, ast.FunctionDef):
            function_name = node.name
            function_source = ast.unparse(node)
            functions_map[function_name] = function_source

    return imports, functions_map