import random
import ast
import os
import logging

from datasets import DatasetDict, Dataset

logging.basicConfig(level=logging.DEBUG)

def code_from_imports_function_map(imports, response_function_map, custom_answer=None):
  logging.info("Creating code from imports and function map")
  answer = response_function_map['answer'] if custom_answer is None else custom_answer
  preamble_template="\n".join(imports)
  code_to_run = preamble_template+"\n"+response_function_map['dummy_data']+"\n"+answer+"\n"+response_function_map['test_answer']+"\n"
  return code_to_run

# Create an isolated namespace
def test_run_code(imports, response_function_map, custom_answer=None,random_seed=42):
  logging.info("Running code in isolated namespace")
  local_namespace = {}
  code_to_run= code_from_imports_function_map(imports, response_function_map) \
    if not custom_answer else code_from_imports_function_map(imports, response_function_map, custom_answer=custom_answer)
  # Execute the code in the isolated namespace
  logging.debug(f"Code to run:\n{code_to_run}")
  exec(code_to_run, {}, local_namespace)
  # Update each function's globals to include the local_namespace
  for key, value in local_namespace.items():
      if callable(value):  # Check if the item is a function
          value.__globals__.update(local_namespace)
  # Access and invoke the test_answer function from the isolated namespace
  test_answer = local_namespace["test_answer"]
  try:
    test_answer(random_seed)  # This executes the function in the isolated context
  except Exception as e:
    print(f"Error in test_answer: {e}")
    return False
  return True