import pandas as pd
import subprocess
import shlex
import zipfile
import torch
from datasets import load_dataset
import os
import numpy as np
import os
import json
import os
import ast
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import ast
import re
import json
import pandas as pd

# Assume openai>=1.0.0
from openai import OpenAI

DEEPINFRA_TOKEN=os.getenv('DEEPINFRA_TOKEN')

def get_dataframe_by_id(df_id):
    parquet_file = f"hf://datasets/cardiffnlp/databench/data/{df_id}/all.parquet"
    print(f"Loading {parquet_file}")
    df = pd.read_parquet(parquet_file)
    return df

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

def fetch_all_dataframes(dataset):
  dataset_ids  = set(map(lambda qa: qa['dataset'],  dataset))
  retval = { ds_id: get_dataframe_by_id(ds_id) for ds_id in dataset_ids }
  return retval

 
# Create an OpenAI client with your deepinfra token and endpoint
openai = OpenAI(
    api_key=f"{DEEPINFRA_TOKEN}",
    base_url="https://api.deepinfra.com/v1/openai",
)

def get_api_prompt_completion(prompt, model="Qwen/Qwen2.5-Coder-32B-Instruct", max_tokens=1024):
    chat_completion = openai.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return chat_completion.choices[0].message.content


def code_from_imports_function_map(imports, response_function_map, custom_answer=None):
  answer = response_function_map['answer'] if custom_answer is None else custom_answer
  preamble_template="\n".join(imports)
  code_to_run = preamble_template+"\n"+response_function_map['dummy_data']+"\n"+answer+"\n"+response_function_map['test_answer']+"\n"
  return code_to_run


# Create an isolated namespace
def test_run_code(imports, response_function_map, custom_answer=None,random_seed=42):
  local_namespace = {}
  code_to_run= code_from_imports_function_map(imports, response_function_map) \
    if not custom_answer else code_from_imports_function_map(imports, response_function_map, custom_answer=custom_answer)
  # Execute the code in the isolated namespace
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

def generate_dataframe_schma_json(df):
  schema = {
       "columns": [
           {"name": col, "type": str(df[col].dtype)}
           for col in df.columns
       ]
   }
  json_schema = json.dumps(schema, indent=4)
  return json_schema


def generate_dataframe_description_json(df):
  description = df.describe().to_json(orient='index', indent=4)
  return description


def generate_random_sample_of_n_rows_json(df, n=10):
    return df.sample(n=n).to_json(orient='records', indent=4)


def test_prompt_generator(row, df):
    question = row['question']
    df_random_sample = '{}'
    if not row['dataset'] == "029_NYTimes":
       df_random_sample = generate_dataframe_description_json(df) 
    prompt =f"""
# OUTPUT: ONLY PYTHON CODE, Replace all TODO with actual code. You can use pandas and numpy.
import pandas as pd
import numpy as np

# Description of dataframe schema.
df_schema = {generate_dataframe_schma_json(df)}

# Description of dataframe columns.
df_descrption = {generate_dataframe_description_json(df)}

# Random sample of 10 rows from the dataframe.
df_random_sample = {df_random_sample}



# It should give the answer to: {question}
# The answer should only contain python code, you are not allowed leave any TODO undone.
def answer(df: pd.DataFrame):
    # Use df to answer :{question}
    df.columns = {list(df.columns)}
    pass # stub function leave as is.

# Create a dummy data in the same format as actual data. so we can
# answer unit test answer to: {question}.
def dummy_data(random_seed) -> pd.DataFrame:
    # pd.DataFrame has columns , {list(df.columns)}
    df = pd.DataFrame(TODO: GENERATE DUMMY DATA HERE, Make it random using random seed)
    return df

# Complete the following function which tests "answer" function using
# a dummy data pd.Dataframe  and tests it with assert statement.
def test_answer(random_seed):
    df.columns = {list(df.columns)}
    dummy_data_df = dummy_data(random_seed)
    # We need to chack that result correctly answers the question: {question}
    result = answer(dummy_data_df)
    # TODO: Don't check datatypes only semantics.
    # assert that "answer" function is correct using dummy data.
    assert result #TODO: Make a unqique semantic test of functionality of "answer" function.
    assert ... #TODO: complete this assert using information from dummy_data to test semantics.
    assert ... #TODO: complete this assert using information from dummy_data to test semantics.
"""
    return prompt


def process_idx(idx, question_df=None,
                                backing_dataset_map=None,
                                model="nvidia/Llama-3.1-Nemotron-70B-Instruct",
                                regenerate=False,
                                filtered_datasets=['029_NYTimes'],
                                split="train"):
    """
    Process a single index to generate test cases.
    """
    print("-" * 20, idx, "-" * 20)
    max_attempts = 10
    found = False

    # Skip if the test file already exists
    output_file = f"/content/drive/MyDrive/TUE-WINTER-2024/CHALLENGES-CL/test_cases/{split}/{model}/test_case_{idx}.py"
    if (os.path.exists(output_file) and not regenerate) or question_df[idx]['dataset'] in set(filtered_datasets):
        print(f"SKIPPING: {idx}")
        return

    while max_attempts > 0 and not found:
        max_attempts -= 1
        try:
            # Generate test prompt
            dataset_id = question_df[idx]['dataset']
            backing_dataset_df = backing_dataset_map[dataset_id]
            test_prompt = test_prompt_generator(question_df[idx], backing_dataset_df)

            # Get API completion
            completion = get_api_prompt_completion(test_prompt, model=model, max_tokens=4*1024)

            # Parse the code into an AST
            parsed_code = ast.parse(extract_code_from_response(completion))
            imports, response_function_map = extract_functions_and_imports(parsed_code)

            # Run the test
            found = test_run_code(imports, response_function_map, random_seed=42)

            if found:
                print("SUCCESS")
                # Save the test case to a file
                code_to_run = code_from_imports_function_map(imports, response_function_map)
                with open(output_file, "w") as f:
                    f.write(code_to_run)
            else:
                print("FAILED")
        except Exception as e:
            print(f"Error in test_answer: {e}")
            print("FAILED")
    return

def run(max_workers=24, split="train", regenerate=False, model="nvidia/Llama-3.1-Nemotron-70B-Instruct"):
    # Parallel execution using ThreadPoolExecutor
    semeval_train_qa = load_dataset("cardiffnlp/databench", name="semeval", split=split)

    datasets_map = fetch_all_dataframes(semeval_train_qa)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Adjust max_workers based on your system
            executor.map(partial(process_idx, 
                                    model=model, 
                                    regenerate=regenerate, 
                                    question_df=semeval_train_qa, 
                                    backing_dataset_map=datasets_map, 
                                    split=split), 
                         range(len(semeval_train_qa)))


def create_test_prompt_file(idx, question_df=None,
                                    backing_dataset_map=None,
                                    split="train",
                                    regenerate=False,
                                    filtered_datasets=['029_NYTimes']):
    """
    Process a single index to generate test cases.
    """
    print("-" * 20, idx, "-" * 20)
    found = False
    output_file = f"/content/drive/MyDrive/TUE-WINTER-2024/CHALLENGES-CL/test_cases/prompts/{split}/test_case_gen_prompt_{idx}.py"
    if (os.path.exists(output_file) and not regenerate) or question_df[idx]['dataset'] in set(filtered_datasets):
        print(f"SKIPPING: {idx}")
        return

    try:
        # Generate test prompt
        dataset_id = question_df[idx]['dataset']
        backing_dataset_df = backing_dataset_map[dataset_id]
        test_prompt = test_prompt_generator(question_df[idx], backing_dataset_df)
        with open(output_file, "w") as f:
            f.write(test_prompt)
    except Exception as e:
        print(f"Error in test_answer: {e}")
        print("FAILED")

def create_all_test_prompts(split="train", regenerate=False):
    semeval_train_qa = load_dataset("cardiffnlp/databench", name="semeval", split=split)
    datasets_map = fetch_all_dataframes(semeval_train_qa)
    for idx in range(len(semeval_train_qa)):
        create_test_prompt_file(idx, question_df=semeval_train_qa, 
                                        backing_dataset_map = datasets_map, 
                                        regenerate=regenerate,
                                        split=split)

run(max_workers=15, split="train", regenerate=False, model="Qwen/Qwen2.5-Coder-32B-Instruct")

# create_all_test_prompts(split="train", regenerate=True)
