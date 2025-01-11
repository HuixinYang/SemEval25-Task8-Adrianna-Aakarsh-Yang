from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from functools import partial
import ast
import os
import numpy as np
import pandas as pd

import test_case_prompt_builder
import test_case_runner
import test_case_query_model
# Assume openai>=1.0.0
from openai import OpenAI

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

            test_prompt = test_case_prompt_builder.build_prompt(question_df[idx], 
                                                                backing_dataset_df, 
                                                                skip_description=filtered_datasets)

            # Get API completion
            completion = test_case_query_model.get_api_prompt_completion(test_prompt, model=model, max_tokens=4*1024)

            # Parse the code into an AST
            parsed_code = ast.parse(test_case_query_model.extract_code_from_response(completion))
            imports, response_function_map = test_case_query_model.extract_functions_and_imports(parsed_code)

            # Run the test
            found = test_case_runner.test_run_code(imports, 
                                                   response_function_map, 
                                                   random_seed=42)

            if found:
                print("SUCCESS")
                # Save the test case to a file
                code_to_run = test_case_runner.code_from_imports_function_map(imports, response_function_map)
                with open(output_file, "w") as f:
                    f.write(code_to_run)
            else:
                print("FAILED")
        except Exception as e:
            print(f"Error in test_answer: {e}")
            print("FAILED")
    return

# GET BACKING DATAFRAMES
def get_dataframe_by_id(df_id):
    parquet_file = f"hf://datasets/cardiffnlp/databench/data/{df_id}/all.parquet"
    print(f"Loading {parquet_file}")
    df = pd.read_parquet(parquet_file)
    return df

def fetch_all_dataframes(dataset):
  dataset_ids  = set(map(lambda qa: qa['dataset'],  dataset))
  retval = { ds_id: get_dataframe_by_id(ds_id) for ds_id in dataset_ids }
  return retval

# GET COMPETITION QUESTIONS

def run(max_workers=24, split="train", regenerate=False, model="nvidia/Llama-3.1-Nemotron-70B-Instruct"):
    # Parallel execution using ThreadPoolExecutor

    # Question Dataset
    semeval_train_qa = load_dataset("cardiffnlp/databench", name="semeval", split=split)
    # Backing Datasets
    datasets_map = fetch_all_dataframes(semeval_train_qa)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Adjust max_workers based on your system
            executor.map(partial(process_idx, 
                                    model=model, 
                                    regenerate=regenerate, 
                                    question_df=semeval_train_qa, # questions 
                                    backing_dataset_map=datasets_map,  # backing datasets 
                                    # TODO: No output directory specified.
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
        test_prompt = test_case_prompt_builder.build_prompt(question_df[idx], backing_dataset_df, skip_description=filtered_datasets)
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

run(max_workers=15, split="competition", regenerate=False, model="Qwen/Qwen2.5-Coder-32B-Instruct")

# create_all_test_prompts(split="train", regenerate=True)
