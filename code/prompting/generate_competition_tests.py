#!/usr/bin/env python
# coding: utf-8

from concurrent.futures import ThreadPoolExecutor
from functools import partial

import ast
import os
import numpy as np
import pandas as pd
import argparse
import logging

import test_case_prompt_builder
import test_case_runner
import test_case_query_model
import test_case_load_dataset

from datasets import Dataset, DatasetDict
from datasets import Dataset, DatasetDict, Features, Value
from datasets import load_dataset
from datasets import Dataset, concatenate_datasets

logging.basicConfig(level=logging.INFO)

def process_idx(idx, question_df=None,
                                backing_dataset_map=None,
                                model="nvidia/Llama-3.1-Nemotron-70B-Instruct",
                                cache_dir="~/.cache",
                                phase="competition",
                                use_cache=True,
                                split=None,
                                regenerate=False,
                                filtered_datasets=['029_NYTimes']):
    """
    Process a single index to generate test cases.
    """
    logging.info(f"Processing index: {idx} phase: {phase} split: {split}")
    max_attempts = 10
    found = False
    
    cache_dir = os.path.expanduser(os.path.join(cache_dir, "test-case-generation", f"{model}"))
    if not os.path.exists(cache_dir) and use_cache:
        os.makedirs(cache_dir)

    current_timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    semeval_id = question_df[idx]['semeval_id']
    output_file = f"{cache_dir}/split-{split}-phase-{phase}-test_case_{semeval_id}.parquet"
    logging.debug(f"CACHE_DIR: {cache_dir} OUTPUT_FILE: {output_file}")

    skip_condition = (os.path.exists(output_file) and not regenerate and use_cache)  \
                    or question_df[idx]['dataset'] in set(filtered_datasets)

    logging.info(f"Output file: {output_file}, skip_condition: {skip_condition} regenerate: {regenerate} use_cache: {use_cache} filtered_datasets: {filtered_datasets}")   
    # skip if the test file already exists
    if skip_condition:
        logging.info(f"SKIPPING: {idx}")
        return output_file

    while max_attempts > 0 and not found:
        max_attempts -= 1
        try:
            # generate test prompt
            dataset_id = question_df[idx]['dataset']
            backing_dataset_df = backing_dataset_map[dataset_id]

            test_prompt = test_case_prompt_builder.build_prompt(question_df[idx], 
                                                                backing_dataset_df, 
                                                                skip_description=filtered_datasets)

            # Get API completion
            completion = test_case_query_model.get_api_prompt_completion(test_prompt, model=model, max_tokens=4*1024)

            # Parse the code into an AST
            parsed_code = \
                ast.parse(test_case_query_model.extract_code_from_response(completion))
                
            imports, response_function_map = \
                test_case_query_model.extract_functions_and_imports(parsed_code)


            # Type check answer
            predicted_type = question_df[idx]['predicted_type']
            if predicted_type:
                found = test_case_runner.assert_answer_predicted_type(predicted_type, 
                                    imports, response_function_map, random_seed=42)
                if not found:
                    logging.error(f"Predicted type: {predicted_type} does not match the answer")
                    break 

            # Run the test
            found = test_case_runner.test_run_code(imports, response_function_map, random_seed=42)

            if found:
                logging.info("SUCCESS")
                # Save the test case to a file
                code_to_run = test_case_runner.code_from_imports_function_map(imports, response_function_map)
                df = pd.DataFrame([{
                                    'semeval_id': question_df[idx]['semeval_id'],
                                    'split': split,
                                    'phase': phase,
                                    'question': question_df[idx]['question'],
                                    'dataset': question_df[idx]['dataset'],
                                    'predicted_type': question_df[idx]['predicted_type'] if 'predicted_type' in question_df[idx] else None,
                                    'model': model,
                                    'content': code_to_run,
                                    'update_timestamp': current_timestamp
                                }])
                logging.info(f"Saving to file: {output_file}, df: {df.to_json()}")
                # save the test case to a file
                df.to_parquet(output_file)
                return output_file
            else:
                logging.error("FAILED", exc_info=True)
        except Exception as e:
            print(f"Error in test_answer: {e}")
            logging.exception("Exception occurred while processing index: %s", idx)
            print("FAILED")
    return None

def run(max_workers=24, 
            test_case_dataset=None,
            question_dataset=None,
            backing_dataset_map=None,
            phase=None, 
            split=None, 
            regenerate=False, 
            model="nvidia/Llama-3.1-Nemotron-70B-Instruct",
            cache_dir="~/.cache",
            use_cache=True):
   
    logging.info(f"Running test cases for phase: {phase}, split: {split}") 
    
    if isinstance(test_case_dataset, DatasetDict):
        test_case_dataset = test_case_dataset[split]
        
    # Parallel execution using ThreadPoolExecutor
    # Adjust max_workers based on your system 
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:  
            results = executor.map(partial(process_idx, model=model, 
                                    cache_dir=cache_dir,
                                    use_cache=use_cache,
                                    regenerate=regenerate, 
                                    phase=phase,
                                    question_df=question_dataset, # questions 
                                    backing_dataset_map=backing_dataset_map,  # backing datasets 
                                    split=split # TODO: No output directory specified.
                                    ), range(len(question_dataset)))
    
    # Update the dataset with test cases
    updated_result_files = list(filter(lambda x: x is not None, results))
    logging.debug(f"Updated files: {updated_result_files} number of questions to update: {len(question_dataset)}")
    
    if not updated_result_files:
        # No test cases generated, return the original dataset
        return test_case_dataset 
    
    try:
        updated_datasets = [Dataset.from_parquet(file) for file in updated_result_files]
        updated_rows = concatenate_datasets(updated_datasets)
        logging.debug(f"Updated rows: {updated_rows}")
    except Exception as e:
        logging.error(f"Error reading datasets: {e}")
        return test_case_dataset
    
    
    # Update or create rows in the test case dataset
    def update_or_create(row):
        matching_rows = updated_rows.filter(lambda x: x['semeval_id'] == row['semeval_id'] and x['model'] == row['model'])
        if len(matching_rows) > 0:
            return matching_rows[0]
        else:
            return row
    # Apply the update_or_create function to each row in the test_case_dataset
    # Ensure correct split handling during update
    if isinstance(test_case_dataset, DatasetDict):
        test_case_dataset[split] = test_case_dataset[split].map(update_or_create)
    else:
        test_case_dataset = test_case_dataset.map(update_or_create)

    
    # Add new rows for semeval_ids that are in updated_rows but not in test_case_dataset
    existing_ids = set(test_case_dataset['semeval_id'])
    new_rows = updated_rows.filter(lambda x: x['semeval_id'] not in existing_ids)
    test_case_dataset = concatenate_datasets([test_case_dataset, new_rows])   
    logging.debug(f"Updated dataset return: {test_case_dataset}")

    retval = test_case_dataset if not isinstance(test_case_dataset, DatasetDict) else DatasetDict({split: test_case_dataset[split]})

    return retval 

def select_based_on_predicted_type(dataset,  predicted_type):
    """
    Select the dataset based on the predicted type
    """
    return dataset.filter(lambda x: x['predicted_type'] == predicted_type)

def main():
    parser = argparse.ArgumentParser(description="Generate test cases with parallel processing.")
    parser.add_argument("--phase", type=str, default="competition", help="Phase of the competition.")
    parser.add_argument("--split", type=str, default="dev", help="Dataset split (train/dev/test).")
    parser.add_argument("--model", type=str, default="nvidia/Llama-3.1-Nemotron-70B-Instruct", help="Model to use.")
    parser.add_argument("--cache-dir", type=str, default="~/.cache", help="Cache directory path.")
    parser.add_argument("--max-workers", type=int, default=os.cpu_count(), help="Number of workers for parallel processing.")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate test cases even if cached.")
    parser.add_argument("--use-cache", action="store_true", help="Use cached test cases if available.")

    args = parser.parse_args()

    cache_dir = os.path.expanduser(args.cache_dir)
    repository_name = "semeval-2025-task-8-test-cases-competition"
    user_repo = "aakarsh-nair"
    # Load datasets
    question_dataset, backing_dataset_map = test_case_load_dataset.load_phase_dataset(phase=args.phase, split=args.split)
    test_case_dataset = load_dataset(f"{user_repo}/{repository_name}", split=args.split)

    # Run the test case generation
    test_case_dataset = run(
        max_workers=args.max_workers, 
        test_case_dataset=test_case_dataset, 
        question_dataset=question_dataset, 
        backing_dataset_map=backing_dataset_map, 
        phase=args.phase, 
        split=args.split, 
        regenerate=args.regenerate, 
        model=args.model, 
        cache_dir=args.cache_dir, 
        use_cache=args.use_cache
    )

    # Save results
    test_case_dataset.save_to_disk((f"{cache_dir}/{repository_name}"))
    test_case_dataset.push_to_hub(f"{user_repo}/{repository_name}", split=args.split)

if __name__ == "__main__":
    main()