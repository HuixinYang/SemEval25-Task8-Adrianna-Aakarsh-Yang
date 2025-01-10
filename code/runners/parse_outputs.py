import argparse
import os
import re
import logging
import json
import pickle

from datasets import Dataset
from datasets import Dataset, DatasetDict, Features, Value
from datasets import load_dataset

logging.basicConfig(level=logging.DEBUG)

def parse_return_statement(code_string):
    """
    Parse the return statement of the 'answer' method from a given code string.
    
    Args:
        code_string (str): The string containing the Python code.

    Returns:
        str: The content of the return statement or None if not found.
    """
    # Updated regex pattern to match only the first line of the return statement
    pattern = r"def answer\(.*?\):.*?return ([^\n]*)"
    match = re.search(pattern, code_string, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return None
    
def read_pickle_output(result_file, idx=None, split='dev'):
    output = None
    with open(result_file, 'rb') as f:
        output =  pickle.load(f)
   
    codes = output['texts']
    rewards = output['rewards'] 

    solution_map = []
    for code, reward in zip(codes, rewards):
        return_completion = parse_return_statement(code)
        logging.debug(f"code: [{return_completion}], reward: {reward}") 
        solution_map.append({ 
                             'code': return_completion, 
                             'id': idx,
                             'reward': reward, 
                             'split': split 
        }) 
    return solution_map

def find_all_output_files(directory):
    """
    Find all the output files in the given directory.
    
    Args:
        directory (str): The path to the directory containing the output files.
        
    Returns:
        list: A list of paths to the output files.
    """
    output_files = []
    logging.debug(f"Checking directory: {directory}")
    for root, dirs, files in os.walk(directory):
        for file in files:
            logging.debug(f"Checking file: {file}")
            if file.endswith(".pkl") :
                logging.debug(f"Found file: {file}")
                output_files.append(os.path.join(root, file))
    return output_files

def parse_problem_index(file_path):
    """
    Parse the problem index from the given file path.
    
    Args:
        file_path (str): The path to the file.
        
    Returns:
        int: The problem index.
    """
    pattern = r"output_list-(\d+)-\d+"
    match = re.search(pattern, file_path)
    if match:
        return int(match.group(1))
    else:
        return None
    
def collect_all_results(directory, split='dev'):
    """
    Collect all the results from the output files in the given directory.
    
    Args:
        directory (str): The path to the directory containing the output files.
        
    Returns:
        dict: A dictionary containing the results.
    """
    results = {}
    output_files = find_all_output_files(directory)
    for file_path in output_files:
        problem_index = parse_problem_index(file_path)
        results[problem_index] = read_pickle_output(file_path, problem_index, split)
    return results

def write_to_json(results, output_file):
    """
    Write the results to a JSON file.
    
    Args:
        results (dict): The results to write.
        output_file (str): The path to the output file.
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def make_dataset(results, output_file=None, save_to_disk=False, prompt_directory=None):
    """
    Make a Hugging Face dataset from the results and write it to a file.
    
    Args:
        results (dict): The results to write.
        output_file (str): The path to the output file.
    """
    features = Features({
        "id": Value("string"),
        "split": Value("string"),
        "question": Value("string"),
        "dataset": Value("string"),
        "prompt": Value("string"),
        "completion": Value("string"),
        "reward": Value("float")
    })
    
    def fine_tune_generator(results, split, prompt_db=None):
        for problem_index, solutions in results.items():
            for solution in solutions:
                yield {
                    "id": problem_index,
                    "split": split,
                    "question": None,         # TODO: Lookup problem in split dataset by problem-id.
                    "dataset": None,          # TODO: Lookup dataset by problem-id.
                    "prompt": prompt_db[problem_index] if not prompt_db else None,           # TODO: Lookup prompt by problem-id.
                    "code": solution['code'],
                    "reward": solution['reward']
                }

    prompt_map = {
        'dev': load_dataset("aakarsh-nair/semeval-2025-task-8-prompts", split="dev"),
        'train': load_dataset("aakarsh-nair/semeval-2025-task-8-prompts", split="train")
    }
    
    # Create a DatasetDict with streaming datasets
    dataset_dict = DatasetDict({
        split: Dataset.from_generator(lambda s=split: fine_tune_generator(results,s, prompt_map[s]), features=features)
        for split in ['dev']
    })
    
    dataset = None 
    
    return dataset
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="The directory containing the output files.")
    parser.add_argument("output", help="The path to the output file.")
    parser.add_argument("split",default='dev', help="The split of the dataset.")

    args = parser.parse_args()
    
    logging.debug(f"Directory: {args.directory}")
    results = collect_all_results(args.directory, args.split)
    write_to_json(results, args.output)

if __name__ == "__main__":
    main()