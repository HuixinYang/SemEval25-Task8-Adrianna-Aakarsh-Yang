import argparse
import os
import re
import logging
import json
import pickle
from datetime import datetime

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

def make_dataset(results, 
                    prompt_map=None, 
                    semeval_map=None , 
                    dataset_name=None, 
                    save_to_disk=False):
    """
    Make a Hugging Face dataset from the results and write it to a file.
    
    Args:
        results (dict): The results to write.
        output_file (str): The path to the output file.
    """
    features = Features({
        "semeval_id": Value("string"),
        "split": Value("string"),
        "question": Value("string"),
        "dataset": Value("string"),
        "prompt": Value("string"),
        "completion": Value("string"),
        "reward": Value("float"),
        "answer": Value("string"), 
        "computed_answer": Value("string"),
        "computed_sample_answer" : Value("string"),
        "is_correct": Value("bool"),
        "is_correct_sample": Value("bool"),
        "type": Value("string"), 
        "sample_answer":Value("string"),
        "create_timestamp": Value("string"),
        "update_timestamp": Value("string")

    })
    
    def fine_tune_generator(results, split, prompt_db=None, semeval_db=None):
        for problem_index, solutions in results.items():
            for solution in solutions:
                yield {
                    "semeval_id": problem_index,
                    "split": split,
                    "question": semeval_db[problem_index]['question'] if semeval_db else None, # TODO: Lookup problem in split dataset by problem-id.
                    "answer": semeval_db[problem_index]['answer'] if semeval_db else None, # TODO: Lookup problem in split dataset by problem-id.
                    "sample_answer": semeval_db[problem_index]['sample_answer'] if semeval_db else None, # TODO: Lookup problem in split dataset by problem-id.
                    "type": semeval_db[problem_index]['type'] if semeval_db else None, # TODO: Lookup problem in split dataset by problem-id.
                    "dataset": semeval_db[problem_index]['dataset'] if semeval_db else None,   # TODO: Lookup dataset by problem-id.
                    "prompt": prompt_db[problem_index]['content'] if  prompt_db else None,  # TODO: Lookup prompt by problem-id.
                    "completion": solution['code'],
                    "create_timestamp": datetime.now().isoformat(),
                    "update_timestamp": None,
                    "computed_answer": None,
                    "is_correct": None,
                    "is_correct_sample": None,
                    "computed_sample_answer": None,
                    "correct_sample": None,
                    "reward": solution['reward']
                }
    
    # Create a DatasetDict with streaming datasets
    dataset  = DatasetDict({
        split: Dataset.from_generator(lambda s=split: 
            fine_tune_generator(results,s, prompt_db=prompt_map[s], semeval_db=semeval_map[s]), features=features)
        for split in ['dev']
    })
   
    if save_to_disk:
        dataset.save_to_disk(dataset_name)
        logging.debug(f"Dataset successfully saved to {dataset_name}")
        
    return dataset
    
def load_prompts():
    prompt_map = {
        'dev': load_dataset("aakarsh-nair/semeval-2025-task-8-prompts", split="dev"),
        'train': load_dataset("aakarsh-nair/semeval-2025-task-8-prompts", split="train")
    }
    return prompt_map 

def load_sem_eval_dataset():
    # What about QA split ?
    return_map = {
        'dev': load_dataset("cardiffnlp/databench", name="semeval", split="dev"),
        'train': load_dataset("cardiffnlp/databench", name="semeval", split="train")
    }
    return return_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="The directory containing the output files.")
    parser.add_argument("--split",default='dev', help="The split of the dataset.")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the dataset to Hugging Face Hub.")
    parser.add_argument("--repo-name", type=str, help="Hugging Face Hub repository name.")

    args = parser.parse_args()
    
    logging.debug(f"Directory: {args.directory}")
    results = collect_all_results(args.directory, args.split)
   
    finetune_dataset = make_dataset(results, prompt_map=load_prompts(), 
                          semeval_map=load_sem_eval_dataset(), 
                          dataset_name=args.repo_name, 
                          save_to_disk=True) 

    # write_to_json(results, args.output)
    if args.push_to_hub:
        finetune_dataset.push_to_hub(args.repo_name)

if __name__ == "__main__":
    main()