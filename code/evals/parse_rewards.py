import argparse 
import json
import logging
import os
import pandas as pd

from utils.dir_utils import find_all_output_files, extract_id_from_path, collect_all_results
from runners.push_results_to_finetune import parse_return_statement 
 
logging.basicConfig(level=logging.DEBUG)

def find_parquet_files(root_dir, model_name):
    """
    Find all the parquet files in the given directory.
    
    Args:
        root_dir (str): The path to the directory containing the parquet files.
        model_name (str): The name of the model.
        
    Returns:
        list: A list of paths to the parquet files.
    """
    output_files =  find_all_output_files(f"{root_dir}/{model_name}", ends_with=".parquet")
    logging.debug(f"Found {len(output_files)} parquet files.")
    return output_files

def parse_rewards(root_dir, model_name):
    """
    Parse rewards from the output of the 
    reward function.
    """  
    logging.debug(f"parse_rewards: Root directory: {root_dir}") 
    results = collect_all_results(f"{root_dir}/{model_name}", ends_with=".parquet", pattern=r".*-(\d+).*")
    parsed_rewards = { problem_index: pd.read_parquet(file_path) for problem_index, file_path in results.items() }
    return parsed_rewards

 
   
if __name__ == "__main__":
    root_dir = os.path.expanduser("~/.cache/pipeline-runs")
    parser = argparse.ArgumentParser(description="Parse rewards from the output of the reward function.")
    parser.add_argument("--root-dir", type=str, default=root_dir, help="The path to the directory containing the parquet files.")
    parser.add_argument("--base-model", type=str, default="codellama/CodeLlama-7b-Python-hf", help="The name of the repository.")
    
    # parser.add_argument
    args = parser.parse_args()
    rewards  = parse_rewards(args.root_dir, args.base_model)
    for i in range(max(rewards.keys())+1):
        if i in rewards:
            reward_completions = rewards[i].to_dict() 
            reward_values =reward_completions['reward']
            return_statements = [parse_return_statement(text) for a, text in reward_completions['text'].items()]
            #return_statements = map(parse_return_statement, reward_completions['text'])
            for j in range(len(reward_values)):
                print(f"{i:2d}-{j:2d} {reward_values[j]} {return_statements[j]}")
            #print(f"{i:2d}. {reward_completions[j]}")
            #print(f"{i:2d}. {parse_return_statement(rewards[i]['text'])}")
            #print(f"{i:2d}. {parse_return_statement(rewards[i]['text'][i])}")
    """
    finetune_dataset = make_dataset(results, prompt_map=load_prompts(), 
                                                semeval_map=load_sem_eval_dataset(), 
                                                dataset_name=args.repo_name, 
                                                save_to_disk=True) 
    """ 
    
