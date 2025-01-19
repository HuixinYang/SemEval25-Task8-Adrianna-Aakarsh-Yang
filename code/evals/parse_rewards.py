import json
import argparse 
import logging
import os
import pandas as pd

from multiprocessing import Pool, Manager, cpu_count
from utils.dir_utils import (find_all_output_files, 
                             collect_all_results,
                             extract_id_from_path)

from dataloading.semeval_load_dataset import load_phase_dataset
from runners.push_results_to_finetune import parse_return_statement 
from py_evaluator.utils import post_process, evaluate_return_statement

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

def read_reward_files(root_dir, model_name):
    """
    Parse rewards from the output of the reward function.
    """  
    logging.debug(f"parse_rewards: Root directory: {root_dir}") 
    results = collect_all_results(f"{root_dir}/{model_name}", 
                                                ends_with=".parquet", 
                                                pattern=r".*-(\d+).*")
    parsed_rewards = {
        problem_index: pd.read_parquet(file_path) 
            for problem_index, file_path in results.items()
    }
    return parsed_rewards

def parse_reward_data(reward_data):
    """
    Parse the return statement from the text.
    """
    parsed_reward_data = {}
    for i in reward_data.keys():
        reward_completions = reward_data[i].to_dict() 
        reward_values = reward_completions['reward']
        return_statements = [ parse_return_statement(text) 
                                for a, text in reward_completions['text'].items() ]
        for j in range(len(reward_values)):
            if i not in parsed_reward_data:
                parsed_reward_data[i] = []
            parsed_reward_data[i].append({
                'reward': reward_values[j],
                'return': return_statements[j]
            })
    return parsed_reward_data

def reshape_and_enrich_data(parsed_rewards, questions_dataset, datasets_map):
    """
    Reshape the data and enrich it with additional information.
    """
    parsed_rewards = parsed_rewards.copy()

    for semeval_id in parsed_rewards.keys():
        semeval_id = int(semeval_id)
        question = questions_dataset[semeval_id] 
        parsed_rewards[semeval_id] = {
            'question': question['question'],
            'semeval_id': semeval_id,
            'dataset': question['dataset'],
            'parsed_rewards': parsed_rewards[semeval_id]
        }
    return parsed_rewards


def process_semeval_id(semeval_id, parsed_rewards, question_dataset, datasets_map, datasets_map_lite):
    """
    Process a single semeval_id for parallelization.
    """
    parsed_reward = parsed_rewards[semeval_id]
    for idx, rollout in enumerate(parsed_reward['parsed_rewards']):
        return_statement = rollout['return']
        eval_result = evaluate_return_statement(return_statement, datasets_map[question_dataset[semeval_id]['dataset']])
        parsed_rewards[semeval_id]['parsed_rewards'][idx]['eval_result'] = eval_result
        eval_result_lite = evaluate_return_statement(return_statement, datasets_map_lite[question_dataset[semeval_id]['dataset']])
        parsed_rewards[semeval_id]['parsed_rewards'][idx]['eval_result_lite'] = eval_result_lite
        logging.debug(f"Eval result {semeval_id}-{idx}: all:{eval_result} lite:{eval_result_lite}")
    return semeval_id, parsed_rewards[semeval_id]


def enrich_with_executions_results(parsed_rewards, question_dataset, datasets_map, datasets_map_lite):
    """
    Enrich the data with the results of the executions.
    """
    semeval_id_list = sorted(map(int, parsed_rewards.keys()))
    with Manager() as manager:
        shared_parsed_rewards = manager.dict(parsed_rewards)
        with Pool(cpu_count()) as pool:
            results = pool.starmap(
                process_semeval_id, 
                [(semeval_id, shared_parsed_rewards, question_dataset, datasets_map, datasets_map_lite) for semeval_id in semeval_id_list]
            )
        for semeval_id, enriched_data in results:
            parsed_rewards[semeval_id] = enriched_data
    return parsed_rewards

if __name__ == "__main__":
    root_dir = os.path.expanduser("~/.cache/pipeline-runs")
    parser = argparse.ArgumentParser(description="Parse rewards from the output of the reward function.")
    parser.add_argument("--root-dir", type=str, default=root_dir, help="The path to the directory containing the parquet files.")
    parser.add_argument("--base-model", type=str, default="codellama/CodeLlama-7b-Python-hf", help="The name of the repository.")
    parser.add_argument("--phase", type=str, default="competition", help="The phase of the competition.")
    parser.add_argument("--split", type=str, default="dev", help="The dataset split.")
    # parser.add_argument
    args = parser.parse_args()

    questions_dataset, datasets_map = load_phase_dataset(phase=args.phase, split=args.split)
    _, lite_datasets_map = load_phase_dataset(phase=args.phase, split=args.split, lite=True)
    reward_data = read_reward_files(args.root_dir, args.base_model)

    parsed_rewards = parse_reward_data(reward_data)
    parsed_rewards = reshape_and_enrich_data(parsed_rewards, questions_dataset, datasets_map)

    parsed_rewards = enrich_with_executions_results(parsed_rewards, questions_dataset, datasets_map, lite_datasets_map)

    with open("parsed_rewards.json", "w") as f:
        f.write(json.dumps(parsed_rewards, indent=4))


    # print(json.dumps(parsed_rewards, indent=4))
    # Evaluate the return statements

    """
    finetune_dataset = make_dataset(results, prompt_map=load_prompts(), 
                                                semeval_map=load_sem_eval_dataset(), 
                                                dataset_name=args.repo_name, 
                                                save_to_disk=True)
    """ 