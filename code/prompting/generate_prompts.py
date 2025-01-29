import os
import pandas as pd
from datasets import (Dataset, 
                      DatasetDict, 
                      Features, 
                      Value)

from test_case_prompt_builder import (build_inference_prompt)
from dataloading.semeval_load_dataset import (load_phase_dataset)

import logging
import argparse

logging.basicConfig(level=logging.INFO)

# Hugging Face Schema Definition - prompts
features = Features({
    "semeval_id": Value("int32"),
    "question": Value("string"),
    "dataset": Value("string"),
    "split": Value("string"),
    "predicted_type": Value("string"),
    "predicted_columns": Value("string"),
    "phase": Value("string"),
    "content": Value("string"),
    "update_timestamp": Value("string") 
})
  
def prompt_generator(question_dataset, dataset_map, split='dev', phase='competition', skip_datasets=None, prompt_style=None):
  """
  Generator function to yield prompts for each question in the dataset.

  Args:
    question_dataset (list): A list of dictionaries where each dictionary represents a question.
    dataset_map (dict): A dictionary mapping dataset names to their respective configurations.
    split (str, optional): The dataset split to use (e.g., 'dev', 'train', 'test'). Defaults to 'dev'.
    phase (str, optional): The phase of the competition (e.g., 'competition', 'evaluation'). Defaults to 'competition'.

  Yields:
    dict: A dictionary containing the following keys:
      - 'semeval_id' (str or None): The SemEval ID of the question.
      - 'question' (str or None): The question text.
      - 'dataset' (str): The name of the dataset.
      - 'split' (str): The dataset split.
      - 'predicted_type' (str or None): The predicted type of the question.
      - 'predicted_columns' (list or None): The predicted columns for the question.
      - 'phase' (str): The phase of the competition.
      - 'content' (str): The generated prompt content.
      - 'update_timestamp' (str): The timestamp when the prompt was generated, formatted as "YYYYMMDDHHMMSS".
  """
  for row_idx in range(len(question_dataset)):
    row = question_dataset[row_idx]
    dataset = row['dataset']
    if skip_datasets and dataset in skip_datasets:
      logging.info(f"Skipping dataset: row{row_idx}:- {dataset}")
      continue
    prompt =  build_inference_prompt(row, dataset_map[dataset])
    current_timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")

    yield {
        'semeval_id': row.get('semeval_id'),
        'question': row.get('question'),
        'dataset': dataset,
        'split': split,
        'predicted_type': row.get('predicted_type'), 
        'predicted_columns': row.get('predicted_columns'),
        'phase': phase,
        'content': prompt,
        'update_timestamp': current_timestamp 
    }

def generate_all_prompts(phase, split, prompt_dataset=None, skip_datasets=None, prompt_style=None):
  """
  Generate prompts for all questions in the dataset.

  Args:
    phase (str): The phase of the dataset to load (e.g., "competition"). Default is "competition".
    split (str): The dataset split to use (e.g., "dev"). Default is "dev".
    prompt_dataset (DatasetDict, optional): An optional dataset to use for generating prompts. Default is None.

  Returns:
    DatasetDict: A dictionary containing the generated prompts for the specified split.

  Raises:
    ValueError: If the provided phase or split is invalid.
  """
  logging.info(f"Generating prompts for phase: {phase} and split: {split}")
  if isinstance(prompt_dataset, DatasetDict):
    prompt_dataset = prompt_dataset[split]
  questions_dataset, dataset_map = load_phase_dataset(phase=phase, split=split)
  logging.info(f"Finished loading dataset: {len(questions_dataset)} questions")
  prompt_dataset  = DatasetDict({
    split:  Dataset.from_generator(
      lambda s=split: prompt_generator(questions_dataset, dataset_map, split=s, phase=phase, skip_datasets=skip_datasets, prompt_style=prompt_style),
      features=features)
    for split in [split]
  })
  logging.debug(f"Prompt Dataset: {prompt_dataset}")
  prompt_dataset = prompt_dataset if not isinstance(prompt_dataset, DatasetDict) else DatasetDict({split: prompt_dataset[split]})
  return prompt_dataset

def parse_args():
  """
  Parses command-line arguments for generating competition prompts.

  Returns:
    argparse.Namespace: Parsed command-line arguments.

  Arguments:
    --phase (str): Phase of the competition. Default is "competition". Required.
    --split (str): Dataset split. Default is "dev". Required.
    --user (str): User name for Hugging Face Hub. Default is "aakarsh-nair".
    --repo-name (str): Repository name for Hugging Face Hub. Default is "semeval-2025-task-8-prompts-competition".
    --cache-dir (str): Cache directory path. Default is "~/.cache/".
    --push-to-hub (bool): Flag to push dataset to Hugging Face Hub. Default is False.
  """
  parser = argparse.ArgumentParser(description="Generate competition prompts")
  parser.add_argument("--phase", type=str, default="competition", required=True, help="Phase of the competition")
  parser.add_argument("--split", type=str, default="dev", required=True, help="Dataset split")
  parser.add_argument("--user", default="aakarsh-nair", type=str, help="User name for Hugging Face Hub")
  parser.add_argument("--repo-name", default="semeval-2025-task-8-prompts-competition", type=str,  help="Repository name for Hugging Face Hub")
  parser.add_argument("--cache-dir", type=str, default="~/.cache", help="Cache directory path")
  parser.add_argument("--push-to-hub", default=False, action="store_true", help="Flag to push dataset to Hugging Face Hub")
  # We skip the NYTimes dataset as including it hangs prompt generation. 
  parser.add_argument("--skip-datasets", type=str,  default=["029_NYTimes"], nargs="+", help="List of datasets to skip")
  parser.add_argument("--prompt-style", type=str, default=None, help="Prompt style to use")
  return parser.parse_args()

def main():
  args  = parse_args()

  logging.info(f"Generating prompts for phase: {args.phase} and split: {args.split}")
  try:
    prompt_dataset = generate_all_prompts(phase=args.phase, split=args.split, skip_datasets=args.skip_datasets, prompt_style=args.prompt_style)
    cache_dir = os.path.expanduser(args.cache_dir)
    save_dir = f"{cache_dir}/{args.phase}/{args.split}/{args.repo_name}"
    if not os.path.exists(save_dir):
      os.makedirs(save_dir, exist_ok=True) 
    logging.info(f"Saving prompts to disk: {save_dir}")
    prompt_dataset.save_to_disk(save_dir)

    if args.push_to_hub:
      logging.info(f"Pushing dataset to Hugging Face Hub")
      prompt_dataset = DatasetDict.load_from_disk(save_dir)
      prompt_dataset.push_to_hub(f"{args.user}/{args.repo_name}")
  except Exception as e:
    logging.error(f"Error generating prompts: {str(e)}")

if __name__ == "__main__":
  main()
