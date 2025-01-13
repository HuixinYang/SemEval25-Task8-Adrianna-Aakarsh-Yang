import os
import json
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, Features, Value

from test_case_prompt_builder import (build_inference_prompt)
from test_case_load_dataset import (load_phase_dataset)

import logging
import argparse

logging.basicConfig(level=logging.INFO)

# Hugging Face Schema Definition - prompts
features = Features({
    "semeval_id": Value("string"),
    "question": Value("string"),
    "dataset": Value("string"),
    "split": Value("string"),
    "predicted_type": Value("string"),
    "phase": Value("string"),
    "content": Value("string"),
    "update_timestamp": Value("string") 
})

def create_prompt_file(qa, row_idx,df, split="dev", output_dir="./"):
  logging.info(f"Building prompt for question: {qa[row_idx]['question']}")
  prompt =  build_inference_prompt(qa[row_idx], df)
  with open(f"{output_dir}/prompt_{row_idx}.py", "w") as f:
    f.write(prompt)
  
def prompt_generator(question_dataset, dataset_map, split='dev', phase='competition'):
  for row_idx in range(len(question_dataset)):
    row = question_dataset[row_idx]
    dataset = row['dataset']
    prompt =  build_inference_prompt(row, dataset_map[dataset])

    yield {
        'semeval_id': row['semeval_id'],
        'split': split,
        'phase': phase,
        'predicted_type': row['predicted_type'] if 'predicted_type' in row else None, 
        'question': row['question'],
        'content': prompt
    }

def generate_all_prompts(phase="competition", split="dev", prompt_dataset=None):
  if isinstance(prompt_dataset, DatasetDict):
    prompt_dataset = prompt_dataset[split]
  questions_dataset, dataset_map = load_phase_dataset(phase=phase, split=split)
  prompt_dataset  =DatasetDict({
    split:  Dataset.from_generator(
      lambda s=split: prompt_generator(questions_dataset, dataset_map, split=s, phase=phase))
    for split in [split]
  })
  logging.debug(f"Prompt Dataset: {prompt_dataset}")
  prompt_dataset = prompt_dataset if not isinstance(prompt_dataset, DatasetDict) else DatasetDict({split: prompt_dataset[split]})
  return prompt_dataset

def main():
  parser = argparse.ArgumentParser(description="Generate competition prompts")
  parser.add_argument("--phase", type=str, default="competition", required=True, help="Phase of the competition")
  parser.add_argument("--split", type=str, default="dev", required=True, help="Dataset split")
  parser.add_argument("--user_name", default="aakarsh-nair", type=str, help="User name for Hugging Face Hub")
  parser.add_argument("--repo_name", default="semeval-2025-task-8-prompts-competition", type=str, required=True, help="Repository name for Hugging Face Hub")
  parser.add_argument("--cache_dir_path", type=str, default="~/.cache/", help="Cache directory path")
  parser.add_argument("--push_to_hub", default=False, action="store_true", help="Flag to push dataset to Hugging Face Hub")

  args = parser.parse_args()

  logging.info(f"Generating prompts for phase: {args.phase} and split: {args.split}")
  prompt_dataset = generate_all_prompts(phase=args.phase, split=args.split)
  cache_dir = os.path.expanduser(args.cache_dir_path)
  prompt_dataset.save_to_disk(f"{cache_dir}/{args.repo_name}")

  if args.push_to_hub:
    logging.info(f"Pushing dataset to Hugging Face Hub")
    prompt_dataset = DatasetDict.load_from_disk(f"{cache_dir}/{args.repo_name}")
    prompt_dataset.push_to_hub(f"{args.user_name}/{args.repo_name}", use_temp_dir=True)


if __name__ == "__main__":
  main()
