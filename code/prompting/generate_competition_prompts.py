import json
import pandas as pd
from datasets import load_dataset

from test_case_prompt_builder import (build_inference_prompt, prompt_generator)

from test_case_load_dataset import (load_phase_dataset)

def create_prompt_file(qa, row_idx,df, split="dev", output_dir="./"):
  prompt =  build_inference_prompt(qa[row_idx], df)
  with open(f"{output_dir}/prompt_{row_idx}.py", "w") as f:
    f.write(prompt)

def generate_all_prompts(split="dev"):
  questions_dataset, dataset_map = load_phase_dataset(phase="competition",
                                                      split=split)
  
  OUTPUT_DIR=f"~/.cache/{split}-prompts"
  for row_idx in range(len(questions_dataset)):
      dataset_id = questions_dataset[row_idx]['dataset']
      if dataset_id == "029_NYTimes":
          continue
      print(f"Generate prompt {row_idx}")
      create_prompt_file(questions_dataset, 
                         row_idx, 
                         dataset_map[dataset_id], 
                         split=split, 
                         output_dir=OUTPUT_DIR)


generate_all_prompts(split="train")

