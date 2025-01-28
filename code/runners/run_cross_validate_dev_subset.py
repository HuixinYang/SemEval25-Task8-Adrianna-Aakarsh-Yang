from concurrent.futures import ThreadPoolExecutor, as_completed

from databench_eval import Runner, Evaluator, utils
from datasets import load_dataset
from dyna_gym.pipelines import uct_for_hf_transformer_pipeline
from py_evaluator.test_case_runner import test_run_code
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer,AutoModel
import argparse
import ast
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
import torch
import traceback
import transformers
import pyarrow as pa
import pyarrow.parquet as pq
from cache_handler import cache_handler
 
from dataloading.semeval_load_dataset import load_phase_dataset
from py_evaluator.test_case_runner import is_predicted_type 

logging.basicConfig(level=logging.INFO)

"""
    1. Fix the schema differences between the COMPETITION prompt and DEV
    2. Run experimiments with different prompts and models using cross-validation 
    on random subsets of the dev data 
    3. Evaluate the performance of the models using the MCTS outputs
    4. Save the results in a parquet file
    
Let us examin differences between competition and dev data

1. TEST-CASES: 
    prompts: aakarsh-nair/semeval-2025-task-8-prompts-competition
        semeval_id:
        split:
        phase:
        question: 
        dataset:
        predicted_type: 
        content:
        update_timestamp:
        
    test-cases: aakarsh-nair/semeval-2025-task-8-test-cases-competition
        semeval_id:
        split:
        phase:
        question:
        dataset:
        predicted_type:
        model:
        content:
        update_timestamp:
        
2. DEV:
    prompts: aakarsh-nair/semeval-2025-task-8-prompts
        id: -> TODO : needs to be renamed to semeval_id
        split:
        content:
        qeustion: -> TODO: needs to be added. 
        dataset: -> TODO: needs to be added.
        predicted_type: -> TODO: needs to be added.
        update_timestamp: -> TODO: needs to be added.
        
        
    test-cases: aakarsh-nair/semeval-2025-task-8-test-cases
        id: -> TODO : needs to be renamed to semeval_id
        split:
        model:
        content:
        question: -> TODO: needs to be added.
        dataset: -> TODO: needs to be added.
        predicted_type: -> TODO: needs to be added.
        update_timestamp: -> TODO: needs to be added.
    
"""

def run(sample_size):
    # 3. Possibly subsample the dev-data
    if sample_size is not None and sample_size < len(questions_dataset):
        random_indices = np.random.choice(len(questions_dataset), 
                                          size=sample_size, 
                                          replace=False)
        
        questions_dataset = questions_dataset.select(random_indices)
        logging.info(f"Subsampled the dev dataset to {len(questions_dataset)} items")


def evaluate_mcts_performance(results_dict):
    """
    Compute a simple metric over the MCTS outputs, e.g., average of the best reward.
    """
    best_rewards = []
    for idx, result in results_dict.items():
        if "rewards" in result and len(result["rewards"]) > 0:
            best_rewards.append(max(result["rewards"]))
        else:
            # If there is an error or empty reward, you can skip or treat as 0
            best_rewards.append(0.0)
    if len(best_rewards) == 0:
        return {"avg_best_reward": 0.0, "num_samples": 0}
    return {
        "avg_best_reward": float(np.mean(best_rewards)),
        "num_samples": len(best_rewards),
    }