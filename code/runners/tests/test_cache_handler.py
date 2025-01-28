from runners.run_with_test_cases_competition import construct_cache_key
from runners.cache_handler import cache_handler


import pytest
from unittest.mock import MagicMock, patch, mock_open
import pandas as pd
import logging
import os
import pickle
import traceback

def test_construct_cache_key():
    args = [1, 2, 3, 4, 5, 6]
    kwargs = {
        'qa_item': {'semeval_id': 1},
        'idx': 2,
        'model': 'model'
    }
    assert construct_cache_key(args, kwargs) ==  \
        ('pipeline-runs/model/mct-result-semeval_id-1.parquet', 2)
        
        
def sample_function(idx, 
                    qa_item, 
                    dataset_map, 
                    test_dataset, 
                    prompt_dataset, 
                    model, 
                    tokenizer, 
                    file_handler, 
                    logger, 
                    config):
    return {"data": "sample_result"}


def test_cache_handler_cache_hit():
    # Setup
    idx = 1
    qa_item = {'semeval_id': 101}
    dataset_map = {}
    test_dataset = pd.DataFrame()
    prompt_dataset = pd.DataFrame()
    model = MagicMock()
    model.name_or_path = "mock_model"
    tokenizer = MagicMock()
    file_handler = MagicMock()
    logger = MagicMock(spec=logging.Logger)
    config = MagicMock()
    
    # Apply decorator with use_cache=True and regenerate=False
    decorated_function = cache_handler(use_cache=True, regenerate=False)(sample_function)
    
    cache_file_path = os.path.expanduser(f"~/.cache/pipeline-runs/{model.name_or_path}/mct-result-semeval_id-{qa_item['semeval_id']}.parquet")
    
    with patch('os.path.exists', return_value=True) as mock_exists, \
         patch('logging.info') as mock_logging_info:
        
        result = decorated_function(idx, qa_item, dataset_map, test_dataset, prompt_dataset, model, tokenizer, file_handler, logger, config)
        
        # Assertions
        assert result is None
        mock_exists.assert_called_once_with(cache_file_path)
        mock_logging_info.assert_called_once_with(f"Skipping cache item {qa_item['semeval_id']} as cache exists.")

'''
def test_cache_handler_cache_miss():
    # Setup
    idx = 2
    qa_item = {'semeval_id': 102}
    dataset_map = {}
    test_dataset = pd.DataFrame()
    prompt_dataset = pd.DataFrame()
    model = MagicMock()
    model.name_or_path = "mock_model"
    tokenizer = MagicMock()
    file_handler = MagicMock()
    logger = MagicMock(spec=logging.Logger)
    config = MagicMock()
    
    # Apply decorator with use_cache=True and regenerate=False
    decorated_function = cache_handler(use_cache=True, regenerate=False)(sample_function)
    
    cache_file_path = os.path.expanduser(f"~/.cache/pipeline-runs/{model.name_or_path}/mct-result-semeval_id-{qa_item['semeval_id']}.parquet")
    output_dir = os.path.dirname(cache_file_path)
    
    with patch('os.path.exists', return_value=False) as mock_exists, \
         patch('os.makedirs') as mock_makedirs, \
         patch('pickle.dump') as mock_pickle_dump, \
         patch('logging.info') as mock_logging_info:
        
        result = decorated_function(idx, qa_item, dataset_map, test_dataset, prompt_dataset, model, tokenizer, file_handler, logger, config)
        
        # Assertions
        assert result == {"data": "sample_result"}
        mock_exists.assert_called_once_with(cache_file_path)
        mock_makedirs.assert_called_once_with(output_dir, exist_ok=True)
        mock_pickle_dump.assert_called_once()
        mock_logging_info.assert_any_call(f"Saved result to cache: {cache_file_path}")
'''
