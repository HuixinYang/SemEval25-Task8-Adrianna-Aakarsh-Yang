import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any

def default_compare(value, truth, semantic):
    STRIP_CHARS = "[]'\" "
    semantic = semantic.strip()
    valid_null_set = [None, "nan", "", " ", np.nan, "np.nan", "None"]

    if str(value).strip(STRIP_CHARS) in valid_null_set and str(truth).strip(STRIP_CHARS) in valid_null_set:
        return True
    if str(value).strip(STRIP_CHARS) in valid_null_set or str(truth).strip(STRIP_CHARS) in valid_null_set:
        return False

    if semantic == "boolean":
        valid_true_values = ['true', 'yes', 'y']
        valid_false_values = ['false', 'no', 'n']
        value_str = str(value).strip(STRIP_CHARS).lower()
        truth_str = str(truth).strip(STRIP_CHARS).lower()
        return (value_str in valid_true_values and truth_str in valid_true_values) or (value_str in valid_false_values and truth_str in valid_false_values)
    elif semantic == "category":
        value_str = str(value).strip(STRIP_CHARS)
        truth_str = str(truth).strip(STRIP_CHARS)
        if value_str == truth_str:
            return True

        try:
            value_date = pd.to_datetime(value_str).date()
            truth_date = pd.to_datetime(truth_str).date()
            return value_date == truth_date
        except (ValueError, TypeError):
            if not value_str and not truth_str:
                return True
            return value_str == truth_str
    elif semantic == "number":
        try:
            value_cleaned = ''.join(char for char in str(value) if char.isdigit() or char in ['.', '-'])
            truth_cleaned = ''.join(char for char in str(truth) if char.isdigit() or char in ['.', '-'])
            return round(float(value_cleaned), 2) == round(float(truth_cleaned), 2)
        except:
            return False
    elif semantic == "list[category]":
        try:
            value_list = [item.strip(STRIP_CHARS) for item in str(value).strip('[]').split(',')]
            truth_list = [item.strip(STRIP_CHARS) for item in str(truth).strip('[]').split(',')]
            value_list = [
                v if v not in valid_null_set else ""
                for v in value_list
            ]
            truth_list = [
                t if t not in valid_null_set else "" for t in truth_list
            ]
            if len(value_list) != len(truth_list):
                return False

            # Attempt to parse each item as a date
            try:
                value_dates = [pd.to_datetime(item).date() for item in value_list]
                truth_dates = [pd.to_datetime(item).date() for item in truth_list]
                return set(value_dates) == set(truth_dates)
            except (ValueError, TypeError):
                # If parsing as dates fails, compare as strings
                return set(value_list) == set(truth_list)
        except Exception as exc:
            return False
    elif semantic == "list[number]":
        try:
            value_list = sorted(float(''.join(c for c in v.strip() if c.isdigit() or c in ['.', '-'])) for v in str(value).strip('[]').split(',') if v.strip())
            truth_list = sorted(float(''.join(c for c in t.strip() if c.isdigit() or c in ['.', '-'])) for t in str(truth).strip('[]').split(',') if t.strip())

            value_list = [int(v * 100) / 100 for v in value_list]
            truth_list = [int(t * 100) / 100 for t in truth_list]

            if len(value_list) != len(truth_list):
                return False
            
            return set(value_list) == set(truth_list)
        except Exception as exc:
            return False
    else:
        raise Exception(f"Semantic not supported: {semantic}")

def main(input_path, truth_path, semantics_path):   
    # Load the data
    responses = []
    with open(input_path, "r") as f:
        for line in f:
            responses.append(line.strip()) 
     
    truths = []
    with open(truth_path, "r") as f:
        for line in f:
            truths.append(line.strip()) 
            
    semantics = []
    with open(semantics_path, "r") as f:
        for line in f:
            semantics.append(line.strip())

    # Compare the predictions
    correct = 0
    correct_by_semantic = {}
    incorrect_by_semantic = {}
    missing_by_semantic = {}
    for response, truth, semantic in tqdm(zip(responses, truths, semantics), total=len(truths)):
        if default_compare(response, truth, semantic):
            correct_by_semantic[semantic] = correct_by_semantic.get(semantic, 0) + 1
            correct += 1
        elif response == "MISSING_SUBMISSION":
            missing_by_semantic[semantic] = missing_by_semantic.get(semantic, 0) + 1
        else:
            incorrect_by_semantic[semantic] = incorrect_by_semantic.get(semantic, 0) + 1
             
    print(f"Total Accuracy: {correct / len(truths)}")
    semantics = set(semantics)    
    for semantic in semantics:
        correct_count = correct_by_semantic.get(semantic, 0)
        incorrect_count = incorrect_by_semantic.get(semantic, 0)
        missing_count = missing_by_semantic.get(semantic, 0)
        total_count = correct_count + incorrect_count + missing_count
        print(f"{semantic} Accuracy: {correct_count / total_count} ({correct_count}/{total_count})")

        print(f"Number of Missing [{semantic}]: {missing_count}")
        print(f"Number of Incorrect [{semantic}]: {incorrect_count}")
        print(f"Number of  Correct [{semantic}]: {correct_count}") 
        
        
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input", type=str, default="predictions_latest.txt")
    parser.add_argument("--truth", type=str, default="answers/answers.txt")
    parser.add_argument("--semantics", type=str, default="answers/semantics.txt") 
    
    args = parser.parse_args()
    main(args.input, args.truth, args.semantics)