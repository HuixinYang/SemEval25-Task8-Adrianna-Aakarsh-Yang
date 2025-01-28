import sys
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import ast
import re
import json
import pandas as pd

# Assume openai>=1.0.0
from openai import OpenAI
from multiprocessing import cpu_count
from datasets import load_dataset
import logging
from datasets import Dataset
from dataloading.semeval_load_dataset import load_phase_dataset

from prompting.generate_prompts import generate_dataframe_schma_json
logging.basicConfig(level=logging.DEBUG)

DEFAULT_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"
DEEPINFRA_TOKEN=os.getenv('DEEPINFRA_TOKEN')

# Create an OpenAI client with your deepinfra token and endpoint
openai = OpenAI(
    api_key=f"{DEEPINFRA_TOKEN}",
    base_url="https://api.deepinfra.com/v1/openai",
)

def parse_list(return_str, backing_df_columns):
    """
    Parses a string representation of a list and returns the list if valid.

    Args:
        return_str (str): The string representation of the list to be parsed.
        backing_df_columns (list): A list of column names from a DataFrame (not used in the function).

    Returns:
        list: The parsed list if the string is a valid list representation.
        None: If the string is not a valid list representation or an error occurs during parsing.

    Raises:
        ValueError: If the provided string does not represent a list.
    """
    try:
        # Use ast.literal_eval to safely evaluate the string as a Python literal
        parsed_list = ast.literal_eval(return_str)
        if isinstance(parsed_list, list):
            return parsed_list
        else:
            raise ValueError("The provided string does not represent a list.")
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing list: {e}")
        return None
    
def parse_columns(return_str, columns_expected):
    """
    Parses the returned columns from a given string and filters them based on the expected columns.

    Args:
        return_str (str): The string containing the returned columns.
        columns_expected (list): A list of expected column names.

    Returns:
        list: A list of columns that are both in the returned columns and the expected columns.
    """
    returned_columns = parse_columns(return_str)
    columns = []
    if returned_columns:
        for column in returned_columns:
            if column in columns_expected:
                columns.append(column)
    return columns

def get_api_prompt_completion(prompt, model=DEFAULT_MODEL, max_tokens=15):
    chat_completion = openai.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    logging.debug(f"Chat Completion: {chat_completion}")
    return chat_completion.choices[0].message.content

def example_schema():
   return json.dumps({ "columns": [ 
    { "name": "rank", "type": "uint16" }, 
    { "name": "personName", "type": "category" }, 
    { "name": "age", "type": "float64" }, 
    { "name": "finalWorth", "type": "uint32" }, 
    { "name": "category", "type": "category" }, 
    { "name": "source", "type": "category" }, 
    { "name": "country", "type": "category" }, 
    { "name": "state", "type": "category" }, 
    { "name": "city", "type": "category" }, 
    { "name": "organization", "type": "category" }, 
    { "name": "selfMade", "type": "bool" }, 
    { "name": "gender", "type": "category" }, 
    { "name": "birthDate", "type": "datetime64[us, UTC]" }, 
    { "name": "title", "type": "category" }, 
    { "name": "philanthropyScore", "type": "float64" }, 
    { "name": "bio", "type": "object" }, 
    { "name": "about", "type": "object" }]}, indent=4)
 
def prompt_generator(question_row, backing_df):
    prompt = f"""
# TODO: Output only the the columns used to to answer the question 
# DO NOT give any explanation or other infromation except the 
# list of columns used to answer the question. 
#
# Question: What is the age of the youngest billionaire? 
# Answer:

# Description of dataframe schema. 
df_schema = { example_schema() } 
 
columns_used  = ['finalWorth', 'selfMade']


# TODO: Output only the the columns used to to answer the question 
# DO NOT give any explanation or other infromation except the 
# list of columns used to answer the question. 
# 
# Question: {question_row['question']}
# Answer:

df_schema = { generate_dataframe_schma_json(backing_df) }

columns_used  ="""
    return prompt

def process_idx(idx, 
                question_df  = None, 
                datasets_map = None,    
                model = None, 
                max_tries=100):
    """
    Processes a question at a given index in a DataFrame, generates a prompt, and extracts code from the response.

    Args:
        idx (int): The index of the question in the DataFrame.
        question_df (pandas.DataFrame, optional): DataFrame containing questions. Defaults to None.
        model (optional): The model to use for generating the prompt completion. Defaults to None.
        regenerate (bool, optional): Flag to indicate if the prompt should be regenerated. Defaults to False.
        split (str, optional): The data split to use, e.g., "competition". Defaults to "competition".

    Returns:
        str: Extracted code from the response.
    """
    assert model is not None, "Please provide a model for generating the prompt completion."
    logging.debug(f"Processing Index: {idx}")
    question = question_df[idx]
    backing_df = datasets_map[question['dataset']]

    prompt = prompt_generator(question, backing_df)
    logging.debug(f"Prompt: {prompt}")
    
    tries = 0
    response = None
    predicted_columns = None
    while tries < max_tries:
        response = get_api_prompt_completion(prompt, model=model)
        logging.debug(f"Response: {response}")
        predicted_columns = parse_list(response, backing_df.columns)
        logging.debug(f"Predicted Columns: {predicted_columns}")
        if predicted_columns is not None and len(predicted_columns) > 0:
            break
        tries += 1
    return predicted_columns 

def predict_question_columns_used(question_dataset, datasets_map,
                                    max_workers=cpu_count(), 
                                    added_column_name="predicted_columns",
                                    split="competition", 
                                    regenerate=False, 
                                    model=DEFAULT_MODEL):
    """
    Predict the question columns used based on the question text.
    """
    # Parallel execution using ThreadPoolExecutor
    results = []
    # Adjust max_workers based on your system.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:  
            results = executor.map(partial(process_idx, 
                                 model=model, 
                                 question_df=question_dataset,
                                 datasets_map=datasets_map), range(len(questions_dataset)))

    results = list(results)
    print("Results: ", results)
    print("len of results: ", len(results))
    
    def update_predicted_type(df, idx, results):
        df[added_column_name] = str(results[idx])
        return df

    question_dataset = question_dataset.map(lambda df, idx: update_predicted_type(df, idx, list(results)), with_indices=True)
    return question_dataset

if __name__ == '__main__':
    questions_dataset, datasets_map = load_phase_dataset(phase="competition", split="dev")
    questions_dataset = predict_question_columns_used(questions_dataset, datasets_map, model="Qwen/Qwen2.5-Coder-32B-Instruct")
    # save the results 
    questions_dataset[list(["question","dataset","predicted_type", "predicted_columns"])].to_csv("result.csv", index=False)