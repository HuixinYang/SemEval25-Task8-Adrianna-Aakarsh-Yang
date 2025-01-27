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

from generate_prompts import generate_dataframe_schma_json
logging.basicConfig(level=logging.INFO)

"""
We predict the question category based on question test. 

The question categories are:
    - boolean: Valid answers include True/False, Y/N, Yes/No (all case insensitive).
    - category: A value from a cell (or a substring of a cell) in the dataset.
    - number: A numerical value from a cell in the dataset, which may represent a computed statistic (e.g., average, maximum, minimum).
    - list[category]: A list containing a fixed number of categories. The expected format is: "['cat', 'dog']". Pay attention to the wording of the question to determine if uniqueness is required or if repeated values are allowed.
    - list[number]: Similar to list[category], but with numbers as its elements.
"""
DEEPINFRA_TOKEN=os.getenv('DEEPINFRA_TOKEN')

# Create an OpenAI client with your deepinfra token and endpoint
openai = OpenAI(
    api_key=f"{DEEPINFRA_TOKEN}",
    base_url="https://api.deepinfra.com/v1/openai",
)

def is_valid_category(response):
    expected_categories = [
        "boolean", 
        "category", 
        "number", 
        "list[category]", 
        "list[number]"
    ]
    return response in expected_categories

def extract_code_from_response(response):
    expected_categories = [
        "boolean", 
        "category", 
        "number", 
        "list[category]", 
        "list[number]"
    ]
    print("Response: ", response)
    for category in expected_categories:
        if category.lower() == response.lower():
            return category 

def get_api_prompt_completion(prompt, model="Qwen/Qwen2.5-Coder-32B-Instruct", max_tokens=15):
    chat_completion = openai.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return chat_completion.choices[0].message.content

def prompt_generator(question_row, backing_df):
    prompt = f""""
'''
# TODO: Output only the the columns used to to answer the question 
# DO NOT give any explanation or other infromation except the 
# list of columns used to answer the question. 
#
# Question: What is the age of the youngest billionaire? 
# Answer:

# Description of dataframe schema. 
df_schema = { "columns": [ 
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
    { "name": "about", "type": "object" } 
    ]
}
 
columns_used  = ['finalWorth', 'selfMade']


# TODO: Output only the the columns used to to answer the question 
# DO NOT give any explanation or other infromation except the 
# list of columns used to answer the question. 
# 
# Question: {question_row['question']}
# Answer:

df_schema = {generate_dataframe_schma_json(backing_df)}

columns_used  = [
"""
    return prompt

def process_idx(idx, 
                question_df  = None, 
                datasets_map = None,    
                model = None, 
                regenerate = False, 
                split="competition", 
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
    logging.info(f"processing index {idx}")
    question = question_df[idx]
    backing_df = datasets_map[question['dataset']]

    prompt = prompt_generator(question, backing_df)
    logging.info(f"Prompt: {prompt}")
    '''
    tries = 0
    response = None
    predicted_category = None
    while tries < max_tries:
        response = get_api_prompt_completion(prompt, model=model)
        predicted_category = extract_code_from_response(response) 
        if is_valid_category(predicted_category):
            break
        tries += 1

    return predicted_category 
    '''

def load_compeition_questions(file_path):
    df  = pd.read_csv(file_path)
    return df

def predict_question_columns_used(question_dataset, datasets_map,
                                    max_workers=cpu_count(), 
                                    split="competition", 
                                    regenerate=False, 
                                    model="nvidia/Llama-3.1-Nemotron-70B-Instruct"):
    """
    Predict the question columns used based on the question text.
    """
    # Parallel execution using ThreadPoolExecutor

    '''
    question_df = None
    if split == "competition":
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'competition', 'test_qa.csv')
        competition_df = load_compeition_questions(file_path)
        question_df =  Dataset.from_pandas(competition_df)
        question_df = question_df.add_column("predicted_column_used", [""] * len(question_df))
    elif split =="dev":
        question_df = load_dataset("cardiffnlp/databench", name="semeval", split=split)
        question_df = question_df.add_column("predicted_columns_used", [""] * len(question_df))
    elif split == "train":
        question_df = load_dataset("cardiffnlp/databench", name="semeval", split=split)
        question_df = question_df.add_column("predicted_columns_used", [""] * len(question_df))
    '''
    
    results = []
    # Adjust max_workers based on your system.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:  
            results = executor.map(partial(process_idx, 
                                 model=model, 
                                 regenerate=regenerate, 
                                 question_df=question_dataset,
                                 datasets_map=datasets_map,    
                                 split=split), range(len(question_df)))

    results = list(results)
    print("Results: ", results)
    print("len of results: ", len(results))
    
    def update_predicted_type(df, idx, results):
        df['predicted_type'] = str(results[idx])
        return df

    question_df = question_df.map(lambda df, idx: update_predicted_type(df, idx, list(results)), with_indices=True)
    return question_df

if __name__ == '__main__':
    questions_dataset, datasets_map = load_phase_dataset(phase="competition", split="dev")
     
    # question_df = predict_question_columns_used(questions_dataset, datasets_map)
    process_idx(0, questions_dataset, datasets_map) 
    # Save the results
    #question_df.to_csv("results.csv", index=False)