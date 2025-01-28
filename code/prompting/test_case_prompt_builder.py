import os
from jinja2 import Template
import json
import pandas as pd
import logging

def generate_dataframe_schma_json(df):
  schema = {
       "columns": [
           {"name": col, "type": str(df[col].dtype)}
           for col in df.columns
       ]
   }
  json_schema = json.dumps(schema, indent=4)
  return json_schema

def generate_dataframe_description_json(df):
  description = df.describe().to_json(orient='index', indent=4)
  return description

def generate_random_sample_of_n_rows_json(df, n=10):
    return df.sample(n=n).to_json(orient='records', indent=4)

def build_prompt(row, df, skip_description=["029_NYTimes"]):
    """
    Takes a dataset row expects coluums: question, dataset, type or predicted_type
    """
    question = row['question']
    logging.info(f"Building prompt for question: {question}")
    df_random_sample = '{}'
    if not row['dataset'] in skip_description:
       df_random_sample = generate_dataframe_description_json(df) 

    current_dir = os.path.dirname(__file__)
    template_path = os.path.join(current_dir, "prompt-templates/test_case_prompt_template.jinja")
    with open(template_path) as file:
        testcase_template = Template(file.read())

    type_hint = row['type'] if 'type' in row else None       
    type_hint = row['predicted_type'] if 'predicted_type' in row else type_hint 
    
    prompt = testcase_template.render(
        predicted_type=type_hint,
        schema=generate_dataframe_schma_json(df),
        description=generate_dataframe_description_json(df),
        random_sample=df_random_sample,
        question=question,
        columns=list(df.columns)
    )
    return prompt

def build_inference_prompt(row, df, skip_description=["029_NYTimes"], prompt_template_type=None):
    question = row['question']
    df_random_sample = '{}'
    if not row['dataset'] in skip_description:
       df_random_sample = generate_dataframe_description_json(df) 

    current_dir = os.path.dirname(__file__)
    template_path = None
    if prompt_template_type is not None:
        template_path = os.path.join(current_dir, "prompt-templates/{prompt_template_type}/prompt_template.jinja")
    else:
        template_path = os.path.join(current_dir, "prompt-templates/prompt_template.jinja")
    with open(template_path) as file:
        testcase_template = Template(file.read())

    type_hint = row['type'] if 'type' in row else None       
    type_hint = row['predicted_type'] if 'predicted_type' in row else type_hint 
   
    columns_used_hint  = row['columns_used'] if 'columns_used' in row else None 
    columns_used_hint = row['predicted_columns'] if 'predicted_columns' in row else columns_used_hint
    
    prompt = testcase_template.render(
        predicted_type=type_hint,
        columns_used=columns_used_hint,
        schema=generate_dataframe_schma_json(df),
        description=generate_dataframe_description_json(df),
        random_sample=df_random_sample,
        question=question,
        columns=list(df.columns)
    )
    return prompt