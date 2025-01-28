# TODO: This file is no longer used MARK FOR DELETION
import json
import pandas as pd
from datasets import load_dataset

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

def generate_dataframe_categorical_cols_json(df):
    categorical_data = {}
    for col_name in df.columns:
        if df[col_name].dtype == 'category' and len(df_all[col_name].unique()) < 300 and  len(df_all[col_name].unique()) > 0 :
            categorical_data[col_name] = df_all[col_name].unique().tolist()

    json_data = json.dumps(categorical_data, indent=4)
    return json_data

def generate_random_sample_of_n_rows_json(df, n=10):
    return df.sample(n=n).to_json(orient='records', indent=4)

def get_dataframe_by_id(df_id):
    parquet_file = f"hf://datasets/cardiffnlp/databench/data/{df_id}/all.parquet"
    print(f"Loading {parquet_file}")
    df = pd.read_parquet(parquet_file)
    return df


def prompt_generator(row, df):
    question = row['question']
    df_random_sample = '{}'
    if not row['dataset'] == "029_NYTimes":
       df_random_sample = generate_dataframe_description_json(df) 
    print(f"Generating:{question}, dataset:{row['dataset']}")
    prompt = f"""
# Instructions: Generate ONLY python code. Do not include explanations.  
# you can use pandas and numpy. Use the meta data information from df_schema, df_descprtion.
import pandas as pd
import numpy as np


# Description of dataframe schema.
df_schema = {generate_dataframe_schma_json(df)}

# Description of dataframe columns.
df_descrption = {generate_dataframe_description_json(df)}

# Randome sample of 10 rows from the dataframe.
df_random_sample = {df_random_sample}

# TODO: complete the following function in one line, by completing the return statement. It should give the answer to: How many rows are there in this dataframe?
def example(df: pd.DataFrame):
    df.columns=["A"]
    return df.shape[0]

# TODO: complete the following function in one line, by completing the return statement. It should give the answer to: {question}
def answer(df: pd.DataFrame):
    df.columns = {list(df.columns)}
    return"""
    return prompt

def create_prompt_file(qa, row_idx,df, split="dev", output_dir="./"):
  prompt = prompt_generator(qa[row_idx], df)
  with open(f"{output_dir}/prompt_{row_idx}.py", "w") as f:
    f.write(prompt)

def fetch_all_dataframes(dataset):
  dataset_ids  = set(map(lambda qa: qa['dataset'],  dataset))
  retval = { ds_id: get_dataframe_by_id(ds_id) for ds_id in dataset_ids }
  return retval


def generate_all_prompts(split="dev"):
  ds = load_dataset("cardiffnlp/databench", name="semeval", split=split)
  dataset_map = fetch_all_dataframes(ds)
  OUTPUT_DIR=f"/content/drive/MyDrive/TUE-WINTER-2024/CHALLENGES-CL/{split}-prompts"
  for row_idx in range(len(ds)):
      dataset_id = ds[row_idx]['dataset']
      if dataset_id == "029_NYTimes":
          continue
      print(f"Generate prompt {row_idx}")
      create_prompt_file(ds, row_idx, dataset_map[dataset_id], split=split, output_dir=OUTPUT_DIR)

if __name__ == "__main__":
    generate_all_prompts(split="train")

