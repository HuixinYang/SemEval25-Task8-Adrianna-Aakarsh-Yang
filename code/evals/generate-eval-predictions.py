from databench_eval import Evaluator
import pandas as pd
from datasets import load_dataset
from multiprocessing import Pool, Manager, cpu_count
from runners.run_with_test_cases_parallel import extract_functions_and_imports
import numpy as np
from datetime import datetime
import logging
import os
import pyarrow as pa
from datasets import Dataset

logging.basicConfig(level=logging.INFO)

responses = []

def load_table(name, shared_datasets):
    """Load dataset once and store it in shared memory."""
    if name not in shared_datasets:
        shared_datasets[name] = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/{name}/all.parquet")
    return shared_datasets[name]

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
    
          
def process_question(index, shared_data, shared_datasets):
    """Process a single question for parallelization."""
    row = shared_data[index]
    question = row['question']
    df = load_table(row["dataset"], shared_datasets)
    completion = row["completion"]
    value = post_process(completion, df)
    truth = row["answer"] 
    semantic = row["type"]
    correct =  default_compare(value, truth, semantic)
    correct_str = "CORRECT" if correct else "INCORRECT"
    correct_value =value

    print("Q:", question, "Completion:", row["completion"], row["dataset"], f"\n{correct_str}: TRUTH:{truth}, VALUE:{correct_value},  SEMANTIC:{semantic}")
    
    row['is_correct'] = bool(correct)
    row['computed_answer'] = str(value)
    row['update_timestamp'] = datetime.now().isoformat()
    
    return row

def post_process(return_statement, df):
    """Placeholder for the post-processing step."""
    method_template =f"""
def answer(df: pd.DataFrame):
    return {return_statement}
"""
    try:
        exec(
            "global ans\n" + method_template + f"\nans = answer(df)"
        ) 
        return ans
    except Exception as e:
         return f"__RUNTIME_ERROR__{str(e)}"

def generate_responses(dataset, num_workers=4):
    """Generate responses using multiprocessing."""
    with Manager() as manager:
        shared_data = manager.list(dataset['dev'])
        shared_datasets = manager.dict()  

        num_workers = cpu_count()
        with Pool(num_workers) as pool:
            responses = pool.starmap(
                process_question, 
                [(i, shared_data, shared_datasets) for i in range(len(shared_data))]
            )
    return responses

print("Loading dataset...")
dataset_df = load_dataset('aakarsh-nair/semeval-2025-task-8-finetune')
push_to_hub = False
output_file = os.path.join(os.path.dirname(__file__), "updated_rows.parquet")
if os.path.exists(output_file):
    print(f"{output_file} exists. Proceeding with evaluation.")
    if push_to_hub:
        # Load data set from the updateds and save it to hub
        updated_rows = pd.read_parquet(output_file).to_dict(orient='records')
        dataset_df['dev'] = Dataset.from_list(updated_rows)
        pd.read_parquet(output_file).to_dict(orient='records')
        dataset_df.push_to_hub("aakarsh-nair/semeval-2025-task-8-finetune")
else:
    print(f"{output_file} does not exist. Skipping evaluation.")
    # Load dataset
    updated_rows = generate_responses(dataset_df)
    print("Done processing questions.")
    # Persist the updated rows to disk

    schema = pa.schema([
                    ('semeval_id', pa.string()),
                    ('split', pa.string()),
                    ('question', pa.string()),
                    ('dataset', pa.string()),
                    ('prompt', pa.string()),
                    ('completion', pa.string()),
                    ('reward', pa.float64()),
                    ('answer', pa.string()),
                    ('computed_answer', pa.string()),
                    ('computed_sample_answer', pa.string()),
                    ('is_correct', pa.bool_()),
                    ('is_correct_sample', pa.bool_()),
                    ('type', pa.string()),
                    ('sample_answer', pa.string()),
                    ('create_timestamp', pa.string()),
                    ('update_timestamp', pa.string())
    ])

    pd.DataFrame(updated_rows).to_parquet(output_file, engine="pyarrow", schema=schema) 
    if push_to_hub:
        dataset_df['dev'] = updated_rows
        dataset_df.push_to_hub("aakarsh-nair/semeval-2025-task-8-finetune")


"""
for idx in range(len(output_list)):
    print("=="*20+str(idx)+"=="*100)
    texts, rewards = output_list[idx]
    df = pd.DataFrame({'text': texts, 'reward': rewards})
    max_reward_index = df['reward'].idxmax()
    df = df[df["reward"] == max(df["reward"])]
    answers = pd.DataFrame()
    for sentence in df['text']:
      #print(sentence)
      sentence = df.iloc[0]['text']
      answer = post_process(sentence, df_all)
      answers = pd.concat([answers, pd.DataFrame({'answer': [answer]})])

    answer_frequency = answers['answer'].value_counts()

    if len(answer_frequency) >0:
      most_frequent_anser = answer_frequency.idxmax()
      print(most_frequent_anser)
      responses.append(most_frequent_anser)
    else:
      responses.append('')
      print('')
    with open("/content/drive/MyDrive/TUE-WINTER-2024/CHALLENGES-CL/predictions.txt", "w") as f:
      for response in responses:
          f.write(str(response) + "\n")
    with open("/content/drive/MyDrive/TUE-WINTER-2024/CHALLENGES-CL/predictions_lite.txt", "w") as f:
      for response in responses:
          f.write(str(response) + "\n")
          
          
evaluator = Evaluator(qa=qa)

# Run the evaluator
with open("/content/drive/MyDrive/TUE-WINTER-2024/CHALLENGES-CL/predictions.txt", "r") as f:
    responses = f.read().splitlines()
with open("/content/drive/MyDrive/TUE-WINTER-2024/CHALLENGES-CL/predictions_lite.txt", "r") as f:
    responses_lite = f.read().splitlines()
print(f"DataBench accuracy is {evaluator.eval(responses)}")  # ~0.15
print(
    f"DataBench_lite accuracy is {evaluator.eval(responses_lite, lite=True)}"
)  # ~0.07
"""