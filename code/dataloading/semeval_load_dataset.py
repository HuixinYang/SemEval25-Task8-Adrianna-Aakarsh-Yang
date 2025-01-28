from datasets import load_dataset
import os
import pandas as pd
from datasets import Dataset


def read_dataframe_by_id(df_id, phase = None, lite = False):
    """
    Read the parquet file by id, either from the competition dataset 
    or the databench dataset.
    """
    parquet_file = None
    file_name = 'sample.parquet' if lite else 'all.parquet'
    
    if phase == "competition":
        parquet_file = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'competition', df_id, file_name)
    else:
        parquet_file = f"hf://datasets/cardiffnlp/databench/data/{df_id}/{file_name}"
        
    print(f"Loading {parquet_file}")
    df = pd.read_parquet(parquet_file)
    return df

def fetch_all_dataframes(dataset, phase=None, lite=False):
    """
    Fetch all dataframes referenced in the dataset 
    """
    dataset_ids = set(map(lambda row: row['dataset'],  dataset))
    retval = { 
        ds_id: read_dataframe_by_id(ds_id, phase=phase, lite=lite) for ds_id in dataset_ids 
    }
    return retval

def load_phase_dataset(phase="competition", split="dev", limit=None, lite=False):
    if phase == "competition":
        # Load the competition dataset
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'competition', 'test_qa_enhanced.csv')
        questions_dataset =  Dataset.from_pandas(pd.read_csv(file_path))
        questions_dataset = questions_dataset.map(lambda example, idx: {"semeval_id": idx}, with_indices=True)
        if limit:
            questions_dataset = questions_dataset.select(range(limit))
        datasets_map = fetch_all_dataframes(questions_dataset, phase=phase, lite=lite)
    else:
        # Questions Dataset 
        questions_dataset = load_dataset("cardiffnlp/databench", name="semeval", split=split)
        # add semeval_id to the dataset
        questions_dataset = questions_dataset.map(lambda example, idx: {"semeval_id": idx}, with_indices=True)
        # For dev dataset, we will use the type as the predicted
        questions_dataset['predicted_type'] = questions_dataset['type']
        # For dev dataset, we will use the columns_used as the predicted columns
        questions_dataset['predicted_columns'] = questions_dataset['columns_used']
        if limit:
            questions_dataset = questions_dataset.select(range(limit))
        datasets_map = fetch_all_dataframes(questions_dataset)
        
    return questions_dataset, datasets_map