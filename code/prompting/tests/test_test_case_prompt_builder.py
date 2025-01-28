import pandas as pd
import re
from test_case_prompt_builder import generate_dataframe_schma_json
from test_case_prompt_builder import (generate_dataframe_schma_json, 
                                      generate_dataframe_description_json, 
                                      generate_random_sample_of_n_rows_json, 
                                      build_prompt)
import os

def test_build_prompt():
    row = {
        'question': "Who is the author of the article?",
        'dataset': "029_NYTimes",
        'type': "category"
    }
    df = pd.DataFrame.from_dict({
        'author': ['John Doe', 'Jane Doe'],
        'article': ['Article 1', 'Article 2']
    })
    
    test_case_prompt = build_prompt(row, df, skip_description=["029_NYTimes"])
    print(test_case_prompt)
    assert test_case_prompt is not None
    match_methods = ["dummy_data", "test_answer", "answer"]
    for method in match_methods:
        assert re.search(method, test_case_prompt) is not None