import os
from parse_outputs import read_pickle_output, find_all_output_files, make_dataset

def test_read_pickle_output():
    result= read_pickle_output("../output/parallel-output_list-32-06-01-2025.pkl") 
    assert len(result) > 3 
    assert result[0].keys() == {'code', 'reward'}
   
def test_find_all_output_files():
    result = find_all_output_files("../output")
    assert len(result) > 3
    assert result[0].endswith(".pkl")
    assert "output_list" in result[0] 
    
    
def test_make_dataset():
    results = {
        '1': [{
            'code': 'return 1',
            'reward': 1,
            'split': 'dev'
        }],
        '2': [{
            'code': 'return 2',
            'reward': 2,
            'split': 'dev'
        }],
    }
   
    # output_file = "../output/parallel-output_list-32-06-01-2025.pkl" 
    dataset = make_dataset(results)
    assert len(dataset) == 2