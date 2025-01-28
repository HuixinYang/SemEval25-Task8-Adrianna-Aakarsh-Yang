from prompting.predict_columns_used import parse_list

def test_parse_list():
    # Setup
    return_str = "['name', 'age']"
   
    columns = ['name', 'age'] 
    # Apply function
    result = parse_list(return_str, columns)
    
    # Assertions
    assert result == ['name', 'age']