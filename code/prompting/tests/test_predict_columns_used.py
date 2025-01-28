from prompting.predict_columns_used import parse_list

def test_parse_list():
    # Setup
    return_str = "['cat', 'dog']"
    
    # Apply function
    result = parse_list(return_str)
    
    # Assertions
    assert result == ['cat', 'dog']