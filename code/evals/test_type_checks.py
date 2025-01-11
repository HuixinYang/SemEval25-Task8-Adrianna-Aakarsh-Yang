from type_checks import (check_boolean, 
                         check_number, 
                         check_category, 
                         check_list_category, 
                         check_list_number)

def test_check_boolean():
    assert check_boolean("true", "true")
    assert check_boolean("yes", "yes")
    assert not check_boolean("true", "false")
    assert not check_boolean("false", "true")
    assert not check_boolean("yes", "no")
    assert not check_boolean("no", "yes")
    assert check_boolean("true", "yes")
    assert check_boolean("yes", "true")
    assert check_boolean("false", "no")
    assert check_boolean("no", "false")
    assert not check_boolean("true", "no")
    assert not check_boolean("no", "true")
   
def test_check_number():
    assert check_number("1", "1")
    assert check_number("1.0", "1.0")
    assert check_number("1.0", "1")
    # check rounding
    assert check_number("1.0", "1.001")
    assert not check_number("1.0", "1.01")

def test_check_category():
    assert check_category("a", "a")
    assert check_category("a", "a ")
    assert check_category("a ", "a")
    assert check_category("a ", "a ")

def test_check_list_category():
    assert check_list_category("['a', 'b']", "['a', 'b']")
    assert check_list_category("['a', 'b']", "['b', 'a']")
    assert not check_list_category("['a', 'b']", "['b', 'a', 'a']") 
    assert not check_list_category("['a', 'b']", "['b']")
    assert not check_list_category("['a', 'b']", "['a']")
    assert not check_list_category("['a', 'b']", "['a', 'c']")
    assert not check_list_category("['a', 'b']", "['c', 'b']")
    
def test_check_list_number():
    assert check_list_number("[1, 2]", "[1, 2]")
    assert check_list_number("[1, 2]", "[2, 1]")
    assert not check_list_number("[1, 2]", "[1, 2, 2]") 
    assert not check_list_number("[1, 2]", "[2]")
    assert not check_list_number("[1, 2]", "[1]")
    assert not check_list_number("[1, 2]", "[1, 3]")
    assert not check_list_number("[1, 2]", "[3, 2]")
    # floating points
    assert check_list_number("[1.0, 2.0]", "[1.0, 2.0]")
    assert check_list_number("[1.0, 2.0]", "[2.0, 1.0]")
    # rounding to 2 decimal places
    assert check_list_number("[1.0, 2.001]", "[1.001, 2.0]")
    assert not check_list_number("[1.0, 2.01]", "[1.01, 2.0, 2.0]")
    