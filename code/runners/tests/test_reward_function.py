
# test_reward_fn.py
import pytest
import logging
import numpy as np
import pandas as pd
import os

# We import the reward function and any dependencies we need
from runners.reward_function import error_detecting_reward_fn
from py_evaluator.utils import (run_all_tests_for_answer, extract_return_statement,
                                generate_method_template, post_process)
from py_evaluator.test_case_runner import is_predicted_type


@pytest.fixture(autouse=True)
def configure_logging():
    logging.basicConfig(level=logging.DEBUG)

@pytest.fixture
def mock_backing_df():
    """
    Returns a mock backing DataFrame. If your real usage needs specific
    columns or data, fill them here. 
    """
    random_seed = 42
    np.random.seed(random_seed) 
    Age = np.random.randint(18, 61, 1470, dtype='uint8') 
    Attrition = np.random.choice(['Yes', 'No'], 1470) 
    BusinessTravel = np.random.choice(['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'], 1470)
    DailyRate = np.random.randint(102, 1500, 1470, dtype='uint16') 
    Department = np.random.choice(['Sales', 'Research & Development', 'Human Resources'], 1470) 
    DistanceFromHome = np.random.randint(1, 30, 1470, dtype='uint8') 
    Education = np.random.randint(1, 6, 1470, dtype='uint8') 
    EducationField = np.random.choice(['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'], 1470) 
    EmployeeCount = np.ones(1470, dtype='uint8') 
    EmployeeNumber = np.arange(1, 1471, dtype='uint16') 
    EnvironmentSatisfaction = np.random.randint(1, 5, 1470, dtype='uint8') 
    Gender = np.random.choice(['Male', 'Female'], 1470) 
    HourlyRate = np.random.randint(30, 101, 1470, dtype='uint8') 
    JobInvolvement = np.random.randint(1, 5, 1470, dtype='uint8') 
    JobLevel = np.random.randint(1, 6, 1470, dtype='uint8') 
    JobRole = np.random.choice(['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Sales Representative', 'Healthcare Representative', 'Manager', 'Research Director', 'Human Resources'], 1470) 
    JobSatisfaction = np.random.randint(1, 5, 1470, dtype='uint8') 
    MaritalStatus = np.random.choice(['Single', 'Married', 'Divorced'], 1470) 
    MonthlyIncome = np.random.randint(1009, 20000, 1470, dtype='uint16') 
    MonthlyRate = np.random.randint(2094, 27000, 1470, dtype='uint16') 
    NumCompaniesWorked = np.random.randint(0, 10, 1470, dtype='uint8') 
    Over18 = np.repeat('Y', 1470) 
    OverTime = np.random.choice(['Yes', 'No'], 1470) 
    PercentSalaryHike = np.random.randint(11, 26, 1470, dtype='uint8') 
    PerformanceRating = np.random.randint(3, 5, 1470, dtype='uint8') 
    RelationshipSatisfaction = np.random.randint(1, 5, 1470, dtype='uint8') 
    StandardHours = np.repeat(80, 1470) 
    StockOptionLevel = np.random.randint(0, 4, 1470, dtype='uint8') 
    TotalWorkingYears = np.random.randint(0, 41, 1470, dtype='uint8') 
    TrainingTimesLastYear = np.random.randint(0, 7, 1470, dtype='uint8') 
    WorkLifeBalance = np.random.randint(1, 5, 1470, dtype='uint8') 
    YearsAtCompany = np.random.randint(0, 41, 1470, dtype='uint8') 
    YearsInCurrentRole = np.random.randint(0, 19, 1470, dtype='uint8') 
    YearsSinceLastPromotion = np.random.randint(0, 16, 1470, dtype='uint8') 
    YearsWithCurrManager = np.random.randint(0, 18, 1470, dtype='uint8') 
    data_dict = {'Age': Age, 'Attrition': Attrition, 'BusinessTravel': BusinessTravel, 'DailyRate': DailyRate, 'Department': Department, 'DistanceFromHome': DistanceFromHome, 'Education': Education, 'EducationField': EducationField, 'EmployeeCount': EmployeeCount, 'EmployeeNumber': EmployeeNumber, 'EnvironmentSatisfaction': EnvironmentSatisfaction, 'Gender': Gender, 'HourlyRate': HourlyRate, 'JobInvolvement': JobInvolvement, 'JobLevel': JobLevel, 'JobRole': JobRole, 'JobSatisfaction': JobSatisfaction, 'MaritalStatus': MaritalStatus, 'MonthlyIncome': MonthlyIncome, 'MonthlyRate': MonthlyRate, 'NumCompaniesWorked': NumCompaniesWorked, 'Over18': Over18, 'OverTime': OverTime, 'PercentSalaryHike': PercentSalaryHike, 'PerformanceRating': PerformanceRating, 'RelationshipSatisfaction': RelationshipSatisfaction, 'StandardHours': StandardHours, 'StockOptionLevel': StockOptionLevel, 'TotalWorkingYears': TotalWorkingYears, 'TrainingTimesLastYear': TrainingTimesLastYear, 'WorkLifeBalance': WorkLifeBalance, 'YearsAtCompany': YearsAtCompany, 'YearsInCurrentRole': YearsInCurrentRole, 'YearsSinceLastPromotion': YearsSinceLastPromotion, 'YearsWithCurrManager': YearsWithCurrManager} 
    return pd.DataFrame(data_dict) 


@pytest.fixture
def mock_prompt_item():
    """
    Returns a mock prompt_item dict that includes 'predicted_type' 
    and anything else you need.
    """
    return [{
        'semeval_id': 1,
        'split': 'dev',
        'phase':'competition',
        'question': 'What is the average age of our employees ?',
        'predicted_type': 'number',
        'content': open(os.path.join(os.path.dirname(__file__), 'test_data/sample_prompt_1.py')).read(),
        'update_timestamp': None,
    }]

@pytest.fixture
def mock_tests_all_pass():
    """
    Returns test cases that, if the code returns x+1, should all pass.
    The structure depends on how run_all_tests_for_answer processes them.
    """
    return [{
            'semeval_id': 1,
            'split': 'dev',
            'phase':'competition',
            'question': 'What is the average age of our employees ?',
            'dataset': '066_IBM_HR' ,
            'predicted_type': 'number',
            'model': 'Qwen/Qwen2.5-Coder-32B-Instruct',
            'content': open(os.path.join(os.path.dirname(__file__), 'test_data/sample_test_1.py')).read(),
            'update_timestamp': None,
        }]

@pytest.fixture
def mock_tests_partial_fail():
    """
    Returns test cases that will pass for 2 + 1, but fail on 2 + 2, etc.
    So we can simulate partial pass scenarios.
    """
    return [
        {'input': [1], 'output': 2},  # 1 + 1 = 2 => passes if code does x+1
        {'input': [2], 'output': 4}   # 2 + 2 = 4 => passes if code does x+2
    ]

def test_all_tests_pass_correct_type(mock_prompt_item, mock_backing_df, mock_tests_all_pass):
    """
    Scenario: The code returns df['Age'].mean(), and all test cases pass, 
    predicted type is 'number' and the actual result type is int. 
    Expect a positive reward of 0.1 * pass_count (2 tests => 0.2).
    """
    # Create the reward function
    logging.debug(f"Prompt Item: test_all_tests_pass_correct_type: {mock_prompt_item}")
    idx = 0
    reward_fn = error_detecting_reward_fn(
        question_idx=idx,
        prompt_item=mock_prompt_item[idx],
        backing_df=mock_backing_df,
        prompt=mock_prompt_item[idx]['content'],
        tests=mock_tests_all_pass
    )

    # Code snippet that pass the test cases. 
    code = f"""{mock_prompt_item[idx]['content']} df['Age'].mean()"""
    
    # The reward function calls run_all_tests_for_answer() internally
    reward = reward_fn(code)
    logging.debug(f"Reward Received:{reward}")
    assert reward is not None, f"Reward should not be None"
    assert reward >= 0.0, f"Expected Expected positive reward"
    assert reward >= 0.5, f"Expected greather or equal to  0.5 got {reward}" 

