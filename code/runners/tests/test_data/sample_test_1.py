import pandas as pd 
import numpy as np 

def dummy_data(random_seed) -> pd.DataFrame: 
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

def answer(df: pd.DataFrame): 
    return df['Age'].mean() 

def test_answer(random_seed): 
    dummy_data_df = dummy_data(random_seed) 
    result = answer(dummy_data_df) 
    expected_result = dummy_data_df['Age'].mean() 
    assert result == expected_result, f'Expected {expected_result}, but got {result}' 
    assert 18 <= result <= 60, f'Average age is not in expected age boundaries. Got: {result}' 
    assert np.isclose(result, expected_result, atol=0.01), f'Calculated average age of {result} does not match expected result {expected_result} closely enough.'