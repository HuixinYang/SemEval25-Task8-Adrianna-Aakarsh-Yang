from datasets import load_dataset
from py_evaluator.test_case_runner import test_run_code
import logging
import os

 
from dataloading.semeval_load_dataset import load_phase_dataset
from py_evaluator.test_case_runner import is_predicted_type 
logging.basicConfig(level=logging.INFO)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from py_evaluator.utils import (run_all_tests_for_answer, 
                                extract_return_statement, 
                                generate_method_template, 
                                post_process)


def error_detecting_reward_fn(question_idx, prompt_item, backing_df, prompt, tests):
    """
    Creates an error checking function that assigns a reward based on the correctness of generated code.

    Parameters:
    question_idx (int): The index of the question being evaluated.
    backing_df (DataFrame): A DataFrame containing the backing data for post-processing.
    prompt (str): The prompt used to generate the code.
    tests (list): A list of test cases to run against the generated code.

    Returns:
    function: A function that takes generated code as input and returns a reward based on its correctness.
    """
    def error_check(code):
        """
        Assign a reward based on the correctness of generated code.
        """
        pass_count = run_all_tests_for_answer(question_idx, code, prompt, tests=tests) 
        logging.debug(f'ERROR CHECK: {pass_count} tests passed')
        logging.debug(f"ERROR_CHECK-Code: {code}")
        return_statement = extract_return_statement(code, prompt)
        answer_method = generate_method_template(return_statement)
        logging.debug(f"ERROR_CHECK:Extracted return statement: {answer_method}")
        result = post_process(answer_method, backing_df)
        logging.info(f"Post Process resutl: {result}")
        predicted_type = prompt_item['predicted_type']
        correct_type = is_predicted_type(result, predicted_type) 
        
        if correct_type and pass_count > 0:
            logging.info(f"PASSED A TEST! ({pass_count} times) and Correct Type")
            
        if not correct_type: 
            logging.info(f"Type mismatch detected: {predicted_type} vs {type(result)}")
            return -1

        # TODO: ADD A PENALTY FOR EXCESS TOKENS AFTER NEWLINE
        # TODO: ADD A PENALTY FOR TYPE MISMATCH 
        # TODO: ADD TRIVIAL CODE DETECTION FOR BOOLEAN RETURN STATEMENTS
        
        elif "CODE_ERROR" in str(result):
            logging.info("CODE ERROR DETECTED")
            logging.info(f"Error: {result} Code:\n{answer_method}")
            return -1
        elif pass_count > 0:
            logging.info(f"PASSED A TEST! ({pass_count} times)")
            return 0.1 * pass_count
        else:
            return 0.1
    return error_check
