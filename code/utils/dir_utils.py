import logging
import os
import  re

logging.basicConfig(level=logging.DEBUG)

def find_all_output_files(directory, ends_with=".pkl"):
    """
    Find all the output files in the given directory.
    
    Args:
        directory (str): The path to the directory containing the output files.
        
    Returns:
        list: A list of paths to the output files.
    """
    output_files = []
    logging.debug(f"Checking directory: {directory}")
    for root, dirs, files in os.walk(directory):
        for file in files:
            logging.debug(f"Checking file: {file}")
            if file.endswith(ends_with):
                logging.debug(f"Found file: {file}")
                output_files.append(os.path.join(root, file))
    logging.debug(f"Found {len(output_files)} output files.")
    return output_files

def extract_id_from_path(file_path, pattern=r"output_list-(\d+)-\d+"):
    """
    Parse the problem index from the given file path.
    
    Args:
        file_path (str): The path to the file.
        
    Returns:
        int: The problem index.
    """
    match = re.search(pattern, file_path)
    if match:
        return int(match.group(1))
    else:
        return None

def collect_all_results(directory, 
                        ends_with=".parquet", 
                        pattern=r".*-(\d+)-.*"):
    """
    Collect all the results from the output files in the given directory.
    
    Args:
        directory (str): The path to the directory containing the output files.
        
    Returns:
        dict: A dictionary containing the results.
    """
    results = {}
    output_files = find_all_output_files(directory, ends_with=ends_with)
    for file_path in output_files:
        problem_index = extract_id_from_path(file_path, pattern=pattern)
        logging.debug(f"Problem index: {problem_index}")
        if problem_index is not None:
            results[problem_index] = file_path
    return results