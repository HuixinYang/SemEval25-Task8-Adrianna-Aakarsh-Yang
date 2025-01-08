import os
from pathlib import Path
from datasets import load_dataset
import argparse

def download_test_cases(output_root, split="dev"):
    # Loading the dataset
    test_cases = load_dataset("aakarsh-nair/semeval-2025-task-8-test-cases", split=split)

    # Loop through each test case
    for idx, test_case in enumerate(test_cases):
        model = test_case['model']
        
        # Create output directory using the model name
        output_dir = os.path.join(output_root, "test_cases", split, *model.split("/"))
        
        # Ensure the directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Write the test case content to a file
        with open(f"{output_dir}/test_case_{idx}.py", "w") as f:
            f.write(test_case['content'])

def main():
    parser = argparse.ArgumentParser(description="Download test cases for SemEval 2025 Task 8")
    parser.add_argument("output_root", type=str, help="Root directory to save the test cases")
    parser.add_argument("--split", type=str, default="dev", help="Dataset split to download (default: dev)")

    args = parser.parse_args()

    # Call the function with parsed arguments
    download_test_cases(args.output_root, args.split)

if __name__ == "__main__":
    main()
