import argparse
from datasets import Dataset, DatasetDict, Features, Value
import os

# Createa set of testcases with the following features 
features = Features({
    "id": Value("string"),
    "split": Value("string"),
    "model": Value("string"),
    "content": Value("string")
    # Consider adding folling features 
    # competition - split
    # predicted_type - string
    # The test cases should be pushed as is to hugging face rather than to specify a output directory.
    # add assertion that the generated test cases matches the expected output type. 
})

# Argument Parsing Setup
def parse_args():
    parser = argparse.ArgumentParser(description="Prepare and upload Hugging Face dataset with splits.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing test cases.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the Hugging Face dataset.")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the dataset to Hugging Face Hub.")
    parser.add_argument("--repo_name", type=str, help="Hugging Face Hub repository name.")
    return parser.parse_args()

# Function to find test cases and assign splits correctly
def find_test_cases(root_dir, split):
    test_cases = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                model = "/".join(root.split("/")[-2:])
                print(f"{os.path.dirname(file)} file:{file}, root:{root}")
                print(f"found model: {model}")
                test_cases.append({
                    'id': file.replace(".py", "").replace("test_case_",""),
                    'path': os.path.join(root, file),
                    'split': split,
                    'model': model 
                })
    return test_cases

# Streaming generator for test cases (memory efficient)
def test_case_generator(test_cases):
    for details in test_cases:
        try:
            with open(details['path'], 'r') as f:
                content = f.read()
            yield {
                'id': details['id'],
                'split': details['split'],
                'model': details['model'],
                'content': content
            }
        except Exception as e:
            print(f"Error reading file {details['path']}: {e}")


def main():
    args = parse_args()

    # Prepare datasets for multiple splits
    test_cases_train = find_test_cases(os.path.join(args.root_dir, "train"), "train")
    test_cases_dev = find_test_cases(os.path.join(args.root_dir, "dev"), "dev")

    all_test_cases = {
        "train": test_cases_train,
        "dev": test_cases_dev
    }

    # Create a DatasetDict with streaming datasets
    dataset_dict = DatasetDict({
        split: Dataset.from_generator(lambda s=split: test_case_generator(all_test_cases[s]), features=features)
        for split in all_test_cases
    })

    # Save the dataset locally or push to Hugging Face Hub based on the flag
    if args.push_to_hub:
        if args.repo_name:
            dataset_dict.push_to_hub(args.repo_name)
            print(f"Dataset successfully pushed to the Hugging Face Hub: {args.repo_name}")
        else:
            print("Error: Please provide a repository name with --repo_name when using --push_to_hub.")
    else:
        dataset_dict.save_to_disk(args.output_dir)
        print(f"Dataset successfully saved to {args.output_dir}")

if __name__ == "__main__":
    # python push_test_cases.py --root_dir "/content/drive/MyDrive/TUE-WINTER-2024/CHALLENGES-CL/test_cases" --push_to_hub --repo_name "aakarsh-nair/semeval-2025-task-8-test-cases"
    main()

