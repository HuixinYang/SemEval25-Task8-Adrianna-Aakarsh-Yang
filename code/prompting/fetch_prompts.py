import os
from pathlib import Path
from datasets import load_dataset
import argparse

def download(output_root, split="dev"):
    # Loading the dataset
    prompts = load_dataset("aakarsh-nair/semeval-2025-task-8-prompts", split=split)

    # Loop through each test case
    for idx, prompt in enumerate(prompts):
        prompt_id = prompt['id'] 
        # Create output directory using the model name
        output_dir = os.path.join(output_root, f"{split}-prompts")
        
        # Ensure the directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Write the prompt case content to a file
        with open(f"{output_dir}/prompt_{prompt_id}.py", "w") as f:
            f.write(prompt['content'])

def main():
    parser = argparse.ArgumentParser(description="Download prompts for SemEval 2025 Task 8")
    parser.add_argument("output_root", type=str, help="Root directory to save the test cases")
    parser.add_argument("--split", type=str, default="dev", help="Dataset split to download (default: dev)")

    args = parser.parse_args()

    # Call the function with parsed arguments
    download(args.output_root, args.split)

if __name__ == "__main__":
    main()
