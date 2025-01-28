#TODO: This file is not used mark for DELETION
import os
from pathlib import Path
from datasets import load_dataset
import argparse
from dataloading.prompts import download_task_prompts

def main():
    parser = argparse.ArgumentParser(description="Download prompts for SemEval 2025 Task 8")
    parser.add_argument("output_root", type=str, help="Root directory to save the test cases")
    parser.add_argument("--split", type=str, default="dev", help="Dataset split to download (default: dev)")

    args = parser.parse_args()

    # Call the function with parsed arguments
    download_task_prompts(args.output_root, args.split)

if __name__ == "__main__":
    main()
