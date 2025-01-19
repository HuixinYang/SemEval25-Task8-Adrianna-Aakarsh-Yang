import argparse

from dataloading import download_generated_test_cases

def main():
    parser = argparse.ArgumentParser(description="Download test cases for SemEval 2025 Task 8")

    parser.add_argument("output_root", type=str, help="Root directory to save the test cases")
    parser.add_argument("--split", type=str, default="dev", help="Dataset split to download (default: dev)")

    args = parser.parse_args()

    # Call the function with parsed arguments
    download_generated_test_cases(args.output_root, args.split)

if __name__ == "__main__":
    main()
