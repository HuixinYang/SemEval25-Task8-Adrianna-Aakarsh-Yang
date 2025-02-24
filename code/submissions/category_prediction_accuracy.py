import argparse
import pandas as pd
import os

def main():
    question_ds_path = os.path.abspath("../../datasets/competition/test_qa_enhanced.csv")
    questions = pd.read_csv(question_ds_path)    

    semantics = []
    with open("answers/semantics.txt") as f:
        for line in f:
            semantics.append(line.strip())
    correct = 0
    correct_by_semantic = {}
    incorrect_by_semantic = {}

    for i in range(questions.shape[0]):
        question = questions.iloc[i]
        print(question["predicted_type"])
        semantic = semantics[i]
        if question["predicted_type"] == semantic:
            correct += 1
            correct_by_semantic[semantic] = correct_by_semantic.get(semantic, 0) + 1
        else:
            incorrect_by_semantic[semantic] = incorrect_by_semantic.get(semantic, 0) + 1 

    print(f"Total Accuracy: {correct / len(questions)}") 
    
    semantics = set(semantics)
    for semantic in semantics:
        correct_count = correct_by_semantic.get(semantic, 0)
        incorrect_count = incorrect_by_semantic.get(semantic, 0)
        total_count = correct_count + incorrect_count
        print(f"{semantic} Accuracy: {correct_count / total_count} ({correct_count}/{total_count})")
        print(f"Number of Incorrect [{semantic}]: {incorrect_count}")
        print(f"Number of  Correct [{semantic}]: {correct_count}")

if __name__== "__main__":
    main()
