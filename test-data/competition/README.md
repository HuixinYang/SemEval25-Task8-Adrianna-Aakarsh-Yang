# SemEval 2025 Task 8

This folder contains the files for the competition of SemEval 2025 Task 8
In here you will find:
* `test_qa.csv`: csv with two columns: question to be answered and dataset ID containing the answer.
There is *one folder for each dataset id*. Within that folder, you will find two files:
* `all.parquet`: the question asked over this dataset is associated with the `DataBench` benchmark.
* `sample.parquet`: the question asked over this dataset is associated with the `DataBench lite` benchmark.

# Submissions
Once you have your predictions, you need to upload a zip file containing two files (do not compress the folder containing them; instead, compress the two files into one). The zip file can have any name, but the files inside must be named as follows:
````
- Archive.zip
    --> predictions.txt // contains your answers for the all.parquet files, one answer per line
    --> predictions_lite.txt // contains your answers for the sample.parquet files, one answer per line
````

You can refer to [this example](https://github.com/jorses/databench_eval/blob/main/examples/stablecode.py) that demonstrates how to make a submission for the dev phase with the development set. To ensure your submission is considered in the ranking, you must click the "publish" button.

# Help

The fastest way to reach us is at jorgeosesgrijalba@gmail.com
