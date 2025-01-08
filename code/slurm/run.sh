#!/bin/bash

#SBATCH --job-name=test_cases_parallel_job       # Job name
#SBATCH --output=test_cases_parallel.out         # Output file
#SBATCH --error=test_cases_parallel.err          # Error file
#SBATCH --partition=gpu_4                        # Partition name (adjust if needed)
#SBATCH --gres=gpu:1                             # Request 1 GPU
#SBATCH --ntasks=1                               # Number of tasks
#SBATCH --cpus-per-task=4                        # CPU cores per task
#SBATCH --mem=32G                                # Memory allocation
#SBATCH --time=00:15:00                         # Time limit (2 hours)

# Load Python and necessary modules
module load devel/python/3.10

# Activate a virtual environment if required (optional)
source ~/semeval-venv-01-07-2025/bin/activate 

export PYTHONPATH=$PYTHONPATH:/home/tu/tu_tu/tu_zxoxo45/challenges-cl/SemEval25-Task8-Adrianna-Aakarsh-Yang/code

# Run the Python script with Hugging Face datasets
python ../runners/run_with_test_cases_parallel.py