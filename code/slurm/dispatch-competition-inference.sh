#!/bin/bash

# Dispatch Slurm jobs for processing indices in batches of 10

START_INDEX=0
END_INDEX=522
BATCH_SIZE=10

for ((i=START_INDEX; i<=END_INDEX; i+=BATCH_SIZE)); do
    JOB_START=$i
    JOB_END=$((i + BATCH_SIZE - 1))
    
    # Ensure the end index does not exceed the total range
    if (( JOB_END > END_INDEX )); then
        JOB_END=$END_INDEX
    fi

    # Create a unique Slurm job file
    JOB_FILE="run_indices_${JOB_START}_${JOB_END}.sbatch"

    cat <<EOT > $JOB_FILE
#!/bin/bash
#SBATCH --job-name=Run_${JOB_START}_${JOB_END}   # Job name
#SBATCH --partition=gpu_4                        # Partition name
#SBATCH --gres=gpu:1                             # Request 1 GPU
#SBATCH --ntasks=1                               # Number of tasks
#SBATCH --cpus-per-task=4                        # Number of CPU threads
#SBATCH --mem=32G                                # Total memory for the job
#SBATCH --time=05:00:00                          # Maximum run time (3 hours)
#SBATCH --output=logs/%x_%j.out                  # Standard output log
#SBATCH --error=logs/%x_%j.err                   # Error log file
#SBATCH --mail-type=FAIL                         # Mail on job failure
#SBATCH --mail-user=aakarsh.nair@student.uni-tuebingen.de        # Your email address for notifications

# Load necessary modules if required
module load devel/python/3.10
source ~/semeval-venv-01-07-2025/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/tu/tu_tu/tu_zxoxo45/challenges-cl/SemEval25-Task8-Adrianna-Aakarsh-Yang/code

# Run the script
python ../runners/run_with_test_cases_competition.py \
    --start-idx=${JOB_START} \
    --end-idx=${JOB_END} \
    --num-threads=3 \
    --rollouts=500 \
    --base-model=codellama/CodeLlama-7b-Python-hf

# Check exit status and log if the script fails
if [ \$? -ne 0 ]; then
    echo "Script failed for indices ${JOB_START}-${JOB_END}" >> logs/errors.log
fi
EOT

    # Submit the job
    sbatch $JOB_FILE

    # Optional: Remove job file after submission to keep the directory clean
    # rm $JOB_FILE
done
