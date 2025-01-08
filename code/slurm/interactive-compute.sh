#!/bin/bash

salloc --partition=single --ntasks=1 --cpus-per-task=4 --mem=32G --time=02:00:00

module load devel/python/3.10
source ~/semeval-venv-01-07-2025/bin/activate