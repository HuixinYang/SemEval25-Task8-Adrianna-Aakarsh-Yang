#!/bin/bash

salloc --partition=gpu_4 --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=01:00:00

module load devel/python/3.10
source ~/semeval-venv-01-07-2025/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/tu/tu_tu/tu_zxoxo45/challenges-cl/SemEval25-Task8-Adrianna-Aakarsh-Yang/code