#!/bin/bash

salloc --partition=gpu_4 --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=03:00:00

source ./env.sh