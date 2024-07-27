#!/bin/bash

# Define the path to virtual environments and the output file
venv_path=~/.venv
venvs=("tensorflow" "keras-tensorflow" "keras-jax" "keras-torch")
output_file=results.txt

# Remove the output file if it already exists
if [ -e "$output_file" ]; then
    rm -f "$output_file"
fi

# Set environment variables
export LD_LIBRARY_PATH=
export NVIDIA_TF32_OVERRIDE=0

# Define the models to benchmark
models=("gpt2")

# Initialize the output file with the header
printf "# Benchmarking $venv_name\n\n" | tee -a $output_file

# Set the KERAS_HOME environment variable
export KERAS_HOME=configs/keras-tensorflow

# Loop through each model and benchmark
for model_name in "${models[@]}"; do
    printf "$model_name:\n" | tee -a $output_file
    printf "fit:\n" | tee -a $output_file
    python3.11 benchmark/$model_name/fit.py $output_file
    printf "\n\n" | tee -a $output_file
done
