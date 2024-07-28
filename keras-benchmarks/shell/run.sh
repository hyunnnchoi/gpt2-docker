#!/bin/bash

venv_path=~/.venv
venvs=(
    "keras-tensorflow"
)
output_file=results.txt

if [ -e "$output_file" ]; then
    rm -f "$output_file"
fi

export LD_LIBRARY_PATH=
export NVIDIA_TF32_OVERRIDE=0

models=(
    #"bert"
    #"sam"
    #"stable_diffusion"
    #"gemma"
    #"mistral"
    "gpt2"
)

printf "# Benchmarking $venv_name\n\n" | tee -a $output_file

export KERAS_HOME=configs/keras-tensorflow

for model_name in "${models[@]}"; do
printf "$model_name:\n" | tee -a $output_file
printf "fit:\n" | tee -a $output_file
python3.11 /workspace/keras-benchmarks/benchmark/$model_name/fit.py $output_file
printf "\n\n" | tee -a $output_file
done
