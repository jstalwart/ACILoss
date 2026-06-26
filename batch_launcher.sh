#!/bin/bash
#SBATCH --cpus-per-task=1 --gpus=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment)   experiment="$2"; shift 2 ;;
        --dataset)      dataset="$2";    shift 2 ;;
        --input_size)   input="$2";      shift 2 ;;
        --output_size)  output="$2";     shift 2 ;;
        --model)        model="$2";      shift 2 ;;
        --loss)         loss="$2";       shift 2 ;;
        --epochs)       epochs="$2";     shift 2 ;;
        *) echo "Unknown argument: $1";   exit 1 ;;
    esac
done

# Check if required arguments are provided
if [[ -z "$experiment" || -z "$dataset" || -z "$input" || -z "$output" || -z "$model" || -z "$loss" ]]; then
    echo "Usage: $0 --experiment <name> --dataset <data> --input_size <input_size> --output_size <pred_horizon> --model <model_name> --loss <ACI/None> [--epochs <epochs>]"
    exit 1
fi

# Defaults
epochs="${epochs:-200}"

# Loop for seed
for seed in 1812 2811 3002 4296 5221
do
    outputs="./slurm-${experiment}.${seed}.out"

    sbatch -J "${experiment}.${seed}" -o "$outputs" launcher.sh \
        --experiment "$experiment" \
        --seed "$seed" \
        --dataset "$dataset" \
        --input_size "$input" \
        --output_size "$output" \
        --epochs "$epochs" \
        --model "$model" \
        --loss "$loss"
done
