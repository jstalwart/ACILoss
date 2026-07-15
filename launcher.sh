#!/bin/bash
#SBATCH --cpus-per-task=1 --gpus=1

# Recommended: stop script if a command fails or a variable is unset
set -e

source ~/miniconda3/bin/activate ACI-env

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment)   experiment="$2"; shift 2 ;;
        --seed)         seed="$2";       shift 2 ;;
        --dataset)      dataset="$2";    shift 2 ;;
        --input_size)   input="$2";      shift 2 ;;
        --output_size)  output="$2";     shift 2 ;;
        --model)        model="$2";      shift 2 ;;
        --loss)         loss="$2";       shift 2 ;;
        --criteria)     criteria="$2";   shift 2 ;;
        --epochs)       epochs="$2";     shift 2 ;;
        --lr)           lr="$2";         shift 2 ;;
        *) echo "Unknown argument: $1";   exit 1 ;;
    esac
done

# Check if required arguments are provided
if [[ -z "$experiment" || -z "$seed" || -z "$dataset" || -z "$input" || -z "$output" || -z "$model" || -z "$loss" ]]; then
    echo "Usage: $0 --experiment <name> --seed <val> --dataset <data> --input_size <input_size> --output_size <pred_horizon> --model <model_name> --loss <ACI/SDM/None> [--epochs <epochs> --criteria <MSE/NLL> --lr <learning_rate>]"
    exit 1
fi

# Set default for epochs if not provided
epochs="${epochs:-2000}"
criteria="${criteria:-MSE}"
lr="${lr:-0.001}"


# Create results directory if it doesn't exist
output_dir="./results/$experiment"
log_dir="./Logs/$experiment"
model_dir="./Models/$experiment"
mkdir -p "$output_dir" "$log_dir" "$model_dir"

result="$output_dir/$seed.out"

# Execute
python launcher.py -n "$experiment" -s "$seed" -d "$dataset" -i "$input" -o "$output" --model "$model" --loss1 "$loss" --loss2 "$criteria" -N "$epochs" -lr $lr > "$result"