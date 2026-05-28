#!/bin/bash
#SBATCH --cpus-per-task=1 --gpus=1

# Recommended: stop script if a command fails or a variable is unset
set -e

source ~/miniconda3/bin/activate emb-env

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment)   experiment="$2"; shift 2 ;;
        --seed)         seed="$2";       shift 2 ;;
        --dataset)      dataset="$2";    shift 2 ;;
        --input_size)   input="$2";      shift 2 ;;
        --output_size)  output="$2";     shift 2 ;;
        --mode)         mode="$2";       shift 2 ;;
        --epochs)       epochs="$2";     shift 2 ;;
        *) echo "Unknown argument: $1";   exit 1 ;;
    esac
done

# Check if required arguments are provided
if [[ -z "$experiment" || -z "$seed" || -z "$dataset" || -z "$input" || -z "$output" || -z "$mode" ]]; then
    echo "Usage: $0 --experiment <name> --seed <val> --dataset <data> --input_size <input_size> --output_size <pred_horizon> --mode <mode> [--epochs <epochs>]"
    exit 1
fi

# Set default for epochs if not provided
epochs="${epochs:-2000}"

# Create results directory if it doesn't exist
output_dir="./results/$experiment"
log_dir="./Logs/$experiment"
model_dir="./Models/$experiment"
mkdir -p "$output_dir" "$log_dir" "$model_dir"

result="$output_dir/$seed.out"

# Execute
python launcher.py -n "$experiment" -s "$seed" -d "$dataset" -i "$input" -o "$output" --mode "$mode" -N "$epochs" > "$result"