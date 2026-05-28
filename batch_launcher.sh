#!/bin/bash
#SBATCH --cpus-per-task=1 --gpus=1

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
if [[ -z "$experiment" || -z "$dataset" || -z "$horizon" ]]; then
    echo "Usage: $0 --experiment <experiment_name> --dataset <dataset> --horizon <horizon> --mode <mode> [--epochs <epochs> --batch <batch> --task <task>]"
    exit 1
fi

# Defaults
epochs="${epochs:-200}"
batch="${batch:-32}"
task="${task:-reconstruction}"
enc_path="./Models/${enc_path:-}"

# Loop for seed
for seed in 1812 2811 3002 4296 5221
do
    outputs="./slurm-${experiment}.${seed}.out"

    if [[ "$mode" == "LLE" ]]; then
        sbatch -J "${experiment}.${seed}" -o "$outputs" --cpus-per-task=4 --mem 64G launcher.sh \
            --experiment "$experiment" \
            --seed "$seed" \
            --dataset "$dataset" \
            --horizon "$horizon" \
            --mode "$mode" \
            --epochs "$epochs" \
            --batch "$batch" \
            --task "$task" \
            --encoder "$enc_path/encoder_$seed.pt"
    else
        sbatch -J "${experiment}.${seed}" -o "$outputs" launcher.sh \
            --experiment "$experiment" \
            --seed "$seed" \
            --dataset "$dataset" \
            --horizon "$horizon" \
            --mode "$mode" \
            --epochs "$epochs" \
            --batch "$batch" \
            --task "$task" \
            --encoder "$enc_path/encoder_$seed.pt"
    fi
    
done
