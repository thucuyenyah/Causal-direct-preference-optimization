#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=120:00:00
#SBATCH --mem=128G
#SBATCH --qos=priority
#SBATCH --mail-type=END,TIME_LIMIT
#SBATCH --mail-user=thin.nguyen@deakin.edu.au


# --- Environment setup ---
module load Anaconda3
source activate
conda activate dpoenv
export PYTHONNOUSERSITE=1
export WANDB_API_KEY=8f17474bb5e6fbb39a20e2e78dac373f97f339e6

# --- Input arguments ---
model_name_idx=$1   # ("qwen05b" "tinyllama11b" "qwen3b") : qwen 0.5 B, tinyllama 1.1B, and qwen 3B
dataset_idx=$2      # 1=hh, 2=shp, 3=pku
variant=$3           
run_sft=$4          # 1=true, 0=false
batch_size=4



if [ -z "$variant" ] || [ -z "$dataset_idx" ]; then
    echo "❌ Usage: sbatch run_2steps.sh <variant: 1=dpo, 2=...> <dataset_idx: 1=hh, 2=shp>"
    exit 1
fi

# --- Model config (fixed) ---
model_names=("qwen05b" "tinyllama11b" "qwen3b")
model_name="${model_names[$((model_name_idx - 1))]}"
if [ ! -f "config/model/${model_name}.yaml" ]; then
    echo "❌ Model config not found: config/model/${model_name}.yaml"
    exit 1
fi

# --- Dataset mapping ---
datasets=("hh" "shp" "pku")
if ((dataset_idx < 1 || dataset_idx > 3)); then
    echo "❌ Invalid dataset index: $dataset_idx (must be 1-3)"
    exit 1
fi
dataset="${datasets[$((dataset_idx - 1))]}"


# --- Loss variant ---
case "$variant" in
    1) variant_name="dpo2samples"; num_samples="backdoor.n_mc_samples=2" ;backdoor_mode="backdoor.enabled=true";;
    2) variant_name="dpo4samples"; num_samples="backdoor.n_mc_samples=4" ;backdoor_mode="backdoor.enabled=true";;
    3) variant_name="dpo6samples"; num_samples="backdoor.n_mc_samples=6" ;backdoor_mode="backdoor.enabled=true";;
    4) variant_name="originaldpo"; num_samples=""; backdoor_mode="backdoor.enabled=false";;
    5) variant_name="dpo1sample"; num_samples="backdoor.n_mc_samples=1" ;backdoor_mode="backdoor.enabled=true";;

esac



# --- Logging ---
log_dir="logs"
mkdir -p "$log_dir"
log_file="${log_dir}/${model_name}_${dataset}_${variant_name}_${SLURM_JOB_ID}_${model_name_idx}_${dataset_idx}_${variant}_${run_sft}.out"
err_file="${log_dir}/${model_name}_${dataset}_${variant_name}_${SLURM_JOB_ID}_${model_name_idx}_${dataset_idx}_${variant}_${run_sft}.err"
exec > "$log_file" 2> "$err_file"

echo "🚀 Starting $variant_name on dataset=$dataset with model=$model_name"
echo "-------------------------------------------------------------"



# --- SFT (only if requested) ---
if [ "$run_sft" = "1" ]; then
    echo "🔧 Running SFT..."
    python -u train.py \
        model=$model_name \
        datasets=[$dataset] \
        loss=sft \
        $backdoor_mode \
        $num_samples \
        exp_name=${dataset}_${model_name}_${variant_name}_sft \
        gradient_accumulation_steps=2 \
        batch_size=$batch_size \
        eval_batch_size=$batch_size \
        trainer=BasicTrainer \
        sample_during_eval=false 
    echo "✅ Finished SFT. Exiting..."
    exit 0
fi

# --- Find the latest SFT checkpoint ---
BASE_DIR=".cache/thinng"
PREFIX="${dataset}_${model_name}_${variant_name}_sft"

latest_suffix=$(find "$BASE_DIR" -maxdepth 1 -type d -name "${PREFIX}*" | \
  sed -E "s|.*/${PREFIX}||" | \
  sort | \
  tail -n 1)

if [ -n "$latest_suffix" ]; then
  ckpt_path="$BASE_DIR/${PREFIX}${latest_suffix}/LATEST/policy.pt"
  echo "✅ Found latest checkpoint: $ckpt_path"
else
  echo "❌ No matching checkpoints found."
  exit 1
fi

#----Causal DPO ---
echo "🔥 Running $variant_name..."
python -u train.py \
    model=$model_name \
    datasets=[$dataset] \
    loss=dpo \
    loss.beta=0.1 \
    $backdoor_mode \
    $num_samples \
    exp_name=${dataset}_${model_name}_${variant_name} \
    gradient_accumulation_steps=2 \
    batch_size=$batch_size \
    eval_batch_size=$batch_size \
    trainer=BasicTrainer \
    sample_during_eval=false \
    model.archive=$ckpt_path

echo "✅ Done: $variant_name on $dataset"