#!/bin/bash
set -e

export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "==============================="
echo "🚀 START ALL DPO JOBS (FREE GPU)"
echo "==============================="

# ==================================================
# HH — variants 1, 4
# ==================================================

# qwen05b
CUDA_VISIBLE_DEVICES=3 bash run_all.sh 1 1 1 0
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 1 1 4 0

# tinyllama11b
CUDA_VISIBLE_DEVICES=3 bash run_all.sh 2 1 1 0
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 2 1 4 0

# qwen3b
CUDA_VISIBLE_DEVICES=3 bash run_all.sh 3 1 1 0
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 3 1 4 0


# ==================================================
# SHP — variants 2, 3, 4
# ==================================================

# qwen05b
CUDA_VISIBLE_DEVICES=3 bash run_all.sh 1 2 2 0
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 1 2 3 0
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 1 2 4 0

# tinyllama11b
CUDA_VISIBLE_DEVICES=3 bash run_all.sh 2 2 2 0
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 2 2 3 0
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 2 2 4 0

# qwen3b
CUDA_VISIBLE_DEVICES=3 bash run_all.sh 3 2 2 0
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 3 2 3 0
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 3 2 4 0


# ==================================================
# PKU — variants 2, 3, 4
# ==================================================

# qwen05b
CUDA_VISIBLE_DEVICES=3 bash run_all.sh 1 3 2 0
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 1 3 3 0
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 1 3 4 0

# tinyllama11b
CUDA_VISIBLE_DEVICES=3 bash run_all.sh 2 3 2 0
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 2 3 3 0
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 2 3 4 0

# qwen3b
CUDA_VISIBLE_DEVICES=3 bash run_all.sh 3 3 2 0
CUDA_VISIBLE_DEVICES=6 bash run_all.sh 3 3 3 0
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 3 3 4 0


echo "🎉 ALL DPO JOBS FINISHED"
