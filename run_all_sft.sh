#!/bin/bash
set -e

export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ===============================
# Qwen0.5B
# ===============================
CUDA_VISIBLE_DEVICES=3 bash run_all.sh 1 1 4 1 &   # hh
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 1 2 4 1 &   # shp
CUDA_VISIBLE_DEVICES=4 bash run_all.sh 1 3 4 1 &   # pku

# ===============================
# TinyLLaMA 1.1B
# ===============================
CUDA_VISIBLE_DEVICES=3 bash run_all.sh 2 1 4 1 &   # hh
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 2 2 4 1 &   # shp
CUDA_VISIBLE_DEVICES=4 bash run_all.sh 2 3 4 1 &   # pku

# ===============================
# Qwen 3B
# ===============================
CUDA_VISIBLE_DEVICES=3 bash run_all.sh 3 1 4 1 &   # hh
CUDA_VISIBLE_DEVICES=7 bash run_all.sh 3 2 4 1 &   # shp
CUDA_VISIBLE_DEVICES=4 bash run_all.sh 3 3 4 1 &   # pku

wait
echo "🎉 ALL JOBS FINISHED"
