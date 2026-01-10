#!/bin/bash
set -e

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6

# # ===============================
# # Qwen 0.5B
# # ===============================
# bash run_all.sh 1 1 4 1   # hh
# bash run_all.sh 1 2 4 1   # shp
# bash run_all.sh 1 3 4 1   # pku

# ===============================
# TinyLLaMA 1.1B
# ===============================
bash run_all.sh 2 1 4 1   # hh
bash run_all.sh 2 2 4 1   # shp
bash run_all.sh 2 3 4 1   # pku

# # ===============================
# # Qwen 3B
# # ===============================
# bash run_all.sh 3 1 4 1   # hh
# bash run_all.sh 3 2 4 1   # shp
# bash run_all.sh 3 3 4 1   # pku

echo "🎉 ALL JOBS FINISHED"
