#!/bin/bash
set -e   # có lỗi thì dừng ngay

export CUDA_VISIBLE_DEVICES=0

echo "=== Qwen0.5B HH ==="
bash run_all.sh 1 1 4 1

echo "=== Qwen0.5B SHP ==="
bash run_all.sh 1 2 4 1

echo "=== Qwen0.5B PKU ==="
bash run_all.sh 1 3 4 1

echo "=== TinyLLaMA 1.1B HH ==="
bash run_all.sh 2 1 4 1

echo "=== TinyLLaMA 1.1B SHP ==="
bash run_all.sh 2 2 4 1

echo "=== TinyLLaMA 1.1B PKU ==="
bash run_all.sh 2 3 4 1

echo "=== Qwen 3B HH ==="
bash run_all.sh 3 1 4 1

echo "=== Qwen 3B SHP ==="
bash run_all.sh 3 2 4 1

echo "=== Qwen 3B PKU ==="
bash run_all.sh 3 3 4 1

echo "🎉 ALL SFT JOBS FINISHED"
