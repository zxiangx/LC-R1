#!/bin/bash

MODEL_PATH=Your_model_path  # replace to your own model path
SAVE_PATH=Your_save_path  # replace to your own save path

if [[ -z "$MODEL_PATH" || "$MODEL_PATH" == "Your_model_path" ]]; then
    echo "Error: MODEL_PATH is not set correctly"
    exit 1
fi
if [[ -z "$SAVE_PATH" || "$SAVE_PATH" == "Your_save_path" ]]; then
    echo "Error: SAVE_PATH is not set correctly"
    exit 1
fi

if ! nvidia-smi | grep -q "CUDA Version"; then
    echo "Error: No NVIDIA GPU detected"
    exit 1
fi

if ! command -v vllm-serve &> /dev/null; then
    echo "Error: vllm-serve not found. Please install vllm."
    exit 1
fi

if [[ ! -f "accelerate_configs/deepspeed_zero3.yaml" ]]; then
    echo "Error: DeepSpeed config file not found"
    exit 1
fi

mkdir -p "$SAVE_PATH"
mkdir -p log

# run vllm-serve
export CUDA_VISIBLE_DEVICES=3
fuser -k 8000/tcp
vllm-serve --model "$MODEL_PATH" --port 8000 > log/vllm.log 2>&1 &
VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

sleep 5
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "Warning: vLLM server may not have started. Check log/vllm.log"
fi

accelerate launch train.py \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    --save_path "$SAVE_PATH" \
    --model_path "$MODEL_PATH" \
    --lr 2e-6 \
    --num_generations 8

kill $VLLM_PID