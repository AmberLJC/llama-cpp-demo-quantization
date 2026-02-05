#!/bin/bash

# Quantization Layer Placement Experiment
# Tests how different layer placement strategies affect model perplexity

set -e

MODEL_DIR="/Users/amberljc/Desktop/github-project/llama-cpp-demo/models"
RESULTS_DIR="/Users/amberljc/Desktop/github-project/llama-cpp-demo/results"
BASE_MODEL="$MODEL_DIR/qwen2-0.5b-instruct-fp16.gguf"
WIKITEXT_FILE="$RESULTS_DIR/wikitext-test.txt"

# Qwen2-0.5B has 24 layers (blk.0 to blk.23)
NUM_LAYERS=24

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Quantization Layer Placement Experiment"
echo "Model: Qwen2-0.5B-Instruct (24 layers)"
echo "=========================================="

# Download a small test corpus for perplexity measurement
if [ ! -f "$WIKITEXT_FILE" ]; then
    echo "Downloading wikitext test data..."
    curl -sL "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt" -o "$WIKITEXT_FILE"
fi

# Function to quantize with specific layer configuration and measure perplexity
run_experiment() {
    local name=$1
    local output_file="$MODEL_DIR/${name}.gguf"
    shift
    local tensor_args=("$@")

    echo ""
    echo "----------------------------------------"
    echo "Experiment: $name"
    echo "----------------------------------------"

    # Quantize
    echo "Quantizing..."
    if [ ${#tensor_args[@]} -eq 0 ]; then
        llama-quantize "$BASE_MODEL" "$output_file" Q4_0 2>&1 | tail -5
    else
        llama-quantize "${tensor_args[@]}" "$BASE_MODEL" "$output_file" Q4_0 2>&1 | tail -5
    fi

    # Get file size
    local size=$(ls -lh "$output_file" | awk '{print $5}')
    echo "Model size: $size"

    # Measure perplexity
    echo "Measuring perplexity..."
    local ppl_output=$(llama-perplexity -m "$output_file" -f "$WIKITEXT_FILE" -c 512 --chunks 10 2>&1 | tail -10)
    local ppl=$(echo "$ppl_output" | grep -oP 'Final estimate: PPL = \K[\d.]+' || echo "N/A")

    echo "Perplexity: $ppl"
    echo "$name,$size,$ppl" >> "$RESULTS_DIR/results.csv"

    # Cleanup to save space
    rm -f "$output_file"
}

# Initialize results file
echo "experiment,size,perplexity" > "$RESULTS_DIR/results.csv"

# ============================================
# Baseline: Full FP16 (no quantization)
# ============================================
echo ""
echo "Baseline: FP16"
echo "Measuring perplexity on FP16 model..."
fp16_ppl=$(llama-perplexity -m "$BASE_MODEL" -f "$WIKITEXT_FILE" -c 512 --chunks 10 2>&1 | tail -10)
fp16_ppl_val=$(echo "$fp16_ppl" | grep -oP 'Final estimate: PPL = \K[\d.]+' || echo "N/A")
fp16_size=$(ls -lh "$BASE_MODEL" | awk '{print $5}')
echo "FP16 Perplexity: $fp16_ppl_val"
echo "baseline_fp16,$fp16_size,$fp16_ppl_val" >> "$RESULTS_DIR/results.csv"

# ============================================
# 1. Uniform Q4_0 (all layers quantized to Q4_0)
# ============================================
run_experiment "uniform_q4_0"

# ============================================
# 2. Keep first N layers in higher precision (Q8_0)
# Strategy: Early layers often learn basic features, might be more sensitive
# ============================================
# Keep first 4 layers (16.7%) in Q8_0
tensor_args=()
for i in $(seq 0 3); do
    tensor_args+=("--tensor-type" "blk.$i=q8_0")
done
run_experiment "first_4_layers_q8" "${tensor_args[@]}"

# Keep first 8 layers (33%) in Q8_0
tensor_args=()
for i in $(seq 0 7); do
    tensor_args+=("--tensor-type" "blk.$i=q8_0")
done
run_experiment "first_8_layers_q8" "${tensor_args[@]}"

# ============================================
# 3. Keep last N layers in higher precision (Q8_0)
# Strategy: Later layers often handle high-level reasoning
# ============================================
# Keep last 4 layers in Q8_0
tensor_args=()
for i in $(seq 20 23); do
    tensor_args+=("--tensor-type" "blk.$i=q8_0")
done
run_experiment "last_4_layers_q8" "${tensor_args[@]}"

# Keep last 8 layers in Q8_0
tensor_args=()
for i in $(seq 16 23); do
    tensor_args+=("--tensor-type" "blk.$i=q8_0")
done
run_experiment "last_8_layers_q8" "${tensor_args[@]}"

# ============================================
# 4. Keep middle layers in higher precision
# Strategy: Middle layers often do most of the "processing"
# ============================================
# Keep middle 8 layers (8-15) in Q8_0
tensor_args=()
for i in $(seq 8 15); do
    tensor_args+=("--tensor-type" "blk.$i=q8_0")
done
run_experiment "middle_8_layers_q8" "${tensor_args[@]}"

# ============================================
# 5. Keep both ends in higher precision (first and last)
# Strategy: Protect input processing and output generation
# ============================================
tensor_args=()
for i in $(seq 0 3); do
    tensor_args+=("--tensor-type" "blk.$i=q8_0")
done
for i in $(seq 20 23); do
    tensor_args+=("--tensor-type" "blk.$i=q8_0")
done
run_experiment "first_last_4_layers_q8" "${tensor_args[@]}"

# ============================================
# 6. Alternating pattern (every other layer higher precision)
# ============================================
tensor_args=()
for i in $(seq 0 2 23); do
    tensor_args+=("--tensor-type" "blk.$i=q8_0")
done
run_experiment "alternating_q8" "${tensor_args[@]}"

# ============================================
# 7. Keep attention layers specifically in higher precision
# Strategy: Attention is often more sensitive to quantization
# ============================================
# Note: This keeps all attention weights (attn_q, attn_k, attn_v, attn_output) in Q8
run_experiment "attn_layers_q8" "--tensor-type" "attn=q8_0"

# ============================================
# 8. Keep FFN layers specifically in higher precision
# Strategy: FFN layers contain most parameters
# ============================================
run_experiment "ffn_layers_q8" "--tensor-type" "ffn=q8_0"

# ============================================
# Print Summary
# ============================================
echo ""
echo "=========================================="
echo "EXPERIMENT RESULTS SUMMARY"
echo "=========================================="
echo ""
cat "$RESULTS_DIR/results.csv" | column -t -s','
echo ""
echo "Results saved to: $RESULTS_DIR/results.csv"
