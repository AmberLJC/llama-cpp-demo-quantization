#!/usr/bin/env python3
"""
Quantization Layer Placement Experiment

This script investigates how the placement of quantized layers affects
model performance (perplexity). We test different strategies:
1. Uniform quantization (baseline)
2. Keep first N layers at higher precision
3. Keep last N layers at higher precision
4. Keep middle layers at higher precision
5. Keep first and last layers at higher precision
6. Alternating patterns
7. Component-specific (attention vs FFN)
"""

import subprocess
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
import time

# Configuration
MODEL_DIR = Path("/Users/amberljc/Desktop/github-project/llama-cpp-demo/models")
RESULTS_DIR = Path("/Users/amberljc/Desktop/github-project/llama-cpp-demo/results")
BASE_MODEL = MODEL_DIR / "qwen2-0.5b-instruct-fp16.gguf"
TEST_FILE = RESULTS_DIR / "wikitext-test.txt"

NUM_LAYERS = 24  # Qwen2-0.5B has 24 transformer blocks
NUM_CHUNKS = 5   # Number of chunks for perplexity measurement (faster testing)


def run_command(cmd: List[str], timeout: int = 600) -> Tuple[str, str, int]:
    """Run a command and return stdout, stderr, returncode."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Timeout", -1


def get_file_size_mb(filepath: Path) -> float:
    """Get file size in MB."""
    return filepath.stat().st_size / (1024 * 1024)


def extract_perplexity(output: str) -> float:
    """Extract perplexity value from llama-perplexity output."""
    # Look for "Final estimate: PPL = X.XXXX"
    match = re.search(r'Final estimate: PPL = ([\d.]+)', output)
    if match:
        return float(match.group(1))
    # Fallback: look for last PPL value in output
    matches = re.findall(r'\[\d+\]([\d.]+)', output)
    if matches:
        return float(matches[-1])
    return float('inf')


def quantize_model(
    output_path: Path,
    quant_type: str = "Q4_0",
    tensor_type_args: List[str] = None
) -> bool:
    """Quantize the model with specified configuration."""
    cmd = ["llama-quantize"]

    if tensor_type_args:
        cmd.extend(tensor_type_args)

    cmd.extend([str(BASE_MODEL), str(output_path), quant_type])

    stdout, stderr, rc = run_command(cmd)

    if rc != 0:
        print(f"  Quantization failed: {stderr[:200]}")
        return False
    return True


def measure_perplexity(model_path: Path) -> float:
    """Measure perplexity of a model on the test set."""
    cmd = [
        "llama-perplexity",
        "-m", str(model_path),
        "-f", str(TEST_FILE),
        "--chunks", str(NUM_CHUNKS)
    ]

    stdout, stderr, rc = run_command(cmd, timeout=300)

    # Perplexity output goes to stderr
    combined = stdout + stderr
    return extract_perplexity(combined)


def run_experiment(
    name: str,
    tensor_type_args: List[str] = None,
    quant_type: str = "Q4_0"
) -> Dict:
    """Run a single quantization experiment."""
    print(f"\n{'='*50}")
    print(f"Experiment: {name}")
    print(f"{'='*50}")

    output_path = MODEL_DIR / f"exp_{name}.gguf"

    # Quantize
    print("  Quantizing model...")
    start_time = time.time()
    success = quantize_model(output_path, quant_type, tensor_type_args)
    quant_time = time.time() - start_time

    if not success:
        return {"name": name, "error": "Quantization failed"}

    # Get size
    size_mb = get_file_size_mb(output_path)
    print(f"  Model size: {size_mb:.2f} MB")

    # Measure perplexity
    print("  Measuring perplexity...")
    start_time = time.time()
    ppl = measure_perplexity(output_path)
    ppl_time = time.time() - start_time
    print(f"  Perplexity: {ppl:.4f}")

    # Cleanup
    output_path.unlink(missing_ok=True)

    return {
        "name": name,
        "size_mb": size_mb,
        "perplexity": ppl,
        "quant_time": quant_time,
        "ppl_time": ppl_time
    }


def generate_layer_args(layers: List[int], quant_type: str = "q8_0") -> List[str]:
    """Generate tensor-type arguments for specific layers."""
    args = []
    for layer in layers:
        args.extend(["--tensor-type", f"blk.{layer}={quant_type}"])
    return args


def main():
    print("="*60)
    print("QUANTIZATION LAYER PLACEMENT EXPERIMENT")
    print("="*60)
    print(f"Model: Qwen2-0.5B-Instruct ({NUM_LAYERS} layers)")
    print(f"Base model: {BASE_MODEL}")
    print(f"Test file: {TEST_FILE}")
    print(f"Chunks for perplexity: {NUM_CHUNKS}")
    print()

    # Check prerequisites
    if not BASE_MODEL.exists():
        print(f"ERROR: Base model not found at {BASE_MODEL}")
        return
    if not TEST_FILE.exists():
        print(f"ERROR: Test file not found at {TEST_FILE}")
        return

    results = []

    # ================================================
    # 1. Baseline: FP16 (no quantization)
    # ================================================
    print("\n" + "="*50)
    print("Baseline: FP16 (no quantization)")
    print("="*50)
    print("  Measuring perplexity...")
    fp16_ppl = measure_perplexity(BASE_MODEL)
    fp16_size = get_file_size_mb(BASE_MODEL)
    print(f"  Size: {fp16_size:.2f} MB")
    print(f"  Perplexity: {fp16_ppl:.4f}")
    results.append({
        "name": "baseline_fp16",
        "size_mb": fp16_size,
        "perplexity": fp16_ppl,
        "strategy": "No quantization"
    })

    # ================================================
    # 2. Uniform Q4_0 (all layers quantized)
    # ================================================
    result = run_experiment("uniform_q4_0")
    result["strategy"] = "All layers Q4_0"
    results.append(result)

    # ================================================
    # 3. First N layers at higher precision
    # Early layers learn basic features
    # ================================================
    for n_layers in [4, 8]:
        layers = list(range(n_layers))
        args = generate_layer_args(layers, "q8_0")
        result = run_experiment(f"first_{n_layers}_layers_q8", args)
        result["strategy"] = f"First {n_layers} layers Q8, rest Q4"
        results.append(result)

    # ================================================
    # 4. Last N layers at higher precision
    # Later layers handle high-level reasoning
    # ================================================
    for n_layers in [4, 8]:
        layers = list(range(NUM_LAYERS - n_layers, NUM_LAYERS))
        args = generate_layer_args(layers, "q8_0")
        result = run_experiment(f"last_{n_layers}_layers_q8", args)
        result["strategy"] = f"Last {n_layers} layers Q8, rest Q4"
        results.append(result)

    # ================================================
    # 5. Middle layers at higher precision
    # Middle layers do most "processing"
    # ================================================
    middle_start = (NUM_LAYERS - 8) // 2
    layers = list(range(middle_start, middle_start + 8))
    args = generate_layer_args(layers, "q8_0")
    result = run_experiment("middle_8_layers_q8", args)
    result["strategy"] = f"Middle 8 layers (L{middle_start}-{middle_start+7}) Q8, rest Q4"
    results.append(result)

    # ================================================
    # 6. First and last layers at higher precision
    # Protect input and output processing
    # ================================================
    layers = list(range(4)) + list(range(NUM_LAYERS - 4, NUM_LAYERS))
    args = generate_layer_args(layers, "q8_0")
    result = run_experiment("first_last_4_layers_q8", args)
    result["strategy"] = "First 4 + Last 4 layers Q8, middle Q4"
    results.append(result)

    # ================================================
    # 7. Alternating pattern (every other layer)
    # Even layers at higher precision
    # ================================================
    layers = list(range(0, NUM_LAYERS, 2))
    args = generate_layer_args(layers, "q8_0")
    result = run_experiment("alternating_even_q8", args)
    result["strategy"] = "Even layers Q8, odd layers Q4"
    results.append(result)

    # ================================================
    # 8. Attention-only higher precision
    # ================================================
    args = ["--tensor-type", "attn_q=q8_0", "--tensor-type", "attn_k=q8_0",
            "--tensor-type", "attn_v=q8_0", "--tensor-type", "attn_output=q8_0"]
    result = run_experiment("attention_q8", args)
    result["strategy"] = "All attention weights Q8, FFN Q4"
    results.append(result)

    # ================================================
    # 9. FFN-only higher precision
    # ================================================
    args = ["--tensor-type", "ffn_down=q8_0", "--tensor-type", "ffn_gate=q8_0",
            "--tensor-type", "ffn_up=q8_0"]
    result = run_experiment("ffn_q8", args)
    result["strategy"] = "All FFN weights Q8, attention Q4"
    results.append(result)

    # ================================================
    # Print Summary
    # ================================================
    print("\n")
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Experiment':<30} {'Size (MB)':<12} {'PPL':<12} {'PPL Δ from FP16':<15}")
    print("-"*80)

    fp16_ppl_baseline = results[0]["perplexity"]

    for r in results:
        if "error" in r:
            print(f"{r['name']:<30} {'ERROR':<12}")
        else:
            ppl_delta = r["perplexity"] - fp16_ppl_baseline
            print(f"{r['name']:<30} {r['size_mb']:<12.2f} {r['perplexity']:<12.4f} {ppl_delta:+.4f}")

    # Save results to JSON
    results_file = RESULTS_DIR / "quantization_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Analysis
    print("\n")
    print("="*80)
    print("ANALYSIS")
    print("="*80)

    # Filter out failed experiments
    valid_results = [r for r in results if "error" not in r and r["name"] != "baseline_fp16"]

    if valid_results:
        # Best quantized model (lowest perplexity)
        best = min(valid_results, key=lambda x: x["perplexity"])
        print(f"\nBest quantized configuration: {best['name']}")
        print(f"  Strategy: {best.get('strategy', 'N/A')}")
        print(f"  Perplexity: {best['perplexity']:.4f}")
        print(f"  Size: {best['size_mb']:.2f} MB")

        # Uniform baseline
        uniform = next((r for r in valid_results if r["name"] == "uniform_q4_0"), None)
        if uniform:
            improvement = uniform["perplexity"] - best["perplexity"]
            print(f"\nImprovement over uniform Q4_0: {improvement:.4f} perplexity points")

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("""
1. Early layers (first few) typically learn basic token embeddings and
   positional encodings - they may be more resilient to quantization.

2. Later layers (last few) handle high-level reasoning and output generation -
   keeping them at higher precision often helps more.

3. Attention layers are often more sensitive to quantization than FFN layers
   because they compute precise similarity scores.

4. The optimal strategy depends on the specific model architecture and task.
""")


if __name__ == "__main__":
    main()
