#!/usr/bin/env python3
"""Generate visualization figures for the quantization experiment."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set modern style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.edgecolor': '#cccccc',
    'grid.color': '#e0e0e0',
    'grid.linewidth': 0.8,
})

# Load results
with open('results/quantization_results.json', 'r') as f:
    results = json.load(f)

# Extract data
names = [r['name'] for r in results]
sizes = [r['size_mb'] for r in results]
perplexities = [r['perplexity'] for r in results]

# Create display names (shorter for plot)
display_names = {
    'baseline_fp16': 'FP16 Baseline',
    'uniform_q4_0': 'Uniform Q4',
    'first_4_layers_q8': 'First 4 Q8',
    'first_8_layers_q8': 'First 8 Q8',
    'last_4_layers_q8': 'Last 4 Q8',
    'last_8_layers_q8': 'Last 8 Q8',
    'middle_8_layers_q8': 'Middle 8 Q8',
    'first_last_4_layers_q8': 'First+Last 4 Q8',
    'alternating_even_q8': 'Alternating Q8',
    'attention_q8': 'Attention Q8',
    'ffn_q8': 'FFN Q8'
}

# Modern color palette (colorblind-friendly)
colors = {
    'baseline_fp16': '#10b981',       # Emerald - baseline
    'uniform_q4_0': '#ef4444',        # Red - uniform/worst
    'first_4_layers_q8': '#60a5fa',   # Light blue - early layers
    'first_8_layers_q8': '#2563eb',   # Blue - best (early)
    'last_4_layers_q8': '#fbbf24',    # Amber - late layers
    'last_8_layers_q8': '#f59e0b',    # Orange - late layers
    'middle_8_layers_q8': '#a78bfa',  # Purple - middle
    'first_last_4_layers_q8': '#34d399', # Teal - mixed
    'alternating_even_q8': '#8b5cf6', # Violet - alternating
    'attention_q8': '#f87171',        # Light red - component
    'ffn_q8': '#4ade80'               # Light green - component
}

# =============================================================================
# Figure 1: Model Size vs Perplexity Tradeoff
# =============================================================================
fig1, ax1 = plt.subplots(figsize=(12, 7.5))

# Annotation config with connecting lines
annotation_config = {
    'baseline_fp16': {'offset': (-70, 25), 'ha': 'right'},
    'first_8_layers_q8': {'offset': (-15, -40), 'ha': 'center'},
    'first_4_layers_q8': {'offset': (-70, 18), 'ha': 'right'},
    'first_last_4_layers_q8': {'offset': (18, -28), 'ha': 'left'},
    'ffn_q8': {'offset': (18, 18), 'ha': 'left'},
    'alternating_even_q8': {'offset': (-70, -12), 'ha': 'right'},
    'uniform_q4_0': {'offset': (15, 8), 'ha': 'left'},
    'last_4_layers_q8': {'offset': (18, 15), 'ha': 'left'},
    'attention_q8': {'offset': (18, -15), 'ha': 'left'},
    'middle_8_layers_q8': {'offset': (15, 8), 'ha': 'left'},
    'last_8_layers_q8': {'offset': (15, -12), 'ha': 'left'},
}

# Plot each point with modern styling
for name, size, ppl in zip(names, sizes, perplexities):
    color = colors.get(name, '#6b7280')
    is_best = name == 'first_8_layers_q8'
    is_baseline = name == 'baseline_fp16'

    marker = 's' if is_baseline else 'o'
    markersize = 220 if is_best else 140
    edge_width = 3 if is_best else 2

    ax1.scatter(size, ppl, c=color, s=markersize, marker=marker,
                edgecolors='white', linewidths=edge_width, zorder=5,
                alpha=0.95)

    # Styled annotations with connecting lines
    cfg = annotation_config.get(name, {'offset': (15, 0), 'ha': 'left'})
    fontweight = 'bold' if is_best else 'medium'
    fontsize = 12 if is_best else 11

    ax1.annotate(
        display_names.get(name, name),
        (size, ppl),
        xytext=cfg['offset'],
        textcoords='offset points',
        fontsize=fontsize,
        fontweight=fontweight,
        ha=cfg['ha'],
        color='#1f2937',
        arrowprops=dict(
            arrowstyle='-',
            color='#9ca3af',
            lw=0.8,
            shrinkA=5,
            shrinkB=5
        ) if abs(cfg['offset'][0]) > 30 or abs(cfg['offset'][1]) > 25 else None
    )

# Pareto frontier with smooth curve
pareto_points = ['baseline_fp16', 'first_8_layers_q8', 'first_4_layers_q8',
                 'alternating_even_q8', 'uniform_q4_0']
pareto_sizes = [sizes[names.index(p)] for p in pareto_points]
pareto_ppls = [perplexities[names.index(p)] for p in pareto_points]
sorted_pareto = sorted(zip(pareto_sizes, pareto_ppls))
pareto_sizes_sorted, pareto_ppls_sorted = zip(*sorted_pareto)
ax1.plot(pareto_sizes_sorted, pareto_ppls_sorted, '--', color='#9ca3af',
         linewidth=2, alpha=0.6, zorder=1)

# Optimal region highlight with gradient effect
from matplotlib.patches import FancyBboxPatch
optimal_box = FancyBboxPatch(
    (460, 12.88), 80, 0.28,
    boxstyle="round,pad=0.02,rounding_size=0.05",
    facecolor='#2563eb', alpha=0.08,
    edgecolor='#2563eb', linewidth=1.5, linestyle='--'
)
ax1.add_patch(optimal_box)
ax1.text(500, 12.82, 'Best Trade-off', ha='center', fontsize=11,
         color='#2563eb', fontweight='bold', style='italic')

# Axis labels with modern styling
ax1.set_xlabel('Model Size (MB)', fontweight='semibold', color='#374151')
ax1.set_ylabel('Perplexity (lower is better)', fontweight='semibold', color='#374151')
fig1.suptitle('Model Size vs Perplexity Trade-off', fontweight='bold',
              color='#111827', fontsize=16, y=0.98)
ax1.set_title('Quantization Strategy Comparison on Qwen2-0.5B',
              fontsize=12, color='#6b7280', pad=28)

# Clean grid
ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.6)
ax1.set_axisbelow(True)

# Spine styling
for spine in ax1.spines.values():
    spine.set_color('#d1d5db')
    spine.set_linewidth(1)

# Secondary x-axis for compression ratio
ax1_top = ax1.twiny()
compression_ticks = [1.0, 1.5, 2.0, 2.5, 3.0]
ax1_top.set_xlim(ax1.get_xlim())
ax1_top.set_xticks([948/c for c in compression_ticks])
ax1_top.set_xticklabels([f'{c}×' for c in compression_ticks])
ax1_top.set_xlabel('Compression Ratio', fontsize=12, color='#6b7280', labelpad=10)
ax1_top.tick_params(colors='#6b7280', pad=3)
for spine in ax1_top.spines.values():
    spine.set_color('#d1d5db')

# Modern legend with better grouping
legend_elements = [
    mpatches.Patch(color='#10b981', label='Baseline (FP16)'),
    mpatches.Patch(color='#2563eb', label='Early Layers Q8'),
    mpatches.Patch(color='#f59e0b', label='Late Layers Q8'),
    mpatches.Patch(color='#8b5cf6', label='Middle/Alternating Q8'),
    mpatches.Patch(color='#4ade80', label='Component-based Q8'),
    mpatches.Patch(color='#ef4444', label='Uniform Q4'),
]
legend = ax1.legend(
    handles=legend_elements,
    loc='upper right',
    frameon=True,
    framealpha=0.95,
    edgecolor='#e5e7eb',
    fancybox=True,
    shadow=False,
    borderpad=0.8
)
legend.get_frame().set_linewidth(1)

# Adjust axis limits to give room for labels
ax1.set_xlim(250, 1000)

plt.tight_layout()
plt.savefig('results/model_size_perplexity_tradeoff.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Generated: results/model_size_perplexity_tradeoff.png")

# =============================================================================
# Figure 2: Experiment Workflow Diagram
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(14, 10))
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Define box style
box_style = dict(boxstyle="round,pad=0.3", facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=2)
highlight_style = dict(boxstyle="round,pad=0.3", facecolor='#3498db', edgecolor='#2c3e50', linewidth=2)

def draw_box(ax, x, y, width, height, text, color='#ecf0f1', text_color='#2c3e50', fontsize=10):
    """Draw a rounded box with text."""
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.2",
                         facecolor=color, edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color, wrap=True)

def draw_arrow(ax, start, end, color='#2c3e50'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

# Title
ax2.text(7, 9.5, 'Quantization Experiment Workflow', ha='center', va='center',
         fontsize=16, fontweight='bold', color='#2c3e50')

# Stage 1: Input
draw_box(ax2, 0.5, 7, 3, 1.2, 'Input Model\n(Qwen2-0.5B FP16\n948 MB)', '#e8f6f3', fontsize=9)

# Stage 2: Quantization Strategies
draw_box(ax2, 5, 7, 4, 1.2, 'Quantization Strategies\n(10 configurations)', '#fdebd0', fontsize=9)

# Strategy boxes (smaller)
strategies = [
    ('First N Q8', '#aed6f1'),
    ('Last N Q8', '#f9e79f'),
    ('Middle Q8', '#d7bde2'),
    ('Alternating', '#a9dfbf'),
    ('Component', '#f5b7b1'),
]
for i, (name, color) in enumerate(strategies):
    x = 4.5 + (i % 3) * 1.5
    y = 5.2 if i < 3 else 4
    draw_box(ax2, x, y, 1.3, 0.9, name, color, fontsize=7)

# Stage 3: llama.cpp Quantize
draw_box(ax2, 10.5, 7, 3, 1.2, 'llama-quantize\n(--tensor-type)', '#d5f5e3', fontsize=9)

# Stage 4: Quantized Models
draw_box(ax2, 10.5, 4.5, 3, 1.2, 'Quantized Models\n(336-492 MB)', '#fad7a0', fontsize=9)

# Stage 5: Evaluation
draw_box(ax2, 10.5, 2, 3, 1.2, 'llama-perplexity\n(WikiText-2)', '#d4e6f1', fontsize=9)

# Stage 6: Results
draw_box(ax2, 5, 0.5, 4, 1.2, 'Results & Analysis\n(JSON + Visualizations)', '#abebc6', fontsize=9)

# Arrows
draw_arrow(ax2, (3.5, 7.6), (5, 7.6))
draw_arrow(ax2, (9, 7.6), (10.5, 7.6))
draw_arrow(ax2, (12, 7), (12, 5.7))
draw_arrow(ax2, (12, 4.5), (12, 3.2))
draw_arrow(ax2, (10.5, 2.6), (9, 1.1))

# Loop arrow from strategies to quantize
ax2.annotate('', xy=(10.5, 5), xytext=(8.5, 5),
             arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2,
                            connectionstyle='arc3,rad=0.3'))

# Add annotations
ax2.text(4.25, 7.8, '1', fontsize=12, fontweight='bold', color='#e74c3c',
         bbox=dict(boxstyle='circle', facecolor='white', edgecolor='#e74c3c'))
ax2.text(7, 8.4, '2', fontsize=12, fontweight='bold', color='#e74c3c',
         bbox=dict(boxstyle='circle', facecolor='white', edgecolor='#e74c3c'))
ax2.text(9.75, 7.8, '3', fontsize=12, fontweight='bold', color='#e74c3c',
         bbox=dict(boxstyle='circle', facecolor='white', edgecolor='#e74c3c'))
ax2.text(12.3, 6.3, '4', fontsize=12, fontweight='bold', color='#e74c3c',
         bbox=dict(boxstyle='circle', facecolor='white', edgecolor='#e74c3c'))
ax2.text(12.3, 3.8, '5', fontsize=12, fontweight='bold', color='#e74c3c',
         bbox=dict(boxstyle='circle', facecolor='white', edgecolor='#e74c3c'))
ax2.text(9.75, 1.3, '6', fontsize=12, fontweight='bold', color='#e74c3c',
         bbox=dict(boxstyle='circle', facecolor='white', edgecolor='#e74c3c'))

# Add step descriptions on the right
steps = [
    "1. Load base model (FP16)",
    "2. Select quantization strategy",
    "3. Apply layer-wise quantization",
    "4. Generate quantized GGUF",
    "5. Measure perplexity",
    "6. Aggregate results"
]
for i, step in enumerate(steps):
    ax2.text(0.5, 3.5 - i*0.5, step, fontsize=9, color='#2c3e50', va='center')

# Key findings box
findings_text = "Key Finding:\nFirst 8 layers Q8 achieves\n1.9x compression with\nonly 1.3% PPL increase"
ax2.text(2, 1.1, findings_text, fontsize=9, color='#1a5276', va='center', ha='center',
         bbox=dict(boxstyle='round', facecolor='#d6eaf8', edgecolor='#1a5276', linewidth=2))

plt.tight_layout()
plt.savefig('results/experiment_workflow.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Generated: results/experiment_workflow.png")

print("\nBoth figures have been saved to the results/ directory.")
