#!/usr/bin/env python3
"""
Create supplementary figure comparing SPARZ with traditional compression methods.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

# Output directory
OUTPUT_DIR = Path("/Users/dimos/sparz_MS/figure1_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("white")

# Original file size for percentage calculation
original_size_mb = 41.65

# Data from all methods (sorted by file size)
data = {
    'Method': [
        'SPARZ', 'bzip2', 'xz/lzma', 'ZSTD', 'gzip',
        'HDF5', 'TIFF-DEFLATE', 'PNG', 'Zarr', 'FFV1'
    ],
    'File Size (%)': [
        49.1, 56.4, 58.8, 63.1, 70.6,
        70.7, 73.6, 75.2, 77.4, 79.9
    ],
    'Type': [
        'SPARZ (Near-lossless)', 'Traditional Lossless', 'Traditional Lossless',
        'SPARZ (Lossless)', 'Traditional Lossless',
        'Traditional Lossless', 'Traditional Lossless', 'Traditional Lossless',
        'Traditional Lossless', 'SPARZ (Lossless)'
    ],
    'Lossless': [
        'Near-lossless', 'Lossless', 'Lossless', 'Lossless', 'Lossless',
        'Lossless', 'Lossless', 'Lossless', 'Lossless', 'Lossless'
    ]
}

df = pd.DataFrame(data)

# Sort by file size
df = df.sort_values('File Size (%)')

# Color palette
colors = {
    'SPARZ (Near-lossless)': '#0E92EE',  # Blue
    'SPARZ (Lossless)': '#FB8500',        # Orange
    'Traditional Lossless': '#888888'     # Gray
}

# Create the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Create horizontal bar chart
bars = ax.barh(
    df['Method'],
    df['File Size (%)'],
    color=[colors[t] for t in df['Type']],
    edgecolor='black',
    linewidth=0.5
)

# Add value labels on bars
for bar, val in zip(bars, df['File Size (%)']):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=9)

# Styling
ax.set_xlabel('File Size (% of original)', fontsize=11)
ax.set_xlim(0, 100)
ax.axvline(x=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#0E92EE', edgecolor='black', label='SPARZ (Near-lossless)'),
    Patch(facecolor='#FB8500', edgecolor='black', label='SPARZ (Lossless)'),
    Patch(facecolor='#888888', edgecolor='black', label='Traditional Lossless')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'supplement_compression_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'supplement_compression_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved supplement figure")

# Also create a table for the supplement
print("\n" + "=" * 80)
print("SUPPLEMENTARY TABLE: Compression Method Comparison")
print("=" * 80)
print(f"{'Method':<20} {'Type':<25} {'File Size (%)':<15} {'SSIM':<10} {'Jaccard':<10}")
print("-" * 80)
for _, row in df.iterrows():
    ssim = '1.0000' if row['Lossless'] == 'Lossless' else '0.9987'
    jaccard = '1.0000' if row['Lossless'] == 'Lossless' else '0.8646'
    print(f"{row['Method']:<20} {row['Type']:<25} {row['File Size (%)']:<15.1f} {ssim:<10} {jaccard:<10}")
print("=" * 80)
print("\nNote: All traditional methods are fully lossless (SSIM=1.0, Jaccard=1.0).")
print("SPARZ achieves better compression than all traditional lossless methods")
print("while maintaining near-lossless quality for SMLM analysis.")
