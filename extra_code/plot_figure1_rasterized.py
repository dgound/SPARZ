#!/usr/bin/env python3
"""
Create Figure 1 panels B-E for SPARZ manuscript with rasterized output.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from PIL import Image
import io

# Load data
df = pd.read_csv('compression_metrics.csv')

# Color scheme
COLORS = {
    'sparz': '#0E92EE',    # Blue - SPARZ near-lossless
    'h264': '#817425',     # Olive - lossy video
    'prores': '#817425',   # Olive - lossy video
    'av1': '#817425',      # Olive - lossy video
    'x265': '#817425',     # Olive - lossy video
    'ffv1': '#FB8500',     # Orange - lossless
    'zstd': '#FB8500',     # Orange - lossless
}

# Order for plotting
ORDER = ['sparz', 'h264', 'prores', 'av1', 'x265', 'ffv1', 'zstd']
LABELS = ['SPARZ', 'h264', 'ProRes', 'AV1', 'x265', 'FFV1', 'ZSTD']

# Reorder dataframe
df = df.set_index('codec').loc[ORDER].reset_index()

def create_bar_plot(ax, values, ylabel, title, ylim=None, add_separator=True):
    """Create a bar plot with color coding."""
    colors = [COLORS[c] for c in ORDER]
    x = np.arange(len(ORDER))
    
    bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if ylim:
        ax.set_ylim(ylim)
    
    # Add vertical separator between lossy and lossless
    if add_separator:
        ax.axvline(x=4.5, color='black', linestyle='-', linewidth=1)
    
    # Add group labels
    ax.text(0, ax.get_ylim()[1] * 1.02, 'Near-lossless', fontsize=8, ha='center')
    ax.text(2.5, ax.get_ylim()[1] * 1.02, 'Lossy', fontsize=8, ha='center')
    ax.text(5.5, ax.get_ylim()[1] * 1.02, 'Lossless', fontsize=8, ha='center')
    
    return bars

def save_rasterized_pdf(fig, filename, dpi=300):
    """Save figure as rasterized PDF by converting to PNG first."""
    # Save to PNG buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    
    # Open PNG and save as PDF
    img = Image.open(buf)
    img.save(filename, 'PDF', resolution=dpi)
    buf.close()
    print(f"Saved rasterized: {filename}")

# Create figure with 4 panels
fig, axes = plt.subplots(1, 4, figsize=(14, 4))

# Panel B: File Size %
create_bar_plot(axes[0], df['file_size_pct'].values, 
                'File Size (%)', 'B. Compression Ratio', ylim=(0, 100))

# Panel C: SSIM
create_bar_plot(axes[1], df['median_ssim'].values,
                'SSIM', 'C. Structural Similarity', ylim=(0.9, 1.01))

# Panel D: Jaccard Index
create_bar_plot(axes[2], df['jaccard_40nm'].values,
                'Jaccard Index', 'D. Localization Overlap (40nm)', ylim=(0, 1.05))

# Panel E: L2 Density Error  
create_bar_plot(axes[3], df['l2_density_error'].values,
                'L2 Density Error', 'E. Spatial Density Error', ylim=(0, 0.006))

plt.tight_layout()
save_rasterized_pdf(fig, 'figure1_plots/figure1_panels_B-E_rasterized.pdf')
plt.close()

# Create individual panels (rasterized)
for panel, ylabel, title, col, ylim in [
    ('B', 'File Size (%)', 'Compression Ratio', 'file_size_pct', (0, 100)),
    ('C', 'SSIM', 'Structural Similarity', 'median_ssim', (0.9, 1.01)),
    ('D', 'Jaccard Index', 'Localization Overlap (40nm)', 'jaccard_40nm', (0, 1.05)),
    ('E', 'L2 Density Error', 'Spatial Density Error', 'l2_density_error', (0, 0.006))
]:
    fig_single, ax_single = plt.subplots(figsize=(4, 4))
    create_bar_plot(ax_single, df[col].values, ylabel, title, ylim=ylim)
    plt.tight_layout()
    save_rasterized_pdf(fig_single, f'figure1_plots/panel_{panel}_{col}_rasterized.pdf')
    plt.close(fig_single)

print("\nAll rasterized PDFs created in figure1_plots/")
