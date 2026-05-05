#!/usr/bin/env python3
"""
Compute Jaccard index and L2 density error from localization h5r files.
"""

import numpy as np
import h5py
from pathlib import Path
import pandas as pd


def load_localizations(h5r_path):
    """Load x, y, z coordinates from h5r file."""
    with h5py.File(h5r_path, 'r') as f:
        fit_results = f['FitResults']['fitResults']
        x = fit_results['x0'][:]
        y = fit_results['y0'][:]
        z = fit_results['z0'][:]
    return x, y, z


def compute_jaccard_3d(coords_ref, coords_test, voxel_size=40.0):
    """
    Compute 3D Jaccard index by voxelizing localizations.

    Args:
        coords_ref: (x, y, z) arrays for reference
        coords_test: (x, y, z) arrays for test
        voxel_size: voxel size in nm (default 40nm)

    Returns:
        Jaccard index (intersection / union)
    """
    x_ref, y_ref, z_ref = coords_ref
    x_test, y_test, z_test = coords_test

    # Get combined bounds
    x_min = min(x_ref.min(), x_test.min()) - voxel_size
    x_max = max(x_ref.max(), x_test.max()) + voxel_size
    y_min = min(y_ref.min(), y_test.min()) - voxel_size
    y_max = max(y_ref.max(), y_test.max()) + voxel_size
    z_min = min(z_ref.min(), z_test.min()) - voxel_size
    z_max = max(z_ref.max(), z_test.max()) + voxel_size

    # Create voxel grid indices
    def to_voxel_indices(x, y, z):
        ix = ((x - x_min) / voxel_size).astype(int)
        iy = ((y - y_min) / voxel_size).astype(int)
        iz = ((z - z_min) / voxel_size).astype(int)
        return set(zip(ix, iy, iz))

    voxels_ref = to_voxel_indices(x_ref, y_ref, z_ref)
    voxels_test = to_voxel_indices(x_test, y_test, z_test)

    intersection = len(voxels_ref & voxels_test)
    union = len(voxels_ref | voxels_test)

    if union == 0:
        return 0.0

    return intersection / union


def compute_l2_density_error(coords_ref, coords_test, bin_size=100.0):
    """
    Compute L2 spatial density error using 2D histogram.

    Args:
        coords_ref: (x, y, z) arrays for reference
        coords_test: (x, y, z) arrays for test
        bin_size: histogram bin size in nm

    Returns:
        Normalized L2 error
    """
    x_ref, y_ref, _ = coords_ref
    x_test, y_test, _ = coords_test

    # Get combined bounds
    x_min = min(x_ref.min(), x_test.min()) - bin_size
    x_max = max(x_ref.max(), x_test.max()) + bin_size
    y_min = min(y_ref.min(), y_test.min()) - bin_size
    y_max = max(y_ref.max(), y_test.max()) + bin_size

    # Create bins
    x_bins = np.arange(x_min, x_max + bin_size, bin_size)
    y_bins = np.arange(y_min, y_max + bin_size, bin_size)

    # Compute 2D histograms
    hist_ref, _, _ = np.histogram2d(x_ref, y_ref, bins=[x_bins, y_bins])
    hist_test, _, _ = np.histogram2d(x_test, y_test, bins=[x_bins, y_bins])

    # Normalize by total counts
    hist_ref = hist_ref / hist_ref.sum() if hist_ref.sum() > 0 else hist_ref
    hist_test = hist_test / hist_test.sum() if hist_test.sum() > 0 else hist_test

    # Compute L2 error
    l2_error = np.sqrt(np.sum((hist_ref - hist_test) ** 2))

    return l2_error


def main():
    loc_dir = Path('localizations')

    # Load reference (raw)
    print("Loading reference (raw) localizations...")
    raw_coords = load_localizations(loc_dir / 'raw_localizations.h5r')
    print(f"  Reference: {len(raw_coords[0])} localizations")

    # Codecs to compare
    codecs = ['sparz', 'h264', 'prores', 'av1', 'x265', 'ffv1', 'zstd']

    results = []

    for codec in codecs:
        h5r_file = loc_dir / f'{codec}_localizations.h5r'
        if not h5r_file.exists():
            print(f"  {codec}: file not found")
            continue

        print(f"\nProcessing {codec}...")
        test_coords = load_localizations(h5r_file)
        n_locs = len(test_coords[0])

        # Compute Jaccard
        jaccard = compute_jaccard_3d(raw_coords, test_coords, voxel_size=40.0)

        # Compute L2 density error
        l2_error = compute_l2_density_error(raw_coords, test_coords, bin_size=100.0)

        print(f"  Localizations: {n_locs}")
        print(f"  Jaccard (40nm): {jaccard:.4f}")
        print(f"  L2 density error: {l2_error:.6f}")

        results.append({
            'codec': codec,
            'n_localizations': n_locs,
            'jaccard_40nm': jaccard,
            'l2_density_error': l2_error
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('localization_metrics.csv', index=False)
    print(f"\nResults saved to localization_metrics.csv")
    print(df.to_string(index=False))

    # Also update the main compression_metrics.csv if it exists
    if Path('compression_metrics.csv').exists():
        metrics_df = pd.read_csv('compression_metrics.csv')
        for _, row in df.iterrows():
            codec = row['codec']
            mask = metrics_df['codec'] == codec
            if mask.any():
                metrics_df.loc[mask, 'jaccard_40nm'] = row['jaccard_40nm']
                metrics_df.loc[mask, 'l2_density_error'] = row['l2_density_error']
        metrics_df.to_csv('compression_metrics.csv', index=False)
        print("\nUpdated compression_metrics.csv with localization metrics")


if __name__ == '__main__':
    main()
