#!/usr/bin/env python3
"""
Compare traditional compression methods for SPARZ manuscript supplement.
All methods are lossless, so SSIM=1.0, Jaccard=1.0, L2=0.
"""

import os
import gzip
import shutil
import tifffile
import numpy as np
import h5py
import zarr
from PIL import Image
from pathlib import Path

# Paths
RAW_DIR = Path("/Users/dimos/sparz_MS")
OUTPUT_DIR = RAW_DIR / "supplement_compression"
OUTPUT_DIR.mkdir(exist_ok=True)

# Original files
original_files = [
    RAW_DIR / "sequence-as-stack-MT0.N1.HD-BP+250.tif",
    RAW_DIR / "sequence-as-stack-MT0.N1.HD-BP-250.tif"
]

# Calculate original size
original_size = sum(f.stat().st_size for f in original_files)
print(f"Original TIFF size: {original_size / 1e6:.2f} MB")

# Load all frames
print("\nLoading original data...")
all_frames = []
for f in original_files:
    with tifffile.TiffFile(f) as tif:
        for page in tif.pages:
            all_frames.append(page.asarray())
all_frames = np.array(all_frames)
print(f"Data shape: {all_frames.shape}, dtype: {all_frames.dtype}")

results = []

# 1. GZIP -9 (compress raw TIFF files)
print("\n1. GZIP -9...")
gzip_size = 0
for f in original_files:
    out_file = OUTPUT_DIR / f"{f.stem}.tif.gz"
    with open(f, 'rb') as f_in:
        with gzip.open(out_file, 'wb', compresslevel=9) as f_out:
            shutil.copyfileobj(f_in, f_out)
    gzip_size += out_file.stat().st_size
print(f"   Size: {gzip_size / 1e6:.2f} MB ({gzip_size / original_size * 100:.1f}%)")
results.append(('gzip -9', gzip_size))

# 2. PNG (maximum compression, lossless)
print("\n2. PNG (maximum compression)...")
png_dir = OUTPUT_DIR / "png"
png_dir.mkdir(exist_ok=True)
png_size = 0
for i, frame in enumerate(all_frames):
    out_file = png_dir / f"frame_{i:04d}.png"
    # PNG with maximum compression
    img = Image.fromarray(frame)
    img.save(out_file, compress_level=9)
    png_size += out_file.stat().st_size
print(f"   Size: {png_size / 1e6:.2f} MB ({png_size / original_size * 100:.1f}%)")
results.append(('PNG', png_size))

# 3. DEFLATE-compressed TIFF
print("\n3. DEFLATE-compressed TIFF...")
deflate_file = OUTPUT_DIR / "combined_deflate.tif"
tifffile.imwrite(deflate_file, all_frames, compression='deflate')
deflate_size = deflate_file.stat().st_size
print(f"   Size: {deflate_size / 1e6:.2f} MB ({deflate_size / original_size * 100:.1f}%)")
results.append(('TIFF-DEFLATE', deflate_size))

# 5. Zarr (default zstd compression)
print("\n5. Zarr (default compression)...")
zarr_dir = OUTPUT_DIR / "data.zarr"
if zarr_dir.exists():
    shutil.rmtree(zarr_dir)
z = zarr.open(str(zarr_dir), mode='w', shape=all_frames.shape, dtype=all_frames.dtype,
              chunks=(10, all_frames.shape[1], all_frames.shape[2]))
z[:] = all_frames
zarr_size = sum(f.stat().st_size for f in zarr_dir.rglob('*') if f.is_file())
print(f"   Size: {zarr_size / 1e6:.2f} MB ({zarr_size / original_size * 100:.1f}%)")
results.append(('Zarr', zarr_size))

# 6. HDF5 with gzip filter
print("\n6. HDF5 (gzip filter)...")
h5_file = OUTPUT_DIR / "data.h5"
if h5_file.exists():
    h5_file.unlink()
with h5py.File(h5_file, 'w') as f:
    f.create_dataset('data', data=all_frames, compression='gzip', compression_opts=9)
h5_size = h5_file.stat().st_size
print(f"   Size: {h5_size / 1e6:.2f} MB ({h5_size / original_size * 100:.1f}%)")
results.append(('HDF5-gzip', h5_size))

# 7. BZIP2 (compress raw TIFF files)
print("\n7. BZIP2...")
import bz2
bz2_size = 0
for f in original_files:
    out_file = OUTPUT_DIR / f"{f.stem}.tif.bz2"
    with open(f, 'rb') as f_in:
        with bz2.open(out_file, 'wb', compresslevel=9) as f_out:
            shutil.copyfileobj(f_in, f_out)
    bz2_size += out_file.stat().st_size
print(f"   Size: {bz2_size / 1e6:.2f} MB ({bz2_size / original_size * 100:.1f}%)")
results.append(('bzip2', bz2_size))

# 8. XZ/LZMA (compress raw TIFF files)
print("\n8. XZ/LZMA...")
import lzma
xz_size = 0
for f in original_files:
    out_file = OUTPUT_DIR / f"{f.stem}.tif.xz"
    with open(f, 'rb') as f_in:
        with lzma.open(out_file, 'wb', preset=9) as f_out:
            shutil.copyfileobj(f_in, f_out)
    xz_size += out_file.stat().st_size
print(f"   Size: {xz_size / 1e6:.2f} MB ({xz_size / original_size * 100:.1f}%)")
results.append(('xz/lzma', xz_size))

# Print summary table
print("\n" + "=" * 70)
print("SUPPLEMENT: TRADITIONAL LOSSLESS COMPRESSION METHODS")
print("=" * 70)
print(f"{'Method':<15} {'Size (MB)':>12} {'File Size %':>12} {'SSIM':>8} {'Jaccard':>8} {'L2':>8}")
print("-" * 70)

# Sort by file size percentage
results_sorted = sorted(results, key=lambda x: x[1])

for method, size in results_sorted:
    pct = size / original_size * 100
    print(f"{method:<15} {size/1e6:>12.2f} {pct:>12.1f} {'1.0000':>8} {'1.0000':>8} {'0.0000':>8}")

print("-" * 70)
print(f"{'Original TIFF':<15} {original_size/1e6:>12.2f} {'100.0':>12} {'1.0000':>8} {'1.0000':>8} {'0.0000':>8}")
print("=" * 70)

# Also add SPARZ codecs for comparison
print("\n\nFor reference, SPARZ manuscript codecs:")
print("-" * 70)
sparz_results = [
    ('ZSTD (SPARZ)', 63.1),
    ('FFV1 (SPARZ)', 79.9),
    ('SPARZ', 49.1),
]
for method, pct in sparz_results:
    size_mb = pct / 100 * original_size / 1e6
    print(f"{method:<15} {size_mb:>12.2f} {pct:>12.1f}")
print("=" * 70)

# Save to CSV
import pandas as pd
df = pd.DataFrame([
    {'Method': method, 'Size_MB': size/1e6, 'File_Size_Pct': size/original_size*100,
     'SSIM': 1.0, 'Jaccard': 1.0, 'L2_Error': 0.0}
    for method, size in results_sorted
])
df.to_csv(OUTPUT_DIR / 'traditional_compression_metrics.csv', index=False)
print(f"\nSaved to: {OUTPUT_DIR / 'traditional_compression_metrics.csv'}")
