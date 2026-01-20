"""
Grid search over SPARZ parameters for Figure 2.

Usage:
  python sparz_grid-search.py --bp1 /path/to/bp1.tif --bp2 /path/to/bp2.tif \
      --psf /path/to/psf.tif --output /path/to/output --sparz-src /path/to/SPARZ/src \
      --localization-script /path/to/run_pyme_biplane_combined.py
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import tifffile


# Grid search parameters
KERNEL_SIZES = [5, 7, 9, 11]
REL_THRESHOLDS = [0.35, 0.45, 0.55, 0.65]
COMPRESSION_LEVELS = [0, 1, 2, 3]
CODEC = 'x265'
JACCARD_VOXEL_SIZE = 40.0
STEM = 'tubulin'


def load_tiff_stack(path):
    with tifffile.TiffFile(path) as tif:
        return np.array([page.asarray() for page in tif.pages])


def combine_sidebyside(bp1, bp2):
    """Combine BP1 and BP2 side-by-side: (frames, H, W*2)."""
    n, h, w = bp1.shape
    combined = np.zeros((n, h, w * 2), dtype=bp1.dtype)
    combined[:, :, :w] = bp1
    combined[:, :, w:] = bp2
    return combined


def compute_ssim(original, reconstructed):
    values = []
    for i in range(len(original)):
        orig = original[i].astype(np.float64)
        recon = reconstructed[i].astype(np.float64)
        data_range = max(orig.max(), recon.max()) - min(orig.min(), recon.min())
        if data_range == 0:
            data_range = 1
        values.append(ssim(orig, recon, data_range=data_range))
    return np.mean(values), np.median(values)


def compute_jaccard(ref_locs, test_locs, voxel_size=40.0):
    if len(ref_locs) == 0 or len(test_locs) == 0:
        return 0.0
    ref_vox = np.floor(ref_locs / voxel_size).astype(int)
    test_vox = np.floor(test_locs / voxel_size).astype(int)
    all_vox = np.vstack([ref_vox, test_vox])
    min_c = all_vox.min(axis=0)
    ref_set = set(map(tuple, ref_vox - min_c))
    test_set = set(map(tuple, test_vox - min_c))
    intersection = len(ref_set & test_set)
    union = len(ref_set | test_set)
    return intersection / union if union > 0 else 1.0


def get_compressed_size(path):
    total = 0
    for ext in ['*.mp4', '*.npz', '*.mkv']:
        for f in Path(path).glob(ext):
            total += f.stat().st_size
    return total


def locs_to_array(results):
    if not results:
        return np.zeros((0, 3))
    return np.array([[r['x'], r['y'], r['z']] for r in results])


def run_grid_search(bp1_path, bp2_path, psf_path, output_dir, sparz_src, localize_fn):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    bp1 = load_tiff_stack(bp1_path)
    bp2 = load_tiff_stack(bp2_path)
    original_size = os.path.getsize(bp1_path) + os.path.getsize(bp2_path)
    print(f"  BP1: {bp1.shape}, BP2: {bp2.shape}")
    print(f"  Original size: {original_size / 1e6:.2f} MB")

    base1 = Path(bp1_path).stem
    base2 = Path(bp2_path).stem

    # Reference localization
    print("\nRunning reference localization on raw data...")
    combined_raw = combine_sidebyside(bp1, bp2)
    ref_results = localize_fn(
        combined_raw.astype(np.float32),
        str(output_dir / 'raw_localizations.h5r'),
        psf_file=psf_path,
        verbose=True
    )
    ref_locs = locs_to_array(ref_results)
    print(f"  Reference localizations: {len(ref_locs)}")

    # Import SPARZ
    sys.path.insert(0, str(sparz_src))
    from SPARZ import SPARZIP, UNSPARZ

    # Grid search
    results = []
    total = len(KERNEL_SIZES) * len(REL_THRESHOLDS) * len(COMPRESSION_LEVELS)

    print(f"\nRunning grid search ({total} combinations)...")
    pbar = tqdm(total=total)

    for kernel in KERNEL_SIZES:
        for thresh in REL_THRESHOLDS:
            for level in COMPRESSION_LEVELS:
                run_name = f'k{kernel}_rt{int(thresh*100)}_lev{level}'
                run_dir = output_dir / run_name
                uncomp_dir = run_dir / 'uncompressed'
                run_dir.mkdir(parents=True, exist_ok=True)
                uncomp_dir.mkdir(parents=True, exist_ok=True)

                try:
                    # Compress
                    z = SPARZIP(
                        path_image_files1=str(bp1_path),
                        stem=STEM,
                        output_path=str(run_dir),
                        path_image_files2=str(bp2_path),
                        relative_threshold=thresh,
                        kernel_size=kernel
                    )
                    z.run(codec=CODEC, compression_level=level)
                    comp_size = get_compressed_size(run_dir)

                    # Decompress
                    u = UNSPARZ(
                        path_sparse_bp1=str(run_dir / f'{base1}.npz'),
                        path_encoded_bp1=str(run_dir / f'{base1}_compression_level_{level}.mp4'),
                        stem=STEM,
                        output_path=str(uncomp_dir),
                        path_sparse_bp2=str(run_dir / f'{base2}.npz'),
                        path_encoded_bp2=str(run_dir / f'{base2}_compression_level_{level}.mp4'),
                        use_roi=True,
                        chunk_size=10
                    )
                    u.run()

                    # Load decompressed
                    dec_files = sorted(uncomp_dir.glob('*.tif'))
                    if len(dec_files) >= 2:
                        recon_bp1 = load_tiff_stack(dec_files[0])
                        recon_bp2 = load_tiff_stack(dec_files[1])

                        # SSIM
                        ssim_mean = (compute_ssim(bp1, recon_bp1)[0] + compute_ssim(bp2, recon_bp2)[0]) / 2

                        # Localize
                        combined_recon = combine_sidebyside(recon_bp1, recon_bp2)
                        test_results = localize_fn(
                            combined_recon.astype(np.float32),
                            str(run_dir / 'localizations.h5r'),
                            psf_file=psf_path,
                            verbose=False
                        )
                        test_locs = locs_to_array(test_results)

                        # Jaccard
                        jaccard = compute_jaccard(ref_locs, test_locs, JACCARD_VOXEL_SIZE)
                        n_locs = len(test_locs)
                    else:
                        ssim_mean = jaccard = np.nan
                        n_locs = 0

                    result = {
                        'kernel_size': kernel,
                        'rel_threshold': thresh,
                        'compression_level': level,
                        'file_size_pct': (comp_size / original_size) * 100,
                        'ssim_mean': ssim_mean,
                        'jaccard': jaccard,
                        'n_localizations': n_locs,
                        'run_name': run_name,
                    }

                except Exception as e:
                    result = {
                        'kernel_size': kernel,
                        'rel_threshold': thresh,
                        'compression_level': level,
                        'run_name': run_name,
                        'error': str(e)
                    }

                results.append(result)
                pbar.set_postfix({'run': run_name})
                pbar.update(1)

    pbar.close()

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'grid_search_results.csv', index=False)
    with open(output_dir / 'grid_search_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_dir / 'grid_search_results.csv'}")
    print(f"Saved to: {output_dir / 'grid_search_results.json'}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for level in COMPRESSION_LEVELS:
        ldf = df[df['compression_level'] == level]
        if 'ssim_mean' in ldf and not ldf['ssim_mean'].isna().all():
            print(f"Level {level}: size={ldf['file_size_pct'].mean():.1f}%, "
                  f"SSIM={ldf['ssim_mean'].mean():.4f}, "
                  f"Jaccard={ldf['jaccard'].mean():.4f}")

    return df


def main():
    parser = argparse.ArgumentParser(description='SPARZ grid search')
    parser.add_argument('--bp1', required=True, help='BP1 (+250) TIFF')
    parser.add_argument('--bp2', required=True, help='BP2 (-250) TIFF')
    parser.add_argument('--psf', required=True, help='PSF file')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--sparz-src', required=True, help='SPARZ src directory')
    parser.add_argument('--localization-script', required=True,
                        help='Path to run_pyme_biplane_combined.py')
    args = parser.parse_args()

    # Import localization function
    sys.path.insert(0, str(Path(args.localization_script).parent))
    from run_pyme_biplane_combined import localize_combined

    run_grid_search(
        args.bp1, args.bp2, args.psf,
        args.output, args.sparz_src, localize_combined
    )


if __name__ == '__main__':
    main()
