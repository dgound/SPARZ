"""
Grid search over SPARZ parameters for Figure 2 of the manuscript.
Varies: compression level (0-3), relative threshold (0.35-0.65), kernel size (5-11)
Collects: file size, SSIM, Jaccard index (via PYME localization)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import tifffile
import h5py

# Add SPARZ src to path
sys.path.insert(0, '/Users/dimos/Documents/GitHub/SPARZ/src')
from SPARZ import SPARZIP, UNSPARZ

# PYME imports
from PYME.localization import ofind
from PYME.localization.FitFactories import SplitterFitInterpBNR
from PYME.localization.splitting import split_image
from PYME.IO import MetaDataHandler


# ============================================================================
# CONFIGURATION
# ============================================================================

# Input data paths (tubulin biplane dataset)
PATH_BP1 = '/Users/dimos/sparz_MS/sequence-as-stack-MT0.N1.HD-BP+250.tif'
PATH_BP2 = '/Users/dimos/sparz_MS/sequence-as-stack-MT0.N1.HD-BP-250.tif'

# PSF file for PYME localization
PSF_FILE = '/Users/dimos/sparz_MS/psf1_split.tif'

# Output directory for grid search results
OUTPUT_BASE = Path('/Users/dimos/SPARZ_grid_search')
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Dataset stem name
STEM = 'tubulin'

# Grid search parameters
KERNEL_SIZES = [5, 7, 9, 11]
REL_THRESHOLDS = [0.35, 0.45, 0.55, 0.65]
COMPRESSION_LEVELS = [0, 1, 2, 3]

# Codec to use
CODEC = 'x265'

# Jaccard voxel size in nm
JACCARD_VOXEL_SIZE = 40.0


# ============================================================================
# PYME LOCALIZATION FUNCTIONS
# ============================================================================

def create_pyme_metadata(psf_file=None):
    """Create PYME metadata with EXACT settings from PYME GUI."""
    md = MetaDataHandler.NestedClassMDHandler()

    # Voxelsize in um
    md['voxelsize.x'] = 0.1
    md['voxelsize.y'] = 0.1

    # Camera settings
    md['Camera.TrueEMGain'] = 300.0
    md['Camera.NoiseFactor'] = 1.41
    md['Camera.ElectronsPerCount'] = 45.00
    md['Camera.ReadNoise'] = 74.4
    md['Camera.ADOffset'] = 100.0

    # Splitter ROIs for side-by-side format (64, 128)
    md['Splitter.Channel0ROI'] = [0.0, 0, 64.0, 128]
    md['Splitter.Channel1ROI'] = [0, 64, 64.0, 128]
    md['Splitter.Flip'] = False

    # Chroma shifts
    md['chroma.dx'] = '{"PYME.Analysis.points.twoColour.lin2Model": {"mx": 0, "my": 0, "x0": 0}}'
    md['chroma.dy'] = '{"PYME.Analysis.points.twoColour.lin2Model": {"mx": 0, "my": 0, "x0": 0}}'

    # Analysis settings
    md['Analysis.FitModule'] = 'SplitterFitInterpBNR'
    md['Analysis.DetectionThreshold'] = 1.0
    md['Analysis.DebounceRadius'] = 4
    md['Analysis.ROISize'] = 5
    md['Analysis.StartAt'] = 0
    md['Analysis.BGRange'] = [-30, 0]
    md['Analysis.AxialShift'] = -250.0
    md['Analysis.EstimatorModule'] = 'biplaneEstimator'
    md['Analysis.subtractBackground'] = True
    md['Analysis.PCTBackground'] = 0.0
    md['Analysis.FitBackground'] = True

    # Channel ratio
    md['chroma.ChannelRatios'] = [0.5]

    if psf_file:
        md['PSFFile'] = psf_file

    md['tIndex'] = 0

    return md


def calc_sigma(data, md):
    """Calculate per-pixel noise estimate using PYME's formula."""
    TrueEMGain = md['Camera.TrueEMGain']
    NoiseFactor = md['Camera.NoiseFactor']
    ElectronsPerCount = md['Camera.ElectronsPerCount']
    ReadNoise = md['Camera.ReadNoise']

    var = ReadNoise**2
    sigma = np.sqrt(var + (NoiseFactor**2) * (ElectronsPerCount * TrueEMGain * np.maximum(data, 1.0) + TrueEMGain**2)) / ElectronsPerCount
    return sigma


def run_pyme_localization(combined_stack, psf_file=None, verbose=False):
    """
    Run PYME biplane localization on combined side-by-side data.

    Args:
        combined_stack: numpy array (frames, 64, 128)
        psf_file: path to PSF file
        verbose: print progress

    Returns:
        list of localization results
    """
    n_frames = combined_stack.shape[0]

    # Create metadata
    md = create_pyme_metadata(psf_file=psf_file)
    ADOffset = md['Camera.ADOffset']

    all_results = []

    for frame_idx in range(n_frames):
        # Get combined frame (64, 128)
        combined_frame = combined_stack[frame_idx]

        # Update frame index
        md['tIndex'] = frame_idx

        # Use PYME's split_image to convert 2D (64,128) to 3D (64,64,2)
        split_frame = split_image(md, combined_frame)

        # Correct for ADOffset before detection
        frame_corrected = split_frame[:, :, 0] - ADOffset

        # Calculate SNR-based threshold
        sigma = calc_sigma(frame_corrected, md)
        threshold = sigma.mean() * md['Analysis.DetectionThreshold']

        # Find candidates on channel 0
        ofd = ofind.ObjectIdentifier(frame_corrected)
        ofd.FindObjects(threshold, numThresholdSteps=0, blurRadius=1.5, debounceRadius=4)
        candidates = list(ofd)

        # Create fit factory
        ff = SplitterFitInterpBNR.FitFactory(split_frame, md)

        # Fit each candidate
        for obj in candidates:
            try:
                res = ff.FromPoint(int(obj.x), int(obj.y))

                if res is not None and res['resultCode'] > 0:
                    result = {
                        'frame': frame_idx,
                        'x': float(res['fitResults']['x0']),
                        'y': float(res['fitResults']['y0']),
                        'z': float(res['fitResults']['z0']),
                    }
                    all_results.append(result)
            except:
                pass

    return all_results


def localizations_to_array(results):
    """Convert list of localization results to numpy array (N, 3)."""
    if not results:
        return np.zeros((0, 3))
    return np.array([[r['x'], r['y'], r['z']] for r in results])


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_tiff_stack(path):
    """Load a TIFF stack and return as numpy array."""
    with tifffile.TiffFile(path) as tif:
        return np.array([page.asarray() for page in tif.pages])


def create_combined_sidebyside(bp1_stack, bp2_stack):
    """Create combined side-by-side stack (frames, 64, 128)."""
    n_frames = bp1_stack.shape[0]
    combined = np.zeros((n_frames, 64, 128), dtype=bp1_stack.dtype)
    combined[:, :, :64] = bp1_stack
    combined[:, :, 64:] = bp2_stack
    return combined


def compute_ssim_score(original, reconstructed):
    """Compute mean SSIM across all frames."""
    ssim_values = []
    for i in range(len(original)):
        orig_frame = original[i].astype(np.float64)
        recon_frame = reconstructed[i].astype(np.float64)
        data_range = max(orig_frame.max(), recon_frame.max()) - min(orig_frame.min(), recon_frame.min())
        if data_range == 0:
            data_range = 1
        s = ssim(orig_frame, recon_frame, data_range=data_range)
        ssim_values.append(s)
    return np.mean(ssim_values), np.median(ssim_values)


def get_compressed_size(output_path):
    """Get size of compressed files (mp4 + npz)."""
    total = 0
    output_path = Path(output_path)
    for ext in ['*.mp4', '*.npz', '*.mkv']:
        for f in output_path.glob(ext):
            total += f.stat().st_size
    return total


def compute_jaccard_index(ref_locs, test_locs, voxel_size=40.0):
    """
    Compute Jaccard index using voxelized localizations.
    """
    if len(ref_locs) == 0 or len(test_locs) == 0:
        return 0.0

    # Convert to voxel coordinates
    ref_voxels = np.floor(ref_locs / voxel_size).astype(int)
    test_voxels = np.floor(test_locs / voxel_size).astype(int)

    # Shift to positive coordinates
    all_voxels = np.vstack([ref_voxels, test_voxels])
    min_coords = all_voxels.min(axis=0)
    ref_voxels = ref_voxels - min_coords
    test_voxels = test_voxels - min_coords

    # Convert to set of tuples
    ref_set = set(map(tuple, ref_voxels))
    test_set = set(map(tuple, test_voxels))

    # Jaccard index
    intersection = len(ref_set & test_set)
    union = len(ref_set | test_set)

    if union == 0:
        return 1.0

    return intersection / union


# ============================================================================
# MAIN GRID SEARCH
# ============================================================================

def run_grid_search():
    """Run the full grid search and collect metrics."""

    # Load original data
    print("Loading original data...")
    original_bp1 = load_tiff_stack(PATH_BP1)
    original_bp2 = load_tiff_stack(PATH_BP2)
    print(f"  BP1 shape: {original_bp1.shape}, BP2 shape: {original_bp2.shape}")

    # Calculate original file sizes
    original_size = os.path.getsize(PATH_BP1) + os.path.getsize(PATH_BP2)
    print(f"  Original total size: {original_size / 1e6:.2f} MB")

    # Extract base filenames
    base_filename1 = Path(PATH_BP1).stem
    base_filename2 = Path(PATH_BP2).stem

    # ========================================================================
    # STEP 1: Run localization on raw data to get reference
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: Running PYME localization on RAW data (reference)")
    print("="*70)

    raw_combined = create_combined_sidebyside(original_bp1, original_bp2)
    print(f"Combined raw shape: {raw_combined.shape}")

    print("Running localization (this may take a while)...")
    raw_results = run_pyme_localization(raw_combined.astype(np.float32), psf_file=PSF_FILE, verbose=True)
    ref_locs = localizations_to_array(raw_results)
    print(f"Reference localizations: {len(ref_locs)}")

    # Save reference localizations
    ref_loc_file = OUTPUT_BASE / 'raw_localizations.npy'
    np.save(ref_loc_file, ref_locs)
    print(f"Saved reference localizations to: {ref_loc_file}")

    # ========================================================================
    # STEP 2: Grid search over parameters
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: Running grid search")
    print("="*70)

    # Results list
    results = []

    # Total iterations
    total_iterations = len(KERNEL_SIZES) * len(REL_THRESHOLDS) * len(COMPRESSION_LEVELS)

    print(f"\nRunning grid search: {total_iterations} parameter combinations")
    print(f"  Kernel sizes: {KERNEL_SIZES}")
    print(f"  Thresholds: {REL_THRESHOLDS}")
    print(f"  Compression levels: {COMPRESSION_LEVELS}")
    print()

    with tqdm(total=total_iterations, desc="Grid Search", unit="run") as pbar:
        for kernel_size in KERNEL_SIZES:
            for rel_thresh in REL_THRESHOLDS:
                for compression_level in COMPRESSION_LEVELS:

                    # Create output directory
                    run_name = f'k{kernel_size}_rt{int(rel_thresh*100)}_lev{compression_level}'
                    output_path = OUTPUT_BASE / run_name
                    uncompressed_path = output_path / 'uncompressed'
                    output_path.mkdir(parents=True, exist_ok=True)
                    uncompressed_path.mkdir(parents=True, exist_ok=True)

                    try:
                        # ----------------------------------------------------
                        # COMPRESSION (SPARZIP)
                        # ----------------------------------------------------
                        z = SPARZIP(
                            path_image_files1=PATH_BP1,
                            stem=STEM,
                            output_path=str(output_path),
                            path_image_files2=PATH_BP2,
                            relative_threshold=rel_thresh,
                            kernel_size=kernel_size
                        )
                        z.run(codec=CODEC, compression_level=compression_level)

                        # Get compressed file sizes
                        compressed_size = get_compressed_size(output_path)

                        # ----------------------------------------------------
                        # DECOMPRESSION (UNSPARZ)
                        # ----------------------------------------------------
                        path_sparse_bp1 = output_path / f'{base_filename1}.npz'
                        path_sparse_bp2 = output_path / f'{base_filename2}.npz'
                        mp4_1 = output_path / f'{base_filename1}_compression_level_{compression_level}.mp4'
                        mp4_2 = output_path / f'{base_filename2}_compression_level_{compression_level}.mp4'

                        u = UNSPARZ(
                            path_sparse_bp1=str(path_sparse_bp1),
                            path_encoded_bp1=str(mp4_1),
                            stem=STEM,
                            output_path=str(uncompressed_path),
                            path_sparse_bp2=str(path_sparse_bp2),
                            path_encoded_bp2=str(mp4_2),
                            use_roi=True,
                            chunk_size=10
                        )
                        u.run()

                        # ----------------------------------------------------
                        # LOAD DECOMPRESSED DATA
                        # ----------------------------------------------------
                        decompressed_files = sorted(uncompressed_path.glob('*.tif'))
                        if len(decompressed_files) >= 2:
                            recon_bp1 = load_tiff_stack(decompressed_files[0])
                            recon_bp2 = load_tiff_stack(decompressed_files[1])

                            # ----------------------------------------------------
                            # COMPUTE SSIM
                            # ----------------------------------------------------
                            ssim_mean_bp1, ssim_median_bp1 = compute_ssim_score(original_bp1, recon_bp1)
                            ssim_mean_bp2, ssim_median_bp2 = compute_ssim_score(original_bp2, recon_bp2)
                            ssim_mean = (ssim_mean_bp1 + ssim_mean_bp2) / 2
                            ssim_median = (ssim_median_bp1 + ssim_median_bp2) / 2

                            # ----------------------------------------------------
                            # RUN PYME LOCALIZATION
                            # ----------------------------------------------------
                            recon_combined = create_combined_sidebyside(recon_bp1, recon_bp2)
                            test_results = run_pyme_localization(
                                recon_combined.astype(np.float32),
                                psf_file=PSF_FILE
                            )
                            test_locs = localizations_to_array(test_results)

                            # ----------------------------------------------------
                            # COMPUTE JACCARD INDEX
                            # ----------------------------------------------------
                            jaccard = compute_jaccard_index(ref_locs, test_locs, voxel_size=JACCARD_VOXEL_SIZE)
                            n_localizations = len(test_locs)

                        else:
                            ssim_mean = ssim_median = np.nan
                            jaccard = np.nan
                            n_localizations = 0

                        # File size percentage
                        file_size_pct = (compressed_size / original_size) * 100

                        # Store result
                        result = {
                            'kernel_size': kernel_size,
                            'rel_threshold': rel_thresh,
                            'compression_level': compression_level,
                            'compressed_size_bytes': compressed_size,
                            'original_size_bytes': original_size,
                            'file_size_pct': file_size_pct,
                            'ssim_mean': ssim_mean,
                            'ssim_median': ssim_median,
                            'jaccard': jaccard,
                            'n_localizations': n_localizations,
                            'run_name': run_name,
                            'output_path': str(output_path)
                        }
                        results.append(result)

                        # Update progress bar
                        pbar.set_postfix({
                            'size': f'{file_size_pct:.1f}%',
                            'ssim': f'{ssim_mean:.3f}',
                            'JI': f'{jaccard:.3f}'
                        })

                    except Exception as e:
                        print(f"\nError for {run_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        results.append({
                            'kernel_size': kernel_size,
                            'rel_threshold': rel_thresh,
                            'compression_level': compression_level,
                            'error': str(e),
                            'run_name': run_name
                        })

                    pbar.update(1)

                    # Save intermediate results after each run
                    df = pd.DataFrame(results)
                    df.to_csv(OUTPUT_BASE / 'grid_search_results.csv', index=False)

    # ========================================================================
    # SAVE FINAL RESULTS
    # ========================================================================
    df = pd.DataFrame(results)
    csv_path = OUTPUT_BASE / 'grid_search_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    json_path = OUTPUT_BASE / 'grid_search_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("GRID SEARCH SUMMARY")
    print("=" * 70)

    for level in COMPRESSION_LEVELS:
        level_df = df[df['compression_level'] == level]
        if 'ssim_mean' in level_df.columns and not level_df['ssim_mean'].isna().all():
            print(f"\nCompression Level {level}:")
            print(f"  File size: {level_df['file_size_pct'].mean():.1f}% +/- {level_df['file_size_pct'].std():.1f}%")
            print(f"  SSIM: {level_df['ssim_mean'].mean():.4f} +/- {level_df['ssim_mean'].std():.4f}")
            if 'jaccard' in level_df.columns:
                print(f"  Jaccard: {level_df['jaccard'].mean():.4f} +/- {level_df['jaccard'].std():.4f}")

    return df


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    run_grid_search()
