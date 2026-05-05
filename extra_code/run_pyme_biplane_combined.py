#!/usr/bin/env python3
"""
PYME biplane localization using SplitterFitInterpBNR.

Input options:
  1. Combined side-by-side TIFF: (frames, 64, 128) where left=BP+250, right=BP-250
  2. Separate BP1 and BP2 TIFFs: automatically combined side-by-side

EXACT settings from PYME GUI - DO NOT MODIFY

Compatible with PYME versions 21.x through 25.x
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
import argparse
import numpy as np
import tifffile
import h5py
from pathlib import Path

# PYME imports with backwards compatibility
try:
    from PYME.localization import ofind
except ImportError:
    from PYME.Analysis import ofind

try:
    from PYME.localization.FitFactories import SplitterFitInterpBNR
except ImportError:
    from PYME.Analysis.FitFactories import SplitterFitInterpBNR

try:
    from PYME.localization.splitting import split_image
except ImportError:
    # Older PYME versions
    try:
        from PYME.Analysis.splitting import split_image
    except ImportError:
        # Fallback: implement split_image manually
        def split_image(md, data):
            """Manual split_image for older PYME versions."""
            roi0 = md['Splitter.Channel0ROI']
            roi1 = md['Splitter.Channel1ROI']
            # ROI format: [x_start, y_start, x_size, y_size] or similar
            # For side-by-side: left half is channel 0, right half is channel 1
            h, w = data.shape
            half_w = w // 2
            ch0 = data[:, :half_w]
            ch1 = data[:, half_w:]
            return np.stack([ch0, ch1], axis=-1)

try:
    from PYME.IO import MetaDataHandler
except ImportError:
    from PYME.IO import MetaDataHandler as MetaDataHandler

# Get PYME version for compatibility checks
try:
    import PYME
    PYME_VERSION = getattr(PYME, '__version__', '0.0.0')
except Exception:
    PYME_VERSION = '0.0.0'


def create_combined_sidebyside(bp1_path, bp2_path):
    """
    Load separate BP1 and BP2 TIFFs and combine them side-by-side.

    Args:
        bp1_path: Path to BP+250 TIFF stack
        bp2_path: Path to BP-250 TIFF stack

    Returns:
        Combined array (frames, height, width*2) with BP1 on left, BP2 on right
    """
    print(f"Loading BP1: {bp1_path}")
    bp1 = tifffile.imread(bp1_path)
    print(f"Loading BP2: {bp2_path}")
    bp2 = tifffile.imread(bp2_path)

    # Handle single frame vs stack
    if bp1.ndim == 2:
        bp1 = bp1[np.newaxis, ...]
    if bp2.ndim == 2:
        bp2 = bp2[np.newaxis, ...]

    n_frames = bp1.shape[0]
    height = bp1.shape[1]
    width = bp1.shape[2]

    print(f"  BP1 shape: {bp1.shape}")
    print(f"  BP2 shape: {bp2.shape}")

    if bp1.shape != bp2.shape:
        raise ValueError(f"BP1 and BP2 shapes must match: {bp1.shape} vs {bp2.shape}")

    # Create combined side-by-side: BP1 on left, BP2 on right
    combined = np.zeros((n_frames, height, width * 2), dtype=bp1.dtype)
    combined[:, :, :width] = bp1
    combined[:, :, width:] = bp2

    print(f"  Combined shape: {combined.shape}")
    return combined


def create_metadata(psf_file=None):
    """Create PYME metadata with EXACT settings from PYME GUI."""
    try:
        md = MetaDataHandler.NestedClassMDHandler()
    except AttributeError:
        # Fallback for older PYME versions
        md = MetaDataHandler.MDHandlerBase()

    # Voxelsize in um
    md['voxelsize.x'] = 0.1
    md['voxelsize.y'] = 0.1

    # Camera settings - EXACT from PYME GUI
    md['Camera.TrueEMGain'] = 300.0
    md['Camera.NoiseFactor'] = 1.41
    md['Camera.ElectronsPerCount'] = 45.00
    md['Camera.ReadNoise'] = 74.4
    md['Camera.ADOffset'] = 100.0

    # Splitter ROIs - EXACT from PYME GUI for side-by-side format
    # Image is (64, 128) - height=64, width=128
    # Channel 0 (BP+250): left half
    # Channel 1 (BP-250): right half
    md['Splitter.Channel0ROI'] = [0.0, 0, 64.0, 128]
    md['Splitter.Channel1ROI'] = [0, 64, 64.0, 128]
    md['Splitter.Flip'] = False

    # Chroma shifts - EXACT from PYME GUI
    md['chroma.dx'] = '{"PYME.Analysis.points.twoColour.lin2Model": {"mx": 0, "my": 0, "x0": 0}}'
    md['chroma.dy'] = '{"PYME.Analysis.points.twoColour.lin2Model": {"mx": 0, "my": 0, "x0": 0}}'

    # Analysis settings - EXACT from PYME GUI
    md['Analysis.FitModule'] = 'SplitterFitInterpBNR'
    md['Analysis.DetectionThreshold'] = 1.0  # Threshold: 1
    md['Analysis.DebounceRadius'] = 4  # Debounce_radius: 4
    md['Analysis.ROISize'] = 5  # ROI half size (GUI default)
    md['Analysis.StartAt'] = 0  # Start_at: 0
    md['Analysis.BGRange'] = [-30, 0]  # Background_subtraction: -30
    md['Analysis.AxialShift'] = -250.0  # Z_shift: -250
    md['Analysis.EstimatorModule'] = 'biplaneEstimator'  # Z_start_estimator: biplaneEstimator
    md['Analysis.subtractBackground'] = True  # Subtract_background_in_fit: yes
    md['Analysis.PCTBackground'] = 0.0  # Use_percentile_for_background: no
    md['Analysis.FitBackground'] = True  # Fit_Background: yes

    # Channel ratio - EXACT from PYME GUI
    md['chroma.ChannelRatios'] = [0.5]  # Channel_Ratios: 0.5

    # PSF file
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


def localize_combined(combined_input, output_path, psf_file=None, max_frames=None, verbose=True):
    """
    Run PYME biplane localization on combined side-by-side data.

    Args:
        combined_input: Either a path to combined TIFF or a numpy array (frames, H, W*2)
        output_path: Path to save h5r results
        psf_file: Path to PSF file
        max_frames: Limit number of frames to process
        verbose: Print progress

    IMPORTANT: Pass full combined frame (64, 128) directly to FitFactory.
    Do NOT manually split - let PYME handle channel extraction via ROI metadata.
    """
    # Handle both path and numpy array input
    if isinstance(combined_input, (str, Path)):
        if verbose:
            print(f"Loading combined data: {combined_input}")
        combined_stack = tifffile.imread(combined_input).astype(np.float32)
    else:
        combined_stack = combined_input.astype(np.float32)

    n_frames = combined_stack.shape[0]
    if max_frames:
        n_frames = min(n_frames, max_frames)

    if verbose:
        print(f"  Shape: {combined_stack.shape}")
        print(f"  Frames to process: {n_frames}")
        if psf_file:
            print(f"  Using PSF file: {psf_file}")

    # Create metadata with EXACT PYME GUI settings
    md = create_metadata(psf_file=psf_file)

    # Camera ADOffset for correction
    ADOffset = md['Camera.ADOffset']

    if verbose:
        print(f"  Running SplitterFitInterpBNR localization...")
        print(f"  Splitter.Channel0ROI: {md['Splitter.Channel0ROI']}")
        print(f"  Splitter.Channel1ROI: {md['Splitter.Channel1ROI']}")
        print(f"  Analysis.AxialShift: {md['Analysis.AxialShift']}")

    # Results storage
    all_results = []

    # Process each frame
    for frame_idx in range(n_frames):
        if verbose and frame_idx % 100 == 0:
            print(f"  Processing frame {frame_idx}/{n_frames}...")

        # Get combined frame (64, 128)
        combined_frame = combined_stack[frame_idx]

        # Update frame index
        md['tIndex'] = frame_idx

        # Use PYME's split_image to convert 2D (64,128) to 3D (64,64,2)
        # This is what the GUI does internally
        split_frame = split_image(md, combined_frame)

        # Correct for ADOffset before detection (use channel 0 for detection)
        frame_corrected = split_frame[:, :, 0] - ADOffset

        # Calculate SNR-based threshold
        sigma = calc_sigma(frame_corrected, md)
        threshold = sigma.mean() * md['Analysis.DetectionThreshold']

        # Find candidates on channel 0
        ofd = ofind.ObjectIdentifier(frame_corrected)
        try:
            ofd.FindObjects(threshold, numThresholdSteps=0, blurRadius=1.5, debounceRadius=4)
        except TypeError:
            # Older PYME versions may have different signature
            ofd.FindObjects(threshold, blurRadius=1.5, debounceRadius=4)
        candidates = list(ofd)

        # Create fit factory with split 3D frame - PYME expects (height, width, 2)
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
                        'A': float(res['fitResults']['A']),
                        'resultCode': int(res['resultCode']),
                    }
                    all_results.append(result)
            except Exception as e:
                pass

    if verbose:
        print(f"  Total localizations: {len(all_results)}")
        if all_results:
            z_vals = [r['z'] for r in all_results]
            print(f"  Z range: {min(z_vals):.1f} to {max(z_vals):.1f} nm")

    # Save to h5r format
    save_h5r(output_path, all_results, md)

    if verbose:
        print(f"  Saved to: {output_path}")

    return all_results


def save_h5r(output_path, results, mdh):
    """Save localization results to h5r format."""
    # Get FitResultsDType with fallback for older PYME versions
    try:
        fresultdtype = SplitterFitInterpBNR.FitResultsDType
    except AttributeError:
        # Fallback dtype for older versions
        fresultdtype = np.dtype([
            ('tIndex', '<i4'),
            ('fitResults', [
                ('x0', '<f4'),
                ('y0', '<f4'),
                ('z0', '<f4'),
                ('A', '<f4'),
                ('sigma', '<f4'),
                ('background', '<f4'),
            ]),
            ('fitError', [
                ('x0', '<f4'),
                ('y0', '<f4'),
                ('z0', '<f4'),
                ('A', '<f4'),
                ('sigma', '<f4'),
                ('background', '<f4'),
            ]),
            ('resultCode', '<i4'),
            ('slicesUsed', [('x', [('start', '<i4'), ('stop', '<i4'), ('step', '<i4')]),
                           ('y', [('start', '<i4'), ('stop', '<i4'), ('step', '<i4')])])
        ])

    n_results = len(results)
    fit_results = np.zeros(n_results, dtype=fresultdtype)

    for i, r in enumerate(results):
        fit_results[i]['tIndex'] = r['frame']
        fit_results[i]['fitResults']['x0'] = r['x']
        fit_results[i]['fitResults']['y0'] = r['y']
        fit_results[i]['fitResults']['z0'] = r['z']
        fit_results[i]['fitResults']['A'] = r.get('A', 1000)
        fit_results[i]['resultCode'] = r.get('resultCode', 1)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('FitResults', data=fit_results)
        f.create_group('MetaData')


def main():
    """Run PYME biplane localization with command-line arguments."""
    parser = argparse.ArgumentParser(
        description='PYME biplane localization using SplitterFitInterpBNR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using separate BP1 and BP2 files:
  python run_pyme_biplane_combined.py --bp1 data_BP+250.tif --bp2 data_BP-250.tif -o output.h5r

  # Using pre-combined side-by-side file:
  python run_pyme_biplane_combined.py --combined combined_sidebyside.tif -o output.h5r

  # Batch mode (original behavior):
  python run_pyme_biplane_combined.py --batch
        """
    )

    # Input options (mutually exclusive groups)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--combined', '-c', type=str,
                            help='Path to combined side-by-side TIFF (frames, H, W*2)')
    input_group.add_argument('--bp1', type=str,
                            help='Path to BP1 (+250) TIFF (use with --bp2)')
    input_group.add_argument('--batch', action='store_true',
                            help='Batch mode: process all combined_{codec}_sidebyside.tif files')

    # BP2 required if BP1 is provided
    parser.add_argument('--bp2', type=str,
                        help='Path to BP2 (-250) TIFF (required with --bp1)')

    # Other options
    parser.add_argument('--output', '-o', type=str,
                        help='Output h5r file path')
    parser.add_argument('--psf', type=str, default='psf1_split.tif',
                        help='Path to PSF file (default: psf1_split.tif)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process')

    args = parser.parse_args()

    # Validate arguments
    if args.bp1 and not args.bp2:
        parser.error("--bp2 is required when using --bp1")

    # Get absolute path for PSF file
    psf_file = os.path.abspath(args.psf)
    if not os.path.exists(psf_file):
        print(f"Warning: PSF file not found: {psf_file}")

    # ========================================================================
    # BATCH MODE
    # ========================================================================
    if args.batch:
        output_dir = Path('localizations')
        output_dir.mkdir(parents=True, exist_ok=True)

        codecs = ['raw', 'sparz', 'av1', 'h264', 'x265', 'prores', 'ffv1', 'zstd']
        results_summary = {}

        for codec in codecs:
            combined_file = f'combined_{codec}_sidebyside.tif'
            if not os.path.exists(combined_file):
                print(f"\nSkipping {codec} - file not found: {combined_file}")
                continue

            print("\n" + "="*60)
            print(f"Processing {codec.upper()}")
            print("="*60)

            output_file = output_dir / f'{codec}_localizations.h5r'
            results = localize_combined(
                combined_file,
                str(output_file),
                psf_file=psf_file,
                verbose=True
            )
            results_summary[codec] = len(results)

        print("\n" + "="*60)
        print("LOCALIZATION SUMMARY")
        print("="*60)
        for codec, count in results_summary.items():
            print(f"  {codec:10s}: {count:6d} localizations")

    # ========================================================================
    # SEPARATE BP1 + BP2 MODE
    # ========================================================================
    elif args.bp1:
        print("="*60)
        print("PYME Biplane Localization")
        print("="*60)

        # Combine BP1 and BP2 side-by-side
        combined_stack = create_combined_sidebyside(args.bp1, args.bp2)

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            bp1_stem = Path(args.bp1).stem
            output_path = f'{bp1_stem}_localizations.h5r'

        print(f"\nRunning localization...")
        results = localize_combined(
            combined_stack,
            output_path,
            psf_file=psf_file,
            max_frames=args.max_frames,
            verbose=True
        )

        print(f"\nTotal localizations: {len(results)}")

    # ========================================================================
    # COMBINED FILE MODE
    # ========================================================================
    elif args.combined:
        print("="*60)
        print("PYME Biplane Localization")
        print("="*60)

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            combined_stem = Path(args.combined).stem
            output_path = f'{combined_stem}_localizations.h5r'

        results = localize_combined(
            args.combined,
            output_path,
            psf_file=psf_file,
            max_frames=args.max_frames,
            verbose=True
        )

        print(f"\nTotal localizations: {len(results)}")


if __name__ == '__main__':
    main()
