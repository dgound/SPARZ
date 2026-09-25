#!/usr/bin/env python
"""SPARZ Command Line Interface

Compress and decompress SMLM microscopy data using video codecs.

Usage:
    sparz "data/*.tiff" -o output/ --codec x265 --level 0
    unsparz "output/*.mp4" -o reconstructed/
"""

import argparse
import os
import sys
from pathlib import Path


def get_default_stem(input_path):
    """Extract stem from input path pattern."""
    # Handle glob patterns
    path = input_path.replace('*', '').replace('?', '')
    return Path(path).stem or 'output'


def cmd_sparz(args):
    """Run SPARZIP compression."""
    from SPARZ import SPARZIP

    stem = args.stem or get_default_stem(args.input)
    output = args.output.rstrip('/') + '/'

    # Create output directory if it doesn't exist
    os.makedirs(output, exist_ok=True)

    z = SPARZIP(
        path_image_files1=args.input,
        stem=stem,
        output_path=output,
        path_image_files2=args.bp2,
        relative_threshold=args.threshold,
        kernel_size=args.kernel,
        find_peaks=not args.no_roi,
        extract_metadata=args.metadata,
        create_single_file=args.single_file,
        num_workers=args.workers,
        align_planes=args.align,
        reflect_bp2=args.reflect,
    )
    z.run(codec=args.codec, compression_level=args.level)


def cmd_unsparz(args):
    """Run UNSPARZ decompression."""
    from SPARZ import UNSPARZ

    stem = args.stem or get_default_stem(args.video)
    output = args.output.rstrip('/') + '/'

    # Create output directory if it doesn't exist
    os.makedirs(output, exist_ok=True)

    # Auto-detect NPZ files if not provided
    npz_path = args.npz
    if npz_path is None and not args.no_roi:
        # Try to find NPZ files in same directory as video
        video_dir = str(Path(args.video).parent)
        npz_path = f"{video_dir}/*.npz"

    u = UNSPARZ(
        path_sparse_bp1=npz_path,
        path_encoded_bp1=args.video,
        stem=stem,
        output_path=output,
        path_sparse_bp2=args.npz2,
        path_encoded_bp2=args.video2,
        use_roi=not args.no_roi,
        num_workers=args.workers,
        output_format=args.format,
    )
    u.run()


def main():
    parser = argparse.ArgumentParser(
        prog='sparz',
        description='SPARZ - Compress SMLM microscopy data using video codecs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sparz compress "data/*.tiff" -o output/ --codec x265 --level 0
  sparz compress "bp1/*.tiff" --bp2 "bp2/*.tiff" -o output/ --codec ffv1
  sparz decompress "output/*.mp4" -o reconstructed/

Codecs:
  x265    HEVC/H.265 (near-lossless, best compression)
  av1     AV1 (near-lossless, slower but smaller)
  x264    H.264 (near-lossless, fast)
  ffv1    FFV1 (truly lossless, larger files)
  prores  ProRes (near-lossless, for editing)
  zstd    Zstandard (truly lossless, no video)
"""
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # SPARZ compress command
    p_compress = subparsers.add_parser(
        'compress', aliases=['c'],
        help='Compress image files to video',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p_compress.add_argument('input', help='Input image files (glob pattern, e.g., "data/*.tiff")')
    p_compress.add_argument('-o', '--output', default='./', help='Output directory (default: ./)')
    p_compress.add_argument('--stem', help='Output filename stem (default: auto from input)')
    p_compress.add_argument('--bp2', help='Second biplane input files for dual-plane data')
    p_compress.add_argument('-c', '--codec', default='x265',
                           choices=['x265', 'av1', 'x264', 'ffv1', 'prores', 'zstd'],
                           help='Compression codec (default: x265)')
    p_compress.add_argument('-l', '--level', type=int, default=0,
                           help='Compression level: 0-3 for video (0=max), 0-22 for zstd (default: 0)')
    p_compress.add_argument('--threshold', type=float, default=0.45,
                           help='Peak detection threshold (default: 0.45)')
    p_compress.add_argument('--kernel', type=int, default=9,
                           help='Kernel size for peak detection (default: 9)')
    p_compress.add_argument('--no-roi', action='store_true',
                           help='Disable ROI/peak detection')
    p_compress.add_argument('--metadata', action='store_true',
                           help='Extract and preserve TIFF metadata')
    p_compress.add_argument('--single-file', action='store_true',
                           help='Package output into single MKV file')
    p_compress.add_argument('-w', '--workers', type=int, default=4,
                           help='Number of parallel workers (default: 4)')
    p_compress.add_argument('--align', action='store_true',
                           help='Align biplane images')
    p_compress.add_argument('--reflect', action='store_true',
                           help='Reflect second biplane')
    p_compress.set_defaults(func=cmd_sparz)

    # UNSPARZ decompress command
    p_decompress = subparsers.add_parser(
        'decompress', aliases=['d', 'x'],
        help='Decompress video back to image files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p_decompress.add_argument('video', help='Encoded video files (glob pattern, e.g., "output/*.mp4")')
    p_decompress.add_argument('-o', '--output', default='./', help='Output directory (default: ./)')
    p_decompress.add_argument('--stem', help='Output filename stem (default: auto from input)')
    p_decompress.add_argument('--npz', help='NPZ sparse files (default: auto-detected)')
    p_decompress.add_argument('--npz2', help='Second biplane NPZ files')
    p_decompress.add_argument('--video2', help='Second biplane video files')
    p_decompress.add_argument('--no-roi', action='store_true',
                             help='Skip ROI patching (for lossless codecs)')
    p_decompress.add_argument('--format', default='tiff',
                             choices=['tiff', 'dat'], help='Output format (default: tiff)')
    p_decompress.add_argument('-w', '--workers', type=int, default=4,
                             help='Number of parallel workers (default: 4)')
    p_decompress.set_defaults(func=cmd_unsparz)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


def main_compress():
    """Entry point for sparz command."""
    sys.argv = ['sparz', 'compress'] + sys.argv[1:]
    main()


def main_decompress():
    """Entry point for unsparz command."""
    sys.argv = ['sparz', 'decompress'] + sys.argv[1:]
    main()


if __name__ == '__main__':
    main()
