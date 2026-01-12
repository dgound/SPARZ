# SPARZ

Compress Single Molecule Localization Microscopy (SMLM) data using video codecs.

SPARZ achieves high compression ratios by encoding microscopy timelapses as video, with optional lossless storage of peak intensities for near-lossless reconstruction.

## Installation

```bash
pip install .
```

Or for development:
```bash
pip install -e .
```

### Dependencies

FFmpeg is required for video encoding:

- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`
- **Windows**: Download from https://ffmpeg.org/download.html

## Quick Start

### Command Line

```bash
# Compress
sparz "data/*.tiff" -o output/ --codec x265 --level 0

# Decompress
unsparz "output/*.mp4" -o reconstructed/
```

### GUI

```bash
sparz-gui
```

### Python API

```python
from SPARZ import SPARZIP, UNSPARZ

# Compress
z = SPARZIP(
    path_image_files1="data/*.tiff",
    stem="experiment",
    output_path="output/"
)
z.run(codec='x265', compression_level=0)

# Decompress
u = UNSPARZ(
    path_sparse_bp1="output/*.npz",
    path_encoded_bp1="output/*.mp4",
    stem="reconstructed",
    output_path="output/"
)
u.run()
```

## Codecs

| Codec | Type | Output | Description |
|-------|------|--------|-------------|
| `x265` | Near-lossless | .mp4 | HEVC/H.265, best compression (default) |
| `av1` | Near-lossless | .mp4 | AV1, smaller but slower |
| `x264` | Near-lossless | .mp4 | H.264, fast encoding |
| `ffv1` | Lossless | .avi | FFV1, truly lossless |
| `prores` | Near-lossless | .mov | ProRes, for video editing |
| `zstd` | Lossless | .zst | Zstandard, no video output |

## Compression Levels

For video codecs (x265, av1, x264, ffv1, prores):
- `0` - Maximum compression (slowest)
- `1` - High compression
- `2` - Balanced
- `3` - Fast (least compression)

For zstd: 0-22 (higher = more compression)

## CLI Reference

### sparz (compress)

```
sparz INPUT [OPTIONS]

Arguments:
  INPUT                 Input files (glob pattern, e.g., "data/*.tiff")

Options:
  -o, --output DIR      Output directory (default: ./)
  -c, --codec CODEC     Codec: x265, av1, x264, ffv1, prores, zstd
  -l, --level N         Compression level
  --bp2 FILES           Second biplane input files
  --no-roi              Disable ROI/peak detection
  --metadata            Extract TIFF metadata
  --single-file         Package into single MKV
  -w, --workers N       Parallel workers (default: 4)
```

### unsparz (decompress)

```
unsparz VIDEO [OPTIONS]

Arguments:
  VIDEO                 Video files (glob pattern)

Options:
  -o, --output DIR      Output directory (default: ./)
  --npz FILES           NPZ sparse files (auto-detected)
  --no-roi              Skip ROI patching (for lossless)
  --format FMT          Output format: tiff, dat
  -w, --workers N       Parallel workers (default: 4)
```

## Biplane Data

For dual-plane microscopy data:

```bash
sparz "bp1/*.tiff" --bp2 "bp2/*.tiff" -o output/
```

```python
z = SPARZIP(
    path_image_files1="bp1/*.tiff",
    path_image_files2="bp2/*.tiff",
    stem="biplane",
    output_path="output/"
)
z.run()
```

## License

MIT
