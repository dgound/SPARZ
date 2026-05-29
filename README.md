# SPARZ

SPARZ compresses Single Molecule Localization Microscopy (SMLM) image stacks by
encoding the dense background as video and, when ROI detection is enabled,
storing peak pixels in sparse sidecars for reconstruction.

The current pipeline supports TIFF and DAT inputs, single-plane and biplane
data, optional TIFF metadata preservation, optional exact residual sidecars,
and optional per-video MKV packaging.

## Installation

```bash
pip install .
```

For development:

```bash
pip install -e .
```

FFmpeg and ffprobe must be available on `PATH`.

- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`
- Windows: install from https://ffmpeg.org/download.html

## Quick Start

### Command Line

The installed console scripts are `sparz` for compression and `unsparz` for
reconstruction.

```bash
# Compress TIFF stacks with x265 level 0.
sparz "data/*.tiff" -o output/ --codec x265 --level 0

# Reconstruct from separate video + NPZ sidecars.
unsparz "output/*.mp4" --npz "output/*.npz" -o reconstructed/

# Reconstruct from single-file MKV packages.
unsparz "output/*.mkv" -o reconstructed/
```

### Python API

```python
from SPARZ import SPARZIP, UNSPARZ

z = SPARZIP(
    path_image_files1="data/*.tiff",
    stem="experiment",
    output_path="output/",
    extract_metadata=True,
    save_residuals=True,
)
z.run(codec="x265", compression_level=0)

u = UNSPARZ(
    path_sparse_bp1="output/*.npz",
    path_encoded_bp1="output/*.mp4",
    stem="reconstructed",
    output_path="reconstructed/",
    use_residuals=True,
)
u.run()
```

### GUI

```bash
sparz-gui
```

## Compression Outputs

For video codecs, SPARZ writes one encoded video per input stack:

```text
<input_stem>_compression_level_<N>.<ext>
```

When ROI detection is enabled for lossy codecs, it also writes:

```text
<input_stem>.npz
```

When residuals are enabled through the Python API, it writes:

```text
<input_stem>_compression_level_<N>_residual.zst
```

For `zstd`, SPARZ writes compressed arrays directly:

```text
<input_stem>_level_<N>.zst
```

## Codecs

| Codec | Output | Notes |
| --- | --- | --- |
| `x265` | `.mp4` | HEVC/H.265. Uses `gray12le`; level 0 uses x265 lossless mode. |
| `av1` | `.mp4` | AV1 with `gray16le`; slower, often compact. |
| `x264` | `.mp4` | H.264 with `gray16le`; fast and broadly compatible. |
| `ffv1` | `.avi` | 16-bit lossless video. ROI NPZ is skipped because the video is sufficient. |
| `prores` | `.mov` | ProRes profiles, useful for editing-oriented workflows. |
| `zstd` | `.zst` | Lossless array compression, not a video container. |

Video compression levels are `0`, `1`, `2`, and `3`. For `x265`, `av1`,
`x264`, and `prores`, lower levels preserve more information and are usually
larger; higher levels are more lossy and are usually smaller. Exact file sizes
are codec- and dataset-dependent and are not guaranteed to be monotonic.

For `ffv1`, all levels are lossless and change FFV1 encoder settings such as
coder, context, and slices. For `zstd`, levels are `0` through `22`, with higher
levels generally spending more CPU for more compression.

## ROI and Residuals

ROI detection is enabled by default. For lossy video codecs, SPARZ stores peak
pixels in an `.npz` sidecar so UNSPARZ can patch them back during
reconstruction.

Optional residual sidecars provide exact reconstruction for near-lossless video
codecs. Residuals are available through the Python API:

```python
z = SPARZIP(
    "data/*.tiff",
    stem="experiment",
    output_path="output/",
    save_residuals=True,
    residual_mode="auto",       # "auto", "background", or "full"
    residual_chunk_size=16,
)
z.run(codec="x265", compression_level=1)
```

Residual modes:

- `auto`: uses `background` when ROI is enabled, otherwise `full`.
- `background`: stores residuals outside ROI pixels and relies on ROI patching.
- `full`: stores residuals for all pixels, including ROI pixels.

Residuals are skipped for `ffv1` because that codec is already 16-bit lossless,
and they are not supported for `zstd` or custom user codecs. `UNSPARZ` applies
residuals by default when matching sidecars are present; pass
`use_residuals=False` in Python to ignore them.

## Metadata

Set `extract_metadata=True` in Python or `--metadata` on the CLI to extract TIFF
metadata before compression. DAT inputs do not carry TIFF metadata.

By default, metadata is embedded in the ROI `.npz` sidecar when one is written.
In Python, `save_metadata_to_json=True` also writes dataset-level JSON metadata
files. For `ffv1` single-file MKV packages, where no ROI NPZ is needed, SPARZ
attaches per-video metadata JSON files to the MKV when metadata extraction is
enabled.

UNSPARZ restores metadata when writing TIFF output. The streaming TIFF writer
writes pages incrementally instead of allocating a full `(T, H, W)` output
array.

## Single-File MKV Packages

Set `create_single_file=True` in Python or `--single-file` on the CLI to package
each encoded video into a matching MKV file. This creates one MKV per encoded
video, not one global archive for a whole experiment.

Packaging does not delete the generated video or sidecar files; the MKV is an
additional packaged copy.

Each MKV contains the video stream and matching attachments:

- ROI `.npz` sidecar when one exists.
- Residual `.zst` sidecar when residuals were requested and saved.
- Per-video metadata JSON when needed.

UNSPARZ treats MKV inputs as self-contained. During MKV reconstruction it only
uses sidecars extracted from the MKV itself; `.npz`, residual, or metadata files
sitting next to the MKV are ignored to avoid stale sidecar matches. To use
external sidecars, reconstruct from separate video files instead of MKV input,
or re-package the MKV with the correct attachments.

MKV extraction uses fresh temporary directories under `mkv_temp` and cleans
them on normal reconstruction completion and on initialization failures.

## Command Line Reference

### `sparz`

```text
sparz INPUT [OPTIONS]
```

Options:

```text
-o, --output DIR      Output directory. Default: ./
--stem NAME           Output stem. Default: derived from input.
--bp2 GLOB            Second biplane input glob.
-c, --codec CODEC     x265, av1, x264, ffv1, prores, or zstd. Default: x265.
-l, --level N         0-3 for video codecs, 0-22 for zstd. Default: 0.
--threshold FLOAT     Peak detection threshold. Default: 0.45.
--kernel N            Peak detection kernel size. Default: 9.
--no-roi              Disable ROI/peak detection.
--metadata            Extract and preserve TIFF metadata.
--single-file         Package each output video into a matching MKV.
-w, --workers N       Parallel workers. Default: 4.
--align               Align biplane images.
--reflect             Reflect the second biplane.
```

The CLI currently does not expose residual-sidecar options; use the Python API
for `save_residuals`, `residual_mode`, and `residual_chunk_size`.

### `unsparz`

```text
unsparz VIDEO [OPTIONS]
```

Options:

```text
-o, --output DIR      Output directory. Default: ./
--stem NAME           Output stem. Default: derived from input.
--npz GLOB            BP1 ROI NPZ sidecars. Auto-detected for non-MKV inputs.
--npz2 GLOB           BP2 ROI NPZ sidecars.
--video2 GLOB         BP2 encoded videos.
--no-roi              Skip ROI patching.
--format FMT          tiff or dat. Default: tiff.
-w, --workers N       Parallel workers. Default: 4.
```

`UNSPARZ` uses streaming reconstruction by default for video inputs. In Python,
pass `streaming=False` to use the legacy eager path.

## Biplane Data

```bash
sparz "bp1/*.tiff" --bp2 "bp2/*.tiff" -o output/ --codec x265 --level 0
unsparz "output/bp1*.mp4" --video2 "output/bp2*.mp4" --npz "output/bp1*.npz" --npz2 "output/bp2*.npz" -o reconstructed/
```

```python
z = SPARZIP(
    path_image_files1="bp1/*.tiff",
    path_image_files2="bp2/*.tiff",
    stem="biplane",
    output_path="output/",
    align_planes=True,
)
z.run(codec="x265", compression_level=0)
```

## License

MIT
