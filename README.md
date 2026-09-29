# SPARZ

SPARZ compresses Single Molecule Localization Microscopy (SMLM) image stacks by
encoding each stack as video and, when ROI detection is enabled, storing the
exact pixel values around detected peaks in sparse sidecars that are patched
back in during reconstruction.

The current pipeline supports TIFF and DAT inputs, single-plane and biplane
data, optional TIFF metadata preservation, optional exact residual sidecars,
and optional per-video MKV packaging. Inputs must be single-channel
(grayscale). Multi-channel TIFFs are rejected with an error.

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

`path_image_files1` and `path_image_files2` accept either a glob string or a
list of file paths, for example to process only the files missing from an
earlier run.

### GUI

```bash
sparz-gui
```

## Compression Outputs

Output files are named after the input files. For video codecs, SPARZ writes
one encoded video per input stack:

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

In biplane mode, if a bp1 and a bp2 input share a file name (for example
`bp1/img.tif` and `bp2/img.tif`), that pair's outputs get a `_bp1` or `_bp2`
suffix on the input stem (`img_bp1_compression_level_0.mp4`, `img_bp2.npz`)
so the two planes do not overwrite each other.

UNSPARZ writes one output per encoded video, named
`<stem>_<video_stem>.tiff` (or `.dat`). When `--stem` is not given on the CLI,
the stem is taken from the input file name, or is `output` when the pattern has
no usable name (such as `"output/*.mp4"`).

## Codecs

| Codec | Output | Notes |
| --- | --- | --- |
| `x265` | `.mp4` | HEVC/H.265. Uses `gray12le`. Level 0 uses x265 lossless mode. |
| `av1` | `.mp4` | AV1 with `gray16le`. Slower, often compact. |
| `x264` | `.mp4` | H.264 with `gray10le`. Fast and broadly compatible. |
| `ffv1` | `.avi` | 16-bit lossless video. ROI NPZ is skipped because the video is sufficient. |
| `prores` | `.mov` | ProRes profiles, useful for editing-oriented workflows. |
| `zstd` | `.zst` | Lossless array compression, not a video container. |

Video compression levels are `0`, `1`, `2`, and `3`. For `x265`, `av1`,
`x264`, and `prores`, lower levels preserve more information and are usually
larger, while higher levels are more lossy and are usually smaller. Exact file
sizes are codec- and dataset-dependent and are not guaranteed to be monotonic.

For `ffv1`, all levels are lossless and change FFV1 encoder settings such as
coder, context, and slices. For `zstd`, levels are `0` through `22`, with higher
levels generally spending more CPU for more compression.

## ROI and Residuals

ROI detection is enabled by default. For lossy video codecs, SPARZ stores peak
pixels in an `.npz` sidecar so UNSPARZ can patch them back during
reconstruction. In biplane mode, both planes use the union of the peaks found
in either plane.

In Python, `peaks_process="median"` replaces each peak neighbourhood with its
median in the video only, which gives the encoder a smoother background. The
`.npz` sidecar and the residuals keep the original values, so the exact peak
intensities are still restored. The median patch is only applied when an `.npz`
sidecar is written.

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
residuals by default when matching sidecars are present. Pass
`use_residuals=False` in Python to ignore them.

## Metadata

Set `extract_metadata=True` in Python or `--metadata` on the CLI to extract TIFF
metadata before compression. DAT inputs do not carry TIFF metadata.

Where the metadata is stored depends on the run:

- When an ROI `.npz` sidecar is written, the metadata is embedded in it.
- When no `.npz` is written (`ffv1`, `zstd`, or ROI detection off), SPARZ
  writes dataset-level `<stem>_metadata_bp1.json` (and `_bp2.json`) files next
  to the compressed outputs.
- In Python, `save_metadata_to_json=True` always writes the JSON files.
- For MKV packages, per-video metadata JSON is attached to the MKV whenever no
  `.npz` carries it.

UNSPARZ restores metadata when writing TIFF output, including the original
`ImageDescription`. It looks for the metadata JSON in its output directory and
then next to the encoded videos. Output TIFFs are written as a single
`(T, H, W)` series, and the streaming TIFF writer writes pages incrementally
instead of allocating the full output array.

## Single-File MKV Packages

Set `create_single_file=True` in Python or `--single-file` on the CLI to package
each encoded video into a matching MKV file. This creates one MKV per encoded
video, not one global archive for a whole experiment.

Each MKV contains the video stream and matching attachments:

- ROI `.npz` sidecar when one exists.
- Residual `.zst` sidecar when residuals were requested and saved.
- Per-video metadata JSON when needed.

After an MKV is created successfully, the source video and the sidecars
attached to it are deleted, so the MKV replaces them. When every video is
packaged successfully, the dataset-level metadata JSON files are removed too.
If packaging fails, the original files are kept.

UNSPARZ treats MKV inputs as self-contained. During MKV reconstruction it only
uses sidecars extracted from the MKV itself. `.npz`, residual, or metadata files
sitting next to the MKV are ignored to avoid stale sidecar matches. To use
external sidecars, reconstruct from separate video files instead of MKV input,
or re-package the MKV with the correct attachments.

MKV extraction uses fresh temporary directories under `mkv_temp` and removes
them, and the `mkv_temp` folder once it is empty, on normal reconstruction
completion and on initialization failures.

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

The CLI currently does not expose residual-sidecar options. Use the Python API
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

When `--npz` is not given, every `.npz` next to the videos is used and matched
to the videos by sorted order. For biplane data, pass `--npz` and `--npz2`
explicitly so each plane gets its own sidecars.

`UNSPARZ` uses streaming reconstruction by default for video inputs. In Python,
pass `streaming=False` to use the legacy eager path.

## Biplane Data

With matching file names in the two plane folders (`bp1/img.tif`,
`bp2/img.tif`), the outputs carry `_bp1` and `_bp2` suffixes:

```bash
sparz "bp1/*.tiff" --bp2 "bp2/*.tiff" -o output/ --codec x265 --level 0
unsparz "output/*_bp1_compression_level_0.mp4" --npz "output/*_bp1.npz" \
        --video2 "output/*_bp2_compression_level_0.mp4" --npz2 "output/*_bp2.npz" \
        --stem reconstructed -o reconstructed/
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

MIT. See [LICENSE](LICENSE).
