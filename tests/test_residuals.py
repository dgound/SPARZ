"""Helper-level tests for the SPARZ residual sidecar layer."""
import json
import os
import sys

import numpy as np
import dask.array as da

try:
    import pytest
except ImportError:
    pytest = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import SPARZ as sparz_mod
from SPARZ import (
    encode_residual_to_bytes,
    decode_residual_from_bytes,
    save_residual_sidecar,
    save_residual_sidecar_streaming,
    iter_residual_chunks_from_path,
    read_residual_header,
    read_residual_index,
    read_residual_chunk_at,
    load_residual_sidecar,
    apply_residuals,
    get_residual_path_for_video,
    find_matching_sidecars_for_video,
    _container_for_mkv_video_codec,
    RESIDUAL_SUFFIX,
    RESIDUAL_VERSION,
    RESIDUAL_VERSION_V2,
    RESIDUAL_FLAG_INDEXED,
    RESIDUAL_INDEX_ENTRY_SIZE,
    SPARZIP,
    UNSPARZ,
)


def _make_decoded_with_loss(original, scale=4):
    """Mimic a lossy codec by zeroing the lowest few bits of each value."""
    mask = np.uint16(0xFFFF) << np.uint16(scale)
    return (original & mask).astype(np.uint16)


def test_residual_roundtrip_full_mode():
    rng = np.random.default_rng(0)
    original = rng.integers(0, 4096, size=(5, 8, 12), dtype=np.uint16)
    decoded = _make_decoded_with_loss(original, scale=2)

    residual = original.astype(np.int32) - decoded.astype(np.int32)
    payload = encode_residual_to_bytes(
        residual, original_dtype="uint16", mode="full",
        codec="x265", compression_level=2,
    )
    decoded_residual, header = decode_residual_from_bytes(payload)

    assert header["mode"] == "full"
    assert header["original_dtype"] == "uint16"
    assert header["codec"] == "x265"
    assert header["compression_level"] == 2
    np.testing.assert_array_equal(decoded_residual, residual)

    reconstructed = apply_residuals(decoded, decoded_residual, original.dtype)
    np.testing.assert_array_equal(reconstructed, original)


def test_residual_background_plus_roi_reconstructs_exactly(tmp_path):
    rng = np.random.default_rng(1)
    original = rng.integers(0, 4096, size=(4, 6, 6), dtype=np.uint16)

    # Synthetic ROI mask: a few "peak" pixels per frame
    roi_mask = np.zeros_like(original, dtype=bool)
    roi_mask[:, 1, 1] = True
    roi_mask[:, 3, 4] = True
    roi_mask[0, 5, 5] = True

    # ROI sparse keeps the ORIGINAL values at those pixels
    roi_sparse = np.where(roi_mask, original, np.uint16(0))
    decoded = _make_decoded_with_loss(original, scale=3)

    # Background residual zeros at ROI pixels by construction
    raw_residual = original.astype(np.int32) - decoded.astype(np.int32)
    background_residual = np.where(roi_mask, np.int32(0), raw_residual)

    sidecar_path = str(tmp_path / "fake_compression_level_0_residual.zst")
    save_residual_sidecar(
        sidecar_path, background_residual,
        original_dtype="uint16", mode="background",
        codec="av1", compression_level=0,
    )
    loaded_residual, header = load_residual_sidecar(sidecar_path)
    assert header["mode"] == "background"

    # Reconstruct in the documented order: decoded -> apply residual -> patch ROI
    after_residual = apply_residuals(decoded, loaded_residual, np.uint16)
    final = np.where(roi_sparse != 0, roi_sparse, after_residual)

    np.testing.assert_array_equal(final, original)
    # Residual should be exactly zero at every ROI pixel
    assert (loaded_residual[roi_mask] == 0).all()


def test_residual_dtype_promotion_to_int32():
    # Construct a residual that exceeds int16 range so we exercise the int32 branch
    decoded = np.zeros((2, 3, 3), dtype=np.uint16)
    original = np.full((2, 3, 3), 50000, dtype=np.uint16)
    residual = original.astype(np.int32) - decoded.astype(np.int32)
    assert residual.max() > 32767

    payload = encode_residual_to_bytes(
        residual, original_dtype="uint16", mode="full",
        codec="x265", compression_level=0,
    )
    decoded_residual, header = decode_residual_from_bytes(payload)
    assert header["residual_dtype"] == "int32"
    np.testing.assert_array_equal(decoded_residual, residual)


def test_find_matching_sidecars_only_matches_correct_stem(tmp_path):
    out_dir = tmp_path
    # Two videos sharing a directory; each has its own NPZ + residual
    v1 = out_dir / "movieA_compression_level_0.mp4"
    v2 = out_dir / "movieB_compression_level_0.mp4"
    v1.write_bytes(b"")
    v2.write_bytes(b"")

    (out_dir / "movieA.npz").write_bytes(b"")
    (out_dir / "movieB.npz").write_bytes(b"")
    (out_dir / ("movieA_compression_level_0" + RESIDUAL_SUFFIX)).write_bytes(b"")
    # NOTE: deliberately NO residual for movieB

    sidecars_a = find_matching_sidecars_for_video(str(v1))
    sidecars_b = find_matching_sidecars_for_video(str(v2))

    assert sidecars_a["npz"] is not None and sidecars_a["npz"].endswith("movieA.npz")
    assert sidecars_a["residual"] is not None
    assert "movieA" in os.path.basename(sidecars_a["residual"])

    assert sidecars_b["npz"] is not None and sidecars_b["npz"].endswith("movieB.npz")
    assert sidecars_b["residual"] is None  # absent file should be reported as None


def test_get_residual_path_for_video_appends_suffix():
    p = "/tmp/foo/bar_compression_level_2.mp4"
    assert get_residual_path_for_video(p) == "/tmp/foo/bar_compression_level_2" + RESIDUAL_SUFFIX
    p2 = "/tmp/foo/bar_compression_level_0.avi"
    assert get_residual_path_for_video(p2) == "/tmp/foo/bar_compression_level_0" + RESIDUAL_SUFFIX


def test_apply_residuals_clips_to_dtype_range():
    # Adding a positive residual that overflows uint16 should clip rather than wrap
    decoded = np.full((1, 2, 2), 60000, dtype=np.uint16)
    residual = np.full((1, 2, 2), 10000, dtype=np.int32)  # would overflow
    out = apply_residuals(decoded, residual, np.uint16)
    assert out.dtype == np.uint16
    assert (out == 65535).all()


def test_apply_residuals_clips_negative():
    decoded = np.full((1, 2, 2), 100, dtype=np.uint16)
    residual = np.full((1, 2, 2), -500, dtype=np.int32)
    out = apply_residuals(decoded, residual, np.uint16)
    assert (out == 0).all()


# ---------------------------------------------------------------------------
# v2 streaming format
# ---------------------------------------------------------------------------

def _chunk_iter_for(residual, chunk_size):
    T = residual.shape[0]
    for t0 in range(0, T, chunk_size):
        t1 = min(T, t0 + chunk_size)
        yield t0, t1, residual[t0:t1]


def test_streaming_v2_roundtrip(tmp_path):
    rng = np.random.default_rng(7)
    original = rng.integers(0, 4096, size=(11, 6, 5), dtype=np.uint16)
    decoded = _make_decoded_with_loss(original, scale=2)
    residual = (original.astype(np.int32) - decoded.astype(np.int32))

    path = str(tmp_path / "v2_residual.zst")
    save_residual_sidecar_streaming(
        path, shape=residual.shape, original_dtype=np.uint16,
        mode="full", codec="x265", compression_level=2,
        chunk_iter=_chunk_iter_for(residual, chunk_size=4),
        chunk_size=4,
    )
    header = read_residual_header(path)
    assert header["version"] == RESIDUAL_VERSION_V2
    assert header["chunk_size"] == 4
    # 11 frames at chunk_size 4 -> 3 chunks (4, 4, 3)
    assert header["n_chunks"] == 3

    full, _h = load_residual_sidecar(path)
    np.testing.assert_array_equal(full.astype(np.int32), residual)
    reconstructed = apply_residuals(decoded, full, np.uint16)
    np.testing.assert_array_equal(reconstructed, original)


def test_streaming_iter_yields_each_chunk_once(tmp_path):
    residual = np.arange(2 * 3 * 3, dtype=np.int16).reshape(2, 3, 3)
    path = str(tmp_path / "two_frames.zst")
    save_residual_sidecar_streaming(
        path, shape=residual.shape, original_dtype=np.uint16,
        mode="full", codec="x264", compression_level=0,
        chunk_iter=_chunk_iter_for(residual.astype(np.int32), chunk_size=1),
        chunk_size=1,
    )
    seen = list(iter_residual_chunks_from_path(path))
    assert len(seen) == 2
    np.testing.assert_array_equal(seen[0][2].astype(np.int32), residual[:1].astype(np.int32))
    np.testing.assert_array_equal(seen[1][2].astype(np.int32), residual[1:].astype(np.int32))


def test_streaming_writer_rejects_chunk_gap(tmp_path):
    residual = np.zeros((4, 2, 2), dtype=np.int32)
    path = str(tmp_path / "gap.zst")

    def bad_iter():
        yield 0, 2, residual[:2]
        # skip frames 2..3, only emit 3..4
        yield 3, 4, residual[3:4]

    try:
        save_residual_sidecar_streaming(
            path, shape=residual.shape, original_dtype=np.uint16,
            mode="full", codec="x265", compression_level=0,
            chunk_iter=bad_iter(), chunk_size=2,
        )
    except ValueError as e:
        assert "chunk gap" in str(e) or "expected start" in str(e)
    else:
        raise AssertionError("expected ValueError for chunk gap")


def test_streaming_writer_rejects_partial_coverage(tmp_path):
    residual = np.zeros((4, 2, 2), dtype=np.int32)
    path = str(tmp_path / "short.zst")

    def short_iter():
        yield 0, 2, residual[:2]
        # never yields frames 2..3

    try:
        save_residual_sidecar_streaming(
            path, shape=residual.shape, original_dtype=np.uint16,
            mode="full", codec="x265", compression_level=0,
            chunk_iter=short_iter(), chunk_size=2,
        )
    except ValueError as e:
        assert "covered" in str(e) or "expected" in str(e)
    else:
        raise AssertionError("expected ValueError for partial coverage")


# ---------------------------------------------------------------------------
# Shape-mismatch failure modes
# ---------------------------------------------------------------------------

def test_apply_residuals_shape_mismatch_raises():
    decoded = np.zeros((4, 5, 5), dtype=np.uint16)
    residual = np.zeros((3, 5, 5), dtype=np.int32)
    try:
        apply_residuals(decoded, residual, np.uint16, context="unit-test")
    except ValueError as e:
        assert "shape mismatch" in str(e)
    else:
        raise AssertionError("expected ValueError on shape mismatch")


def _make_sparzip_skeleton():
    """Build a SPARZIP-like object without running __init__ (for unit-test reuse)."""
    obj = SPARZIP.__new__(SPARZIP)
    obj.find_roi = False
    obj.residual_mode = "auto"
    obj.residual_chunk_size = 4
    obj.output_path = "/tmp/"
    return obj


def test_compute_residual_chunk_shape_mismatch_raises():
    obj = _make_sparzip_skeleton()
    orig = np.zeros((3, 4, 4), dtype=np.uint16)
    decoded = np.zeros((4, 4, 4), dtype=np.uint16)
    try:
        obj.compute_residual_chunk(orig, decoded, mode="full")
    except ValueError as e:
        assert "shape mismatch" in str(e)
    else:
        raise AssertionError("expected ValueError on chunk shape mismatch")


def _make_sparzip_for_process_images(bp1_arrays, bp2_arrays=None, peak_process=None):
    obj = SPARZIP.__new__(SPARZIP)
    obj.single_plane = bp2_arrays is None
    obj.kernel_size = 3
    obj.rel_threshold = 0.5
    obj.peak_process = peak_process
    obj.bp1 = [da.from_array(a, chunks=(1, a.shape[1], a.shape[2])) for a in bp1_arrays]
    obj.bp2 = (
        [da.from_array(a, chunks=(1, a.shape[1], a.shape[2])) for a in bp2_arrays]
        if bp2_arrays is not None else None
    )
    return obj


def test_process_images_single_plane_uses_each_stack_shape():
    first = np.zeros((1, 5, 5), dtype=np.uint16)
    first[0, 2, 2] = 100
    second = np.zeros((1, 6, 6), dtype=np.uint16)
    second[0, 3, 3] = 100

    obj = _make_sparzip_for_process_images([first, second])
    processed, processed_bp2 = obj.process_images()

    assert processed_bp2 is None
    assert processed[0].compute().shape == first.shape
    second_dense = processed[1].compute().todense()
    assert second_dense.shape == second.shape
    assert second_dense[0, 3, 3] == 100


def test_process_images_biplane_uses_each_stack_shape():
    bp1_first = np.zeros((1, 5, 5), dtype=np.uint16)
    bp1_first[0, 2, 2] = 100
    bp2_first = np.zeros((1, 5, 5), dtype=np.uint16)
    bp2_first[0, 1, 1] = 100
    bp1_second = np.zeros((1, 6, 6), dtype=np.uint16)
    bp1_second[0, 3, 3] = 100
    bp2_second = np.zeros((1, 6, 6), dtype=np.uint16)
    bp2_second[0, 4, 4] = 100

    obj = _make_sparzip_for_process_images(
        [bp1_first, bp1_second],
        [bp2_first, bp2_second],
    )
    processed_bp1, processed_bp2 = obj.process_images()

    assert processed_bp1[0].compute().shape == bp1_first.shape
    assert processed_bp2[0].compute().shape == bp2_first.shape
    bp1_second_dense = processed_bp1[1].compute().todense()
    bp2_second_dense = processed_bp2[1].compute().todense()
    assert bp1_second_dense.shape == bp1_second.shape
    assert bp2_second_dense.shape == bp2_second.shape
    assert bp1_second_dense[0, 3, 3] == 100
    assert bp2_second_dense[0, 4, 4] == 100


def test_process_images_median_patch_runs_before_sparse_graph():
    frame = np.zeros((1, 5, 5), dtype=np.uint16)
    frame[0, 2, 2] = 100

    obj = _make_sparzip_for_process_images([frame], peak_process="median")
    processed, _ = obj.process_images()

    assert processed[0].compute().todense().sum() == 0


# ---------------------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------------------

def test_resolve_mode_full_with_roi_stays_full():
    obj = _make_sparzip_skeleton()
    obj.find_roi = True
    obj.residual_mode = "full"
    assert obj._resolve_residual_mode() == "full"


def test_resolve_mode_auto_picks_background_when_roi_on():
    obj = _make_sparzip_skeleton()
    obj.find_roi = True
    obj.residual_mode = "auto"
    assert obj._resolve_residual_mode() == "background"


def test_resolve_mode_auto_picks_full_when_roi_off():
    obj = _make_sparzip_skeleton()
    obj.find_roi = False
    obj.residual_mode = "auto"
    assert obj._resolve_residual_mode() == "full"


def test_resolve_mode_background_without_roi_raises():
    obj = _make_sparzip_skeleton()
    obj.find_roi = False
    obj.residual_mode = "background"
    try:
        obj._resolve_residual_mode()
    except ValueError as e:
        assert "background" in str(e) and "find_roi" in str(e)
    else:
        raise AssertionError("expected ValueError for background mode without ROI")


# ---------------------------------------------------------------------------
# Per-video sidecar matching for MKV-style temp dirs
# ---------------------------------------------------------------------------

def test_find_matching_sidecars_in_extracted_temp_dir(tmp_path):
    """When MKV is extracted, sidecars and the video share a temp dir.

    The extracted video filename matches the MKV stem (e.g. ``movieA_compression_level_0.mp4``),
    so find_matching_sidecars_for_video must locate the correctly-named .npz and
    residual .zst alongside it.
    """
    temp_dir = tmp_path / "mkv_temp"
    temp_dir.mkdir()
    video_path = temp_dir / "movieA_compression_level_0.mp4"
    video_path.write_bytes(b"")
    (temp_dir / "movieA.npz").write_bytes(b"")
    (temp_dir / ("movieA_compression_level_0" + RESIDUAL_SUFFIX)).write_bytes(b"")
    # A different, unrelated stem must NOT be matched.
    (temp_dir / "movieB.npz").write_bytes(b"")
    (temp_dir / ("movieB_compression_level_0" + RESIDUAL_SUFFIX)).write_bytes(b"")

    s = find_matching_sidecars_for_video(str(video_path))
    assert s["npz"].endswith("movieA.npz")
    assert os.path.basename(s["residual"]).startswith("movieA")


def test_find_matching_sidecars_with_explicit_search_dir(tmp_path):
    video_path = tmp_path / "elsewhere" / "movieA_compression_level_0.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"")
    side_dir = tmp_path / "sidecars"
    side_dir.mkdir()
    (side_dir / "movieA.npz").write_bytes(b"")
    (side_dir / ("movieA_compression_level_0" + RESIDUAL_SUFFIX)).write_bytes(b"")

    # Without search_dir, looks next to the video and finds nothing.
    s_default = find_matching_sidecars_for_video(str(video_path))
    assert s_default["npz"] is None and s_default["residual"] is None

    # With explicit search_dir, finds both.
    s_side = find_matching_sidecars_for_video(str(video_path), search_dir=str(side_dir))
    assert s_side["npz"] is not None
    assert s_side["residual"] is not None


# ---------------------------------------------------------------------------
# Background residual must remain non-trivial / full mode preserves ROI deltas
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# v2 indexed format: random access without scanning earlier chunks
# ---------------------------------------------------------------------------

def test_v2_writer_emits_index(tmp_path):
    residual = np.arange(6 * 2 * 2, dtype=np.int16).reshape(6, 2, 2)
    path = str(tmp_path / "indexed.zst")
    save_residual_sidecar_streaming(
        path, shape=residual.shape, original_dtype=np.uint16,
        mode="full", codec="x265", compression_level=0,
        chunk_iter=_chunk_iter_for(residual.astype(np.int32), chunk_size=2),
        chunk_size=2,
    )
    header = read_residual_header(path)
    assert header["flags"] & RESIDUAL_FLAG_INDEXED
    assert header["n_chunks"] == 3
    assert "index" in header
    index = header["index"]
    # Three chunks of 2 frames each, with strictly increasing offsets.
    assert [e[0] for e in index] == [0, 2, 4]
    assert [e[1] for e in index] == [2, 2, 2]
    offsets = [e[2] for e in index]
    assert all(offsets[i] < offsets[i + 1] for i in range(len(offsets) - 1))


def test_read_residual_chunk_at_does_not_scan(tmp_path, monkeypatch):
    """Loading the last chunk via the index should read only its bytes.

    We monkeypatch the zstd decompressor to count calls and verify that exactly
    one decompression happens when we request a single chunk by index entry.
    """
    residual = np.arange(8 * 2 * 2, dtype=np.int16).reshape(8, 2, 2)
    path = str(tmp_path / "indexed_single_read.zst")
    save_residual_sidecar_streaming(
        path, shape=residual.shape, original_dtype=np.uint16,
        mode="full", codec="x265", compression_level=0,
        chunk_iter=_chunk_iter_for(residual.astype(np.int32), chunk_size=2),
        chunk_size=2,
    )
    header = read_residual_header(path)
    index = header["index"]
    assert len(index) == 4
    last_entry = index[-1]
    assert last_entry[0] == 6  # last chunk starts at frame 6

    import zstandard as zstd_mod
    calls = {"n": 0}
    real_decompress = zstd_mod.ZstdDecompressor.decompress

    def counting_decompress(self, data):
        calls["n"] += 1
        return real_decompress(self, data)

    monkeypatch.setattr(zstd_mod.ZstdDecompressor, "decompress", counting_decompress)

    chunk = read_residual_chunk_at(path, last_entry, header)
    assert calls["n"] == 1
    np.testing.assert_array_equal(chunk.astype(np.int32), residual[6:8].astype(np.int32))


def test_read_residual_index_handles_legacy_v1_files(tmp_path):
    """v1 (single-blob) sidecars have no index — read_residual_index returns None."""
    residual = np.arange(2 * 2 * 2, dtype=np.int16).reshape(2, 2, 2)
    path = str(tmp_path / "v1.zst")
    save_residual_sidecar(
        path, residual.astype(np.int32),
        original_dtype=np.uint16, mode="full",
        codec="x265", compression_level=0,
    )
    assert read_residual_index(path) is None


# ---------------------------------------------------------------------------
# Two-pass dtype auto-selection
# ---------------------------------------------------------------------------

def _make_sparzip_for_chunk_iter():
    obj = SPARZIP.__new__(SPARZIP)
    obj.find_roi = False
    obj.residual_mode = "full"
    obj.residual_chunk_size = 4
    obj.output_path = "/tmp/"
    return obj


def test_scan_residual_dtype_picks_int16_for_small_residuals():
    obj = _make_sparzip_for_chunk_iter()
    chunks = [
        (0, 4, np.full((4, 2, 2), 100, dtype=np.int32)),
        (4, 8, np.full((4, 2, 2), -200, dtype=np.int32)),
    ]
    factory = lambda: iter(chunks)
    assert obj._scan_residual_dtype(factory) == np.int16


def test_scan_residual_dtype_picks_int32_for_large_residuals():
    obj = _make_sparzip_for_chunk_iter()
    chunks = [
        (0, 4, np.full((4, 2, 2), 100, dtype=np.int32)),
        (4, 8, np.full((4, 2, 2), 50000, dtype=np.int32)),  # exceeds int16
    ]
    factory = lambda: iter(chunks)
    assert obj._scan_residual_dtype(factory) == np.int32


def test_scan_residual_dtype_at_int16_boundary():
    obj = _make_sparzip_for_chunk_iter()
    chunks = [
        (0, 1, np.array([[[32767]]], dtype=np.int32)),
        (1, 2, np.array([[[-32768]]], dtype=np.int32)),
    ]
    factory = lambda: iter(chunks)
    assert obj._scan_residual_dtype(factory) == np.int16

    chunks_overflow = [(0, 1, np.array([[[32768]]], dtype=np.int32))]
    assert obj._scan_residual_dtype(lambda: iter(chunks_overflow)) == np.int32


# ---------------------------------------------------------------------------
# Decoded vs original frame-count divergence (extra / short)
# ---------------------------------------------------------------------------

class _StubDask:
    def __init__(self, arr):
        self._arr = arr
        self.shape = arr.shape
        self.dtype = arr.dtype

    def __getitem__(self, idx):
        return _StubDask(self._arr[idx])

    def compute(self):
        return self._arr


def _stub_iter_decoded(self_obj, video_path, chunk_size):
    arr = self_obj._test_decoded
    T = arr.shape[0]
    for t0 in range(0, T, chunk_size):
        t1 = min(T, t0 + chunk_size)
        yield t0, t1, arr[t0:t1]


def _make_factory_obj(decoded_frames, original_frames, chunk_size=2, mode="full"):
    obj = SPARZIP.__new__(SPARZIP)
    obj.find_roi = False
    obj.residual_mode = "full"
    obj.residual_chunk_size = chunk_size
    obj.output_path = "/tmp/"
    obj._test_decoded = decoded_frames
    # Bind per-instance overrides without touching the class.
    import types
    obj.iter_decoded_video_chunks = types.MethodType(_stub_iter_decoded, obj)
    return obj


def test_factory_raises_on_extra_decoded_frames():
    decoded = np.zeros((6, 2, 2), dtype=np.uint16)
    original = np.zeros((4, 2, 2), dtype=np.uint16)
    obj = _make_factory_obj(decoded, original, chunk_size=2)
    factory = obj._make_residual_chunk_iter_factory(
        "fake_video.mp4", _StubDask(original),
        mode="full", roi_info=None, chunk_size=2, H=2, W=2,
    )
    try:
        list(factory())
    except ValueError as e:
        assert "more frames than original" in str(e)
    else:
        raise AssertionError("expected ValueError when decoded has extra frames")


def test_factory_raises_on_short_decoded():
    decoded = np.zeros((2, 2, 2), dtype=np.uint16)
    original = np.zeros((4, 2, 2), dtype=np.uint16)
    obj = _make_factory_obj(decoded, original, chunk_size=2)
    factory = obj._make_residual_chunk_iter_factory(
        "fake_video.mp4", _StubDask(original),
        mode="full", roi_info=None, chunk_size=2, H=2, W=2,
    )
    try:
        list(factory())
    except ValueError as e:
        assert "fewer frames than original" in str(e)
    else:
        raise AssertionError("expected ValueError when decoded has fewer frames")


def test_factory_yields_matched_chunks_when_aligned():
    rng = np.random.default_rng(42)
    original = rng.integers(0, 4096, size=(6, 2, 2), dtype=np.uint16)
    decoded = _make_decoded_with_loss(original, scale=2)
    obj = _make_factory_obj(decoded, original, chunk_size=2)
    factory = obj._make_residual_chunk_iter_factory(
        "fake_video.mp4", _StubDask(original),
        mode="full", roi_info=None, chunk_size=2, H=2, W=2,
    )
    chunks = list(factory())
    assert [(c[0], c[1]) for c in chunks] == [(0, 2), (2, 4), (4, 6)]
    full_residual = np.concatenate([c[2] for c in chunks], axis=0)
    np.testing.assert_array_equal(
        full_residual,
        original.astype(np.int32) - decoded.astype(np.int32),
    )


# ---------------------------------------------------------------------------
# Biplane MKV-style sidecar alignment
# ---------------------------------------------------------------------------

def _make_unsparz_for_extract():
    """Build an UNSPARZ skeleton without running __init__."""
    obj = UNSPARZ.__new__(UNSPARZ)
    obj.temp_dirs_to_cleanup = []
    obj.path_residual_bp1 = None
    obj.path_residual_bp2 = None
    return obj


def test_wire_sidecars_from_records_aligns_per_plane(tmp_path):
    """_wire_sidecars_from_records assigns per-plane lists in input order."""
    obj = _make_unsparz_for_extract()
    bp1_records = [
        {"video": str(tmp_path / "movie1A_compression_level_0.mp4"),
         "npz": str(tmp_path / "movie1A.npz"),
         "residual": str(tmp_path / "movie1A_compression_level_0_residual.zst")},
        {"video": str(tmp_path / "movie2A_compression_level_0.mp4"),
         "npz": str(tmp_path / "movie2A.npz"),
         "residual": None},
    ]
    bp2_records = [
        {"video": str(tmp_path / "movie1B_compression_level_0.mp4"),
         "npz": str(tmp_path / "movie1B.npz"),
         "residual": str(tmp_path / "movie1B_compression_level_0_residual.zst")},
        {"video": str(tmp_path / "movie2B_compression_level_0.mp4"),
         "npz": str(tmp_path / "movie2B.npz"),
         "residual": None},
    ]

    obj._wire_sidecars_from_records(bp1_records, plane="bp1")
    obj._wire_sidecars_from_records(bp2_records, plane="bp2")

    # Both planes get full NPZ lists in record order — no swapping.
    assert obj.path_sparse_bp1 == [r["npz"] for r in bp1_records]
    assert obj.path_sparse_bp2 == [r["npz"] for r in bp2_records]
    assert obj.path_sparse_bp1 != obj.path_sparse_bp2

    # Residual lists keep None gaps for videos with no sidecar.
    assert obj.path_residual_bp1 == [r["residual"] for r in bp1_records]
    assert obj.path_residual_bp2 == [r["residual"] for r in bp2_records]
    # And the bp2 list is not the same object as bp1.
    assert obj.path_residual_bp1[0].endswith("movie1A_compression_level_0_residual.zst")
    assert obj.path_residual_bp2[0].endswith("movie1B_compression_level_0_residual.zst")


def test_wire_sidecars_passes_partial_npz_list_through(tmp_path):
    """Partial NPZ coverage now passes a per-video list through with None gaps.

    This lets per-video codec-aware validation decide whether each missing
    NPZ is acceptable (e.g. ffv1 doesn't need one). Older behaviour would
    set path_sparse_bp1 to None on any gap, which over-rejected ffv1 inputs.
    """
    obj = _make_unsparz_for_extract()
    records = [
        {"video": str(tmp_path / "a.mp4"), "npz": str(tmp_path / "a.npz"),
         "residual": None, "codec": "x265"},
        {"video": str(tmp_path / "b.mkv"), "npz": None,
         "residual": None, "codec": "ffv1"},
    ]
    obj._wire_sidecars_from_records(records, plane="bp1")
    assert obj.path_sparse_bp1 == [records[0]["npz"], None]
    assert obj._codec_bp1 == ["x265", "ffv1"]


def test_extract_one_mkv_uses_stem_matched_sidecars(tmp_path):
    """_extract_one_mkv uses a fresh temp dir and stem-matched sidecars.

    We cannot run ffmpeg/ffprobe in unit tests, so we stub them out: the codec
    probe returns ``None`` (which maps to ``.mkv``), video extraction creates
    a dummy output, and the attachment extractor writes only this MKV's
    sidecars into the fresh temp directory.
    """
    obj = _make_unsparz_for_extract()
    mkv_dir = tmp_path / "mkvs"
    mkv_dir.mkdir()
    mkv_file = mkv_dir / "movieA_compression_level_0.mkv"
    mkv_file.write_bytes(b"")

    stale_root = mkv_dir / "mkv_temp"
    stale_root.mkdir()
    stale_video = stale_root / "movieA_compression_level_0.mkv"
    stale_video.write_bytes(b"stale")
    (stale_root / "movieA_compression_level_0_metadata_bp1.json").write_text(
        json.dumps([{"tags": {"Software": "stale-metadata"}}])
    )
    (stale_root / "movieB_compression_level_0_metadata_bp1.json").write_text(
        json.dumps([{"tags": {"Software": "wrong-video"}}])
    )
    (mkv_dir / "movieA.npz").write_bytes(b"external-npz")
    (mkv_dir / ("movieA_compression_level_0" + RESIDUAL_SUFFIX)).write_bytes(b"external-residual")
    (mkv_dir / "movieA_compression_level_0_metadata_bp1.json").write_text(
        json.dumps([{"tags": {"Software": "external-metadata"}}])
    )

    def _fake_extract_attachments(_self, _mkv_file, temp_dir):
        temp_dir_path = os.path.abspath(temp_dir)
        with open(os.path.join(temp_dir_path, "movieA.npz"), "wb") as f:
            f.write(b"")
        with open(os.path.join(temp_dir_path, "movieA_compression_level_0" + RESIDUAL_SUFFIX), "wb") as f:
            f.write(b"")
        return 1, 1

    # Stub out the ffmpeg pieces so no subprocess is launched.
    import types
    obj._extract_mkv_attachments = types.MethodType(_fake_extract_attachments, obj)

    import SPARZ as _sparz
    real_probe = _sparz.ffmpeg.probe
    real_input = _sparz.ffmpeg.input
    _sparz.ffmpeg.probe = lambda *_a, **_kw: {"streams": []}

    class _FakeFfmpegInput:
        def output(self, output_path, *_args, **_kwargs):
            self.output_path = output_path
            return self

        def run(self, *_args, **_kwargs):
            with open(self.output_path, "wb") as f:
                f.write(b"fresh")

    _sparz.ffmpeg.input = lambda *_a, **_kw: _FakeFfmpegInput()
    try:
        record = obj._extract_one_mkv(str(mkv_file))
    finally:
        _sparz.ffmpeg.probe = real_probe
        _sparz.ffmpeg.input = real_input

    assert record["video"] != str(stale_video)
    assert os.path.basename(record["video"]) == "movieA_compression_level_0.mkv"
    assert os.path.dirname(record["video"]) == record["temp_dir"]
    assert record["temp_dir"] in obj.temp_dirs_to_cleanup
    assert record["npz"].endswith("movieA.npz")
    assert os.path.dirname(record["npz"]) == record["temp_dir"]
    assert os.path.basename(record["residual"]).startswith("movieA")
    assert record["metadata_bp1"] is None


def test_extract_one_mkv_ignores_external_mkv_dir_sidecars(tmp_path, monkeypatch):
    """MKV single-file mode should not silently bind same-directory sidecars."""
    obj = _make_unsparz_for_extract()
    mkv_dir = tmp_path / "mkvs"
    mkv_dir.mkdir()
    mkv_file = mkv_dir / "movieA_compression_level_0.mkv"
    mkv_file.write_bytes(b"")
    (mkv_dir / "movieA.npz").write_bytes(b"external-npz")
    (mkv_dir / ("movieA_compression_level_0" + RESIDUAL_SUFFIX)).write_bytes(b"external-residual")
    (mkv_dir / "movieA_compression_level_0_metadata_bp1.json").write_text(
        json.dumps([{"tags": {"Software": "external-metadata"}}])
    )

    import types
    obj._extract_mkv_attachments = types.MethodType(lambda _self, _m, _t: (0, 0), obj)

    monkeypatch.setattr(sparz_mod.ffmpeg, "probe", lambda *_a, **_kw: {"streams": []})

    class _FakeFfmpegInput:
        def output(self, output_path, *_args, **_kwargs):
            self.output_path = output_path
            return self

        def run(self, *_args, **_kwargs):
            with open(self.output_path, "wb") as f:
                f.write(b"fresh")

    monkeypatch.setattr(sparz_mod.ffmpeg, "input", lambda *_a, **_kw: _FakeFfmpegInput())

    record = obj._extract_one_mkv(str(mkv_file))
    assert os.path.dirname(record["video"]) == record["temp_dir"]
    assert record["npz"] is None
    assert record["residual"] is None
    assert record["metadata_bp1"] is None
    assert record["metadata_bp2"] is None


def test_unsparz_init_cleans_mkv_temp_dirs_on_error(tmp_path, monkeypatch):
    """Fresh MKV extraction dirs must not survive __init__ validation failures."""
    mkv_file = tmp_path / "movieA_compression_level_0.mkv"
    mkv_file.write_bytes(b"")
    temp_dir = tmp_path / "mkv_temp" / "movieA_compression_level_0_fixture"
    temp_dir.mkdir(parents=True)
    extracted = temp_dir / "movieA_compression_level_0.mkv"
    extracted.write_bytes(b"fresh")

    def _fake_extract_from_mkv(self, _mkv_files, _path_encoded_bp2):
        self.path_sparse_bp1 = None
        self.path_sparse_bp2 = None
        self._codec_bp1 = ["x265"]
        self._codec_bp2 = None
        self.temp_dirs_to_cleanup.append(str(temp_dir))
        return [str(extracted)], None

    def _fail_validation(_self):
        raise ValueError("forced MKV validation failure")

    monkeypatch.setattr(UNSPARZ, "extract_from_mkv", _fake_extract_from_mkv)
    monkeypatch.setattr(UNSPARZ, "_validate_mkv_sidecars", _fail_validation)

    try:
        UNSPARZ(
            path_sparse_bp1=None,
            path_encoded_bp1=str(mkv_file),
            stem="out",
            output_path=str(tmp_path),
            use_roi=True,
            streaming=True,
        )
    except ValueError as e:
        assert "forced MKV validation failure" in str(e)
    else:
        raise AssertionError("expected MKV validation failure")

    assert not temp_dir.exists()


def test_unsparz_streaming_run_cleans_mkv_temp_dirs_on_error(tmp_path, monkeypatch):
    """Streaming reconstruction failures must still clean MKV temp dirs."""
    temp_dir = tmp_path / "mkv_temp" / "streaming_failure"
    temp_dir.mkdir(parents=True)
    (temp_dir / "movieA_compression_level_0.mkv").write_bytes(b"fresh")

    obj = UNSPARZ.__new__(UNSPARZ)
    obj.streaming = True
    obj.temp_dirs_to_cleanup = [str(temp_dir)]

    def _fail_streaming(_self):
        raise RuntimeError("forced streaming failure")

    monkeypatch.setattr(UNSPARZ, "_run_streaming_impl", _fail_streaming)

    try:
        obj.run()
    except RuntimeError as e:
        assert "forced streaming failure" in str(e)
    else:
        raise AssertionError("expected streaming failure")

    assert not temp_dir.exists()


def test_unsparz_eager_run_cleans_mkv_temp_dirs_on_error(tmp_path, monkeypatch):
    """Legacy eager reconstruction failures must still clean MKV temp dirs."""
    temp_dir = tmp_path / "mkv_temp" / "eager_failure"
    temp_dir.mkdir(parents=True)
    (temp_dir / "movieA_compression_level_0.mkv").write_bytes(b"fresh")

    obj = UNSPARZ.__new__(UNSPARZ)
    obj.streaming = False
    obj.temp_dirs_to_cleanup = [str(temp_dir)]

    def _fail_eager(_self):
        raise RuntimeError("forced eager failure")

    monkeypatch.setattr(UNSPARZ, "_run_eager", _fail_eager)

    try:
        obj.run()
    except RuntimeError as e:
        assert "forced eager failure" in str(e)
    else:
        raise AssertionError("expected eager failure")

    assert not temp_dir.exists()


def test_find_metadata_sidecars_exact_only_ignores_unmatched_singleton(tmp_path):
    """Exact metadata matching must not assign another video's lone JSON sidecar."""
    video = tmp_path / "movieB_compression_level_0.mkv"
    video.write_bytes(b"")
    decoy = tmp_path / "movieA_compression_level_0_metadata_bp1.json"
    decoy.write_text(json.dumps([{"tags": {"Software": "movieA"}}]))

    exact_only = sparz_mod.find_metadata_sidecars_for_video(
        str(video), search_dir=str(tmp_path), allow_directory_fallback=False
    )
    assert exact_only["bp1"] is None
    assert exact_only["bp2"] is None

    fallback = sparz_mod.find_metadata_sidecars_for_video(
        str(video), search_dir=str(tmp_path)
    )
    assert fallback["bp1"] == str(decoy)


def test_load_sparse_accepts_list(tmp_path):
    """UNSPARZ.load_sparse should accept an explicit list, not only a glob string."""
    # Use the static helper to avoid any ffmpeg setup.
    paths = ["/tmp/a.npz", "/tmp/b.npz"]
    resolved = UNSPARZ._resolve_paths(paths)
    assert resolved == sorted(paths)
    glob_resolved = UNSPARZ._resolve_paths("/tmp/*.npz")  # may match nothing, that's fine
    assert isinstance(glob_resolved, list)


# ---------------------------------------------------------------------------
# Original test renamed for clarity (kept for behavior parity)
# ---------------------------------------------------------------------------

def test_full_mode_residual_does_not_zero_at_roi():
    """In full mode, residual carries non-zero values at ROI pixels too."""
    rng = np.random.default_rng(2)
    original = rng.integers(100, 4096, size=(2, 3, 3), dtype=np.uint16)
    decoded = _make_decoded_with_loss(original, scale=2)

    residual = original.astype(np.int32) - decoded.astype(np.int32)
    payload = encode_residual_to_bytes(
        residual, original_dtype="uint16", mode="full",
        codec="x265", compression_level=0,
    )
    loaded, header = decode_residual_from_bytes(payload)
    assert header["mode"] == "full"
    # At least some pixels carry a non-zero residual (this would be zero in background mode).
    assert np.any(loaded != 0)


# ---------------------------------------------------------------------------
# Codec/container helper for MKV extraction
# ---------------------------------------------------------------------------

def test_container_for_mkv_video_codec_picks_correct_extension():
    assert _container_for_mkv_video_codec("h264") == ("mp4", "mp4")
    assert _container_for_mkv_video_codec("hevc") == ("mp4", "mp4")
    assert _container_for_mkv_video_codec("av1") == ("mp4", "mp4")
    # ffv1 must not be muxed into mp4
    assert _container_for_mkv_video_codec("ffv1") == ("mkv", "matroska")
    # ProRes wants .mov
    assert _container_for_mkv_video_codec("prores") == ("mov", "mov")
    assert _container_for_mkv_video_codec("prores_ks") == ("mov", "mov")
    # Unknown / missing codec falls back to a permissive container
    assert _container_for_mkv_video_codec(None) == ("mkv", "matroska")
    assert _container_for_mkv_video_codec("something_weird") == ("mkv", "matroska")
    # Case-insensitive
    assert _container_for_mkv_video_codec("HEVC") == ("mp4", "mp4")


# ---------------------------------------------------------------------------
# MKV sidecar validation
# ---------------------------------------------------------------------------

def _make_unsparz_for_validation(use_roi, path_encoded_bp1, path_encoded_bp2=None,
                                 path_sparse_bp1=None, path_sparse_bp2=None,
                                 path_residual_bp1=None, path_residual_bp2=None,
                                 use_residuals=True):
    obj = UNSPARZ.__new__(UNSPARZ)
    obj.use_roi = use_roi
    obj.use_residuals = use_residuals
    obj.path_encoded_bp1 = list(path_encoded_bp1)
    obj.path_encoded_bp2 = list(path_encoded_bp2) if path_encoded_bp2 else None
    obj.path_sparse_bp1 = path_sparse_bp1
    obj.path_sparse_bp2 = path_sparse_bp2
    obj.path_residual_bp1 = path_residual_bp1
    obj.path_residual_bp2 = path_residual_bp2
    obj._codec_bp1 = None
    obj._codec_bp2 = None
    return obj


def test_validate_mkv_sidecars_missing_bp1_roi_raises():
    obj = _make_unsparz_for_validation(
        use_roi=True,
        path_encoded_bp1=["/tmp/movieA_compression_level_0.mp4"],
        path_sparse_bp1=None,  # advertised use_roi but no NPZ
    )
    try:
        obj._validate_mkv_sidecars()
    except ValueError as e:
        msg = str(e)
        assert "use_roi" in msg or "ROI" in msg
        assert "bp1" in msg
    else:
        raise AssertionError("expected ValueError when bp1 ROI sidecar missing")


def test_validate_mkv_sidecars_biplane_missing_bp2_roi_raises(tmp_path):
    npz1 = tmp_path / "movieA.npz"
    npz1.write_bytes(b"")
    obj = _make_unsparz_for_validation(
        use_roi=True,
        path_encoded_bp1=[str(tmp_path / "movieA_compression_level_0.mp4")],
        path_encoded_bp2=[str(tmp_path / "movieB_compression_level_0.mp4")],
        path_sparse_bp1=[str(npz1)],
        path_sparse_bp2=None,  # bp2 missing entirely
    )
    try:
        obj._validate_mkv_sidecars()
    except ValueError as e:
        msg = str(e)
        assert "bp2" in msg
        assert "ROI" in msg or "use_roi" in msg
    else:
        raise AssertionError("expected ValueError when bp2 ROI sidecar missing")


def test_validate_mkv_sidecars_partial_residual_raises(tmp_path):
    npz1 = tmp_path / "movieA.npz"; npz1.write_bytes(b"")
    npz2 = tmp_path / "movieB.npz"; npz2.write_bytes(b"")
    res1 = tmp_path / ("movieA_compression_level_0" + RESIDUAL_SUFFIX); res1.write_bytes(b"")
    obj = _make_unsparz_for_validation(
        use_roi=True,
        path_encoded_bp1=[
            str(tmp_path / "movieA_compression_level_0.mp4"),
            str(tmp_path / "movieB_compression_level_0.mp4"),
        ],
        path_sparse_bp1=[str(npz1), str(npz2)],
        # Advertised: residual exists for movieA but not for movieB (gap in list).
        path_residual_bp1=[str(res1), None],
    )
    try:
        obj._validate_mkv_sidecars()
    except ValueError as e:
        msg = str(e)
        assert "residual" in msg.lower()
        assert "bp1" in msg
    else:
        raise AssertionError("expected ValueError when residual sidecar partial")


def test_validate_mkv_sidecars_no_residuals_advertised_passes(tmp_path):
    """If no residuals were attached at all, validation should not complain."""
    npz1 = tmp_path / "movieA.npz"; npz1.write_bytes(b"")
    obj = _make_unsparz_for_validation(
        use_roi=True,
        path_encoded_bp1=[str(tmp_path / "movieA_compression_level_0.mp4")],
        path_sparse_bp1=[str(npz1)],
        path_residual_bp1=None,  # nothing advertised
    )
    obj._validate_mkv_sidecars()  # should not raise


def test_validate_mkv_sidecars_use_roi_false_skips_roi_check():
    obj = _make_unsparz_for_validation(
        use_roi=False,
        path_encoded_bp1=["/tmp/movieA_compression_level_0.mp4"],
        path_sparse_bp1=None,  # no NPZ but use_roi=False so OK
    )
    obj._validate_mkv_sidecars()  # should not raise


# ---------------------------------------------------------------------------
# Streaming reconstruction equivalence
# ---------------------------------------------------------------------------

def _make_unsparz_for_streaming():
    obj = UNSPARZ.__new__(UNSPARZ)
    obj.use_roi = True
    obj.use_residuals = True
    obj.streaming = True
    obj.chunk_size = 4
    obj.output_format = "tiff"
    return obj


def test_apply_residual_and_roi_to_chunk_matches_full_reference():
    """One-chunk apply must equal manual reference (residual + ROI patch)."""
    rng = np.random.default_rng(11)
    H, W = 4, 4
    n_frames = 3
    original = rng.integers(0, 4096, size=(n_frames, H, W), dtype=np.uint16)
    decoded = _make_decoded_with_loss(original, scale=2)

    # ROI: a few pixels per frame
    roi_mask = np.zeros_like(original, dtype=bool)
    roi_mask[:, 0, 0] = True
    roi_mask[1, 2, 3] = True

    raw_residual = original.astype(np.int32) - decoded.astype(np.int32)
    background_residual = np.where(roi_mask, np.int32(0), raw_residual)

    coords = np.array(np.where(roi_mask))
    data = original[roi_mask].astype(np.uint16)
    roi_info = {"coords": coords, "data": data, "shape": (n_frames, H, W)}

    obj = _make_unsparz_for_streaming()
    residual_iter = iter([(0, n_frames, background_residual.astype(np.int32), {})])
    out = obj._apply_residual_and_roi_to_chunk(
        decoded, 0, n_frames, residual_iter, roi_info, np.uint16, H, W
    )
    np.testing.assert_array_equal(out, original)


def test_streaming_chunked_matches_eager_apply():
    """Streaming chunk-by-chunk apply == single-shot apply on full arrays."""
    rng = np.random.default_rng(13)
    H, W = 5, 6
    T = 10
    chunk = 3
    original = rng.integers(0, 4096, size=(T, H, W), dtype=np.uint16)
    decoded = _make_decoded_with_loss(original, scale=2)

    # Background-mode residual + ROI sparse from the same data
    roi_mask = np.zeros_like(original, dtype=bool)
    roi_mask[:, 0, 0] = True
    roi_mask[3, 2, 4] = True
    roi_mask[7, 4, 5] = True
    raw_residual = original.astype(np.int32) - decoded.astype(np.int32)
    background_residual = np.where(roi_mask, np.int32(0), raw_residual)
    coords = np.array(np.where(roi_mask))
    data = original[roi_mask].astype(np.uint16)
    roi_info = {"coords": coords, "data": data, "shape": (T, H, W)}

    obj = _make_unsparz_for_streaming()

    # Build a residual iterator chunked in `chunk` frames at a time.
    def residual_iter():
        for t0 in range(0, T, chunk):
            t1 = min(T, t0 + chunk)
            yield t0, t1, background_residual[t0:t1].astype(np.int32), {}

    res_it = residual_iter()
    out_buf = np.empty_like(original)
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        out_chunk = obj._apply_residual_and_roi_to_chunk(
            decoded[t0:t1], t0, t1, res_it, roi_info, np.uint16, H, W
        )
        out_buf[t0:t1] = out_chunk

    # Reference: one-shot apply via the existing helpers.
    after_residual = apply_residuals(decoded, background_residual, np.uint16)
    expected = np.where(roi_mask, original, after_residual)
    np.testing.assert_array_equal(out_buf, expected)
    # And of course it should equal the original
    np.testing.assert_array_equal(out_buf, original)


def test_streaming_apply_raises_on_residual_misalignment():
    obj = _make_unsparz_for_streaming()
    decoded = np.zeros((3, 2, 2), dtype=np.uint16)
    # residual iterator yields a chunk that does not align
    misaligned = iter([(1, 4, np.zeros((3, 2, 2), dtype=np.int32), {})])
    try:
        obj._apply_residual_and_roi_to_chunk(
            decoded, 0, 3, misaligned, None, np.uint16, 2, 2
        )
    except ValueError as e:
        assert "misaligned" in str(e)
    else:
        raise AssertionError("expected ValueError on residual chunk misalignment")


def test_streaming_roi_only_chunk_no_residual():
    """No residual but with ROI: should still patch ROI pixels exactly."""
    obj = _make_unsparz_for_streaming()
    H, W = 3, 3
    decoded = np.full((2, H, W), 1234, dtype=np.uint16)
    roi_mask = np.zeros_like(decoded, dtype=bool)
    roi_mask[0, 1, 1] = True
    roi_mask[1, 2, 2] = True
    roi_values = np.array([5555, 7777], dtype=np.uint16)
    coords = np.array(np.where(roi_mask))
    roi_info = {"coords": coords, "data": roi_values, "shape": (2, H, W)}

    out = obj._apply_residual_and_roi_to_chunk(
        decoded, 0, 2, None, roi_info, np.uint16, H, W
    )
    # ROI pixels overwritten, others left as decoded
    assert out[0, 1, 1] == 5555
    assert out[1, 2, 2] == 7777
    # Untouched pixel stays at decoded value
    assert out[0, 0, 0] == 1234


def test_streaming_apply_raises_when_residual_exhausts_early():
    obj = _make_unsparz_for_streaming()
    decoded = np.zeros((3, 2, 2), dtype=np.uint16)
    empty_it = iter([])
    try:
        obj._apply_residual_and_roi_to_chunk(
            decoded, 0, 3, empty_it, None, np.uint16, 2, 2
        )
    except ValueError as e:
        assert "exhausted" in str(e)
    else:
        raise AssertionError("expected ValueError when residual exhausts early")


# ---------------------------------------------------------------------------
# UNSPARZ constructor: streaming flag plumbing
# ---------------------------------------------------------------------------

def test_unsparz_streaming_flag_default_true_in_signature():
    import inspect
    sig = inspect.signature(UNSPARZ.__init__)
    assert "streaming" in sig.parameters
    assert sig.parameters["streaming"].default is True


# ---------------------------------------------------------------------------
# True streaming TIFF writer
# ---------------------------------------------------------------------------

def _make_unsparz_for_tiff_writer():
    obj = UNSPARZ.__new__(UNSPARZ)
    obj.use_roi = False
    obj.use_residuals = False
    obj.streaming = True
    obj.chunk_size = 4
    obj.output_format = "tiff"
    return obj


def test_streaming_tiff_does_not_allocate_full_output(tmp_path, monkeypatch):
    """The streaming TIFF path must not allocate ``np.empty((T, H, W))``.

    We replace ``np.empty`` with a tracker and assert that no full-stack
    allocation goes through the streaming path.
    """
    obj = _make_unsparz_for_tiff_writer()
    T, H, W = 6, 3, 4
    rng = np.random.default_rng(99)
    frames = rng.integers(0, 4096, size=(T, H, W), dtype=np.uint16)

    def _chunks():
        for t0 in range(0, T, 2):
            t1 = min(T, t0 + 2)
            yield t0, t1, frames[t0:t1].copy()

    real_empty = np.empty
    saw_full_alloc = {"hit": False}

    def tracking_empty(shape, dtype=None, *args, **kwargs):
        # Flag any allocation with the full-T leading dimension.
        if isinstance(shape, tuple) and len(shape) == 3:
            if shape[0] == T and shape[1] == H and shape[2] == W:
                saw_full_alloc["hit"] = True
        return real_empty(shape, dtype=dtype, *args, **kwargs)

    monkeypatch.setattr(np, "empty", tracking_empty)
    out = tmp_path / "stream.tiff"
    obj._write_streaming_tiff(str(out), _chunks(), T, H, W, np.uint16, file_metadata={})
    assert saw_full_alloc["hit"] is False, "_write_streaming_tiff allocated np.empty((T,H,W))"

    # And the file should round-trip equal.
    import tifffile as _tf
    written = _tf.imread(str(out))
    np.testing.assert_array_equal(written, frames)


def test_streaming_tiff_preserves_basic_metadata(tmp_path):
    """Description / Software extratag / resolution should be on the first page."""
    obj = _make_unsparz_for_tiff_writer()
    T, H, W = 3, 2, 2
    frames = np.arange(T * H * W, dtype=np.uint16).reshape(T, H, W)

    def _chunks():
        yield 0, T, frames.copy()

    file_metadata = {
        # Synthetic resolution so the helper produces non-empty output
        'tags': {
            'XResolution': [4242, 1],
            'YResolution': [4242, 1],
            'ResolutionUnit': 2,
            # Software is tag 305; format_metadata_for_tifffile should turn
            # this into an extratag.
            'Software': 'SPARZ-streaming-test',
            'ImageDescription': 'SPARZ streaming smoke test',
        },
    }

    out = tmp_path / "meta.tiff"
    obj._write_streaming_tiff(str(out), _chunks(), T, H, W, np.uint16, file_metadata=file_metadata)

    import tifffile as _tf
    with _tf.TiffFile(str(out)) as tf:
        assert len(tf.pages) == T
        first = tf.pages[0]
        # Description / metadata land on page 0.
        desc = first.tags.get('ImageDescription')
        assert desc is not None
        # Resolution was passed through.
        x_res = first.tags.get('XResolution')
        y_res = first.tags.get('YResolution')
        assert x_res is not None
        assert y_res is not None


def test_streaming_tiff_uses_bigtiff_for_large_output(tmp_path, monkeypatch):
    """When projected raw size exceeds ~4 GB we should pass bigtiff=True to TiffWriter."""
    obj = _make_unsparz_for_tiff_writer()
    # Pretend we're writing a huge file by lying about the dims; we never
    # actually allocate one — TiffWriter's constructor is what we want to
    # observe, so we intercept it.
    import tifffile as _tf
    seen = {"bigtiff": None}
    real_writer = _tf.TiffWriter

    class _SpyWriter(real_writer):
        def __init__(self, path, bigtiff=False, *a, **kw):
            seen["bigtiff"] = bool(bigtiff)
            super().__init__(path, bigtiff=bigtiff, *a, **kw)

    monkeypatch.setattr(_tf, "TiffWriter", _SpyWriter)

    T = 3
    H = W = 2
    frames = np.zeros((T, H, W), dtype=np.uint16)

    def _chunks():
        yield 0, T, frames

    # Estimated size = T*H*W*2 = 24 bytes — small file.
    out_small = tmp_path / "small.tiff"
    obj._write_streaming_tiff(str(out_small), _chunks(), T, H, W, np.uint16, file_metadata={})
    assert seen["bigtiff"] is False

    # Now lie about T to force the BigTIFF branch (T*H*W*itemsize > 4GB).
    seen["bigtiff"] = None
    big_T = 2 * 1024 * 1024 * 1024  # 2G frames at 2x2x2 bytes = 8 GB projected
    big_frames = np.zeros((1, H, W), dtype=np.uint16)

    def _one_chunk():
        # Yield only one frame so we don't actually iterate billions.
        yield 0, 1, big_frames

    # Patch TiffWriter to also short-circuit the actual write to keep the
    # test fast — we only care that bigtiff was decided correctly.
    class _NopWriter:
        def __init__(self, path, bigtiff=False, *a, **kw):
            seen["bigtiff"] = bool(bigtiff)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def write(self, *a, **kw):
            pass

    monkeypatch.setattr(_tf, "TiffWriter", _NopWriter)
    out_big = tmp_path / "big.tiff"
    obj._write_streaming_tiff(str(out_big), _one_chunk(), big_T, H, W, np.uint16, file_metadata={})
    assert seen["bigtiff"] is True


# ---------------------------------------------------------------------------
# FFV1 MKV does not require ROI sidecars
# ---------------------------------------------------------------------------

def test_validate_mkv_sidecars_ffv1_without_npz_passes(tmp_path):
    """ffv1 is truly lossless; missing NPZ should be tolerated."""
    obj = _make_unsparz_for_validation(
        use_roi=True,
        path_encoded_bp1=[str(tmp_path / "movieA_compression_level_0.mkv")],
        path_sparse_bp1=None,  # no NPZ at all
    )
    obj._codec_bp1 = ["ffv1"]
    # Should not raise.
    obj._validate_mkv_sidecars()


def test_validate_mkv_sidecars_lossy_without_npz_still_raises(tmp_path):
    """Same situation but lossy codec → ROI sidecar still required."""
    obj = _make_unsparz_for_validation(
        use_roi=True,
        path_encoded_bp1=[str(tmp_path / "movieA_compression_level_0.mp4")],
        path_sparse_bp1=None,
    )
    obj._codec_bp1 = ["hevc"]
    try:
        obj._validate_mkv_sidecars()
    except ValueError as e:
        assert "ROI" in str(e) or "use_roi" in str(e)
    else:
        raise AssertionError("expected ValueError for lossy codec without NPZ")


def test_validate_mkv_sidecars_mixed_codec_per_video(tmp_path):
    """Per-video rule: ffv1 video can lack NPZ, lossy video next to it cannot."""
    npzA = tmp_path / "movieA.npz"; npzA.write_bytes(b"")
    obj = _make_unsparz_for_validation(
        use_roi=True,
        path_encoded_bp1=[
            str(tmp_path / "movieA_compression_level_0.mp4"),  # lossy: needs NPZ ✓
            str(tmp_path / "movieB_compression_level_0.mkv"),  # ffv1: NPZ optional
        ],
        path_sparse_bp1=[str(npzA), None],
    )
    obj._codec_bp1 = ["x265", "ffv1"]
    obj._validate_mkv_sidecars()  # passes

    # Now make the lossy video the one without NPZ — must raise.
    obj2 = _make_unsparz_for_validation(
        use_roi=True,
        path_encoded_bp1=[
            str(tmp_path / "movieA_compression_level_0.mp4"),  # lossy with no NPZ
            str(tmp_path / "movieB_compression_level_0.mkv"),  # ffv1
        ],
        path_sparse_bp1=[None, None],
    )
    obj2._codec_bp1 = ["x265", "ffv1"]
    try:
        obj2._validate_mkv_sidecars()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when lossy video lacks NPZ")


# ---------------------------------------------------------------------------
# use_residuals=False skips residual validation
# ---------------------------------------------------------------------------

def test_validate_mkv_sidecars_skips_residual_check_when_use_residuals_false(tmp_path):
    """Partial residual list must not raise when the user disabled residuals."""
    npz1 = tmp_path / "movieA.npz"; npz1.write_bytes(b"")
    npz2 = tmp_path / "movieB.npz"; npz2.write_bytes(b"")
    res1 = tmp_path / ("movieA_compression_level_0" + RESIDUAL_SUFFIX); res1.write_bytes(b"")

    obj = _make_unsparz_for_validation(
        use_roi=True,
        path_encoded_bp1=[
            str(tmp_path / "movieA_compression_level_0.mp4"),
            str(tmp_path / "movieB_compression_level_0.mp4"),
        ],
        path_sparse_bp1=[str(npz1), str(npz2)],
        path_residual_bp1=[str(res1), None],  # advertised but partial
    )
    obj._codec_bp1 = ["x265", "x265"]
    obj.use_residuals = False  # opt out
    obj._validate_mkv_sidecars()  # must not raise


# ---------------------------------------------------------------------------
# Biplane count consistency
# ---------------------------------------------------------------------------

def test_validate_count_consistency_biplane_mismatch_raises():
    obj = UNSPARZ.__new__(UNSPARZ)
    obj.path_encoded_bp1 = ["/tmp/a1.mp4", "/tmp/a2.mp4"]
    obj.path_encoded_bp2 = ["/tmp/b1.mp4"]  # only 1
    obj.use_roi = False
    try:
        obj._validate_count_consistency()
    except ValueError as e:
        assert "BP1" in str(e) and "BP2" in str(e)
    else:
        raise AssertionError("expected ValueError on biplane count mismatch")


def test_validate_count_consistency_matched_passes():
    obj = UNSPARZ.__new__(UNSPARZ)
    obj.path_encoded_bp1 = ["/tmp/a1.mp4", "/tmp/a2.mp4"]
    obj.path_encoded_bp2 = ["/tmp/b1.mp4", "/tmp/b2.mp4"]
    obj.use_roi = False
    obj._validate_count_consistency()  # no raise


def test_validate_count_consistency_single_plane_passes():
    obj = UNSPARZ.__new__(UNSPARZ)
    obj.path_encoded_bp1 = ["/tmp/a1.mp4"]
    obj.path_encoded_bp2 = None
    obj.use_roi = False
    obj._validate_count_consistency()  # no raise


# ---------------------------------------------------------------------------
# _roi_npz_path_for_index — list with None gaps
# ---------------------------------------------------------------------------

def test_roi_npz_path_for_index_list_with_none():
    """Per-video lists must be indexed positionally, not sorted (which would crash on None)."""
    obj = UNSPARZ.__new__(UNSPARZ)
    paths = ["/tmp/a.npz", None]
    assert obj._roi_npz_path_for_index(paths, 0) == "/tmp/a.npz"
    assert obj._roi_npz_path_for_index(paths, 1) is None
    # Out-of-range index returns None instead of crashing.
    assert obj._roi_npz_path_for_index(paths, 5) is None


def test_roi_npz_path_for_index_none_input():
    obj = UNSPARZ.__new__(UNSPARZ)
    assert obj._roi_npz_path_for_index(None, 0) is None


def test_roi_npz_path_for_index_glob_string_resolves_and_sorts(tmp_path):
    obj = UNSPARZ.__new__(UNSPARZ)
    a = tmp_path / "a.npz"; a.write_bytes(b"")
    b = tmp_path / "b.npz"; b.write_bytes(b"")
    pattern = str(tmp_path / "*.npz")
    # Glob resolution sorts alphabetically.
    assert obj._roi_npz_path_for_index(pattern, 0).endswith("a.npz")
    assert obj._roi_npz_path_for_index(pattern, 1).endswith("b.npz")


def test_resolve_paths_filters_none_entries():
    """_resolve_paths is used for flat collections; None entries should be filtered."""
    paths = ["/tmp/a.npz", None, "/tmp/b.npz"]
    resolved = UNSPARZ._resolve_paths(paths)
    assert resolved == sorted(["/tmp/a.npz", "/tmp/b.npz"])


# ---------------------------------------------------------------------------
# FFV1 + streaming=False eager mode
# ---------------------------------------------------------------------------

def _make_unsparz_for_eager_adjustment(use_roi=True, codec_bp1=None, codec_bp2=None,
                                       sparse_bp1=None, sparse_bp2=None,
                                       has_bp2=False):
    obj = UNSPARZ.__new__(UNSPARZ)
    obj.use_roi = use_roi
    obj._mkv_extracted = True
    obj.path_encoded_bp1 = ["/tmp/a.mkv"] if codec_bp1 and len(codec_bp1) == 1 else \
                            [f"/tmp/a{i}.mkv" for i in range(len(codec_bp1 or []))]
    obj.path_encoded_bp2 = ["/tmp/b.mkv"] if has_bp2 and codec_bp2 and len(codec_bp2) == 1 else \
                            ([f"/tmp/b{i}.mkv" for i in range(len(codec_bp2 or []))] if has_bp2 else None)
    obj.path_sparse_bp1 = sparse_bp1
    obj.path_sparse_bp2 = sparse_bp2
    obj._codec_bp1 = codec_bp1
    obj._codec_bp2 = codec_bp2
    return obj


def test_eager_adjust_disables_roi_when_all_ffv1_no_npz():
    obj = _make_unsparz_for_eager_adjustment(
        use_roi=True, codec_bp1=["ffv1", "ffv1"], sparse_bp1=None,
    )
    obj._adjust_roi_for_eager_mode()
    assert obj.use_roi is False


def test_eager_adjust_disables_roi_biplane_all_ffv1():
    obj = _make_unsparz_for_eager_adjustment(
        use_roi=True,
        codec_bp1=["ffv1"], sparse_bp1=None,
        codec_bp2=["ffv1"], sparse_bp2=None, has_bp2=True,
    )
    obj._adjust_roi_for_eager_mode()
    assert obj.use_roi is False


def test_eager_adjust_raises_on_mixed_lossy_ffv1_with_gaps(tmp_path):
    """Lossy + ffv1 with partial NPZ in eager mode → raise pointing to streaming=True."""
    npz = tmp_path / "lossy.npz"; npz.write_bytes(b"")
    obj = _make_unsparz_for_eager_adjustment(
        use_roi=True,
        codec_bp1=["x265", "ffv1"],
        sparse_bp1=[str(npz), None],
    )
    try:
        obj._adjust_roi_for_eager_mode()
    except ValueError as e:
        assert "streaming=True" in str(e)
    else:
        raise AssertionError("expected ValueError for mixed lossy+ffv1 gaps in eager mode")


def test_eager_adjust_no_change_when_all_npz_present(tmp_path):
    npz1 = tmp_path / "a.npz"; npz1.write_bytes(b"")
    npz2 = tmp_path / "b.npz"; npz2.write_bytes(b"")
    obj = _make_unsparz_for_eager_adjustment(
        use_roi=True,
        codec_bp1=["x265", "x265"],
        sparse_bp1=[str(npz1), str(npz2)],
    )
    obj._adjust_roi_for_eager_mode()
    assert obj.use_roi is True
    assert obj.path_sparse_bp1 == [str(npz1), str(npz2)]


# ---------------------------------------------------------------------------
# load_original_metadata accepts list with None entries
# ---------------------------------------------------------------------------

def _make_unsparz_for_metadata(tmp_path, sparse_bp1, has_bp2=False, sparse_bp2=None):
    obj = UNSPARZ.__new__(UNSPARZ)
    obj.path_sparse_bp1 = sparse_bp1
    obj.path_sparse_bp2 = sparse_bp2
    obj.path_encoded_bp1 = ["/tmp/a.mkv"]
    obj.path_encoded_bp2 = ["/tmp/b.mkv"] if has_bp2 else None
    obj.output_path = str(tmp_path) + "/"
    obj.stem = "test"
    return obj


def test_load_original_metadata_accepts_list_with_none(tmp_path):
    """Metadata loader must read zstd-compressed NPZ metadata from list inputs."""
    # Build a real NPZ with embedded metadata so load_metadata_from_npz can read it.
    import io as _io
    import zstandard as zstd
    metadata_dict = {"tags": {"Software": "fixture-software"}, "is_encoded": False}
    payload = {
        "encoding": np.array("delta"),
        "data": np.array([1, 2, 3], dtype=np.int16),
        "delta_coords": np.array([[0, 1, 2]], dtype=np.int32),
        "shape": np.array([3, 3, 3], dtype=np.int64),
        "metadata_json": np.array(json.dumps(metadata_dict)),
    }
    buf = _io.BytesIO()
    np.savez(buf, **payload)
    cctx = zstd.ZstdCompressor(level=3)
    compressed = cctx.compress(buf.getvalue())
    npz_path = tmp_path / "movieA.npz"
    npz_path.write_bytes(compressed)

    obj = _make_unsparz_for_metadata(tmp_path, sparse_bp1=[str(npz_path), None])
    paths = obj._collect_npz_paths(obj.path_sparse_bp1)
    assert paths == [str(npz_path)]
    loaded = obj.load_metadata_from_npz(str(npz_path))
    assert loaded["tags"]["Software"] == "fixture-software"

    metadata_bp1, metadata_bp2 = obj.load_original_metadata()
    assert metadata_bp2 is None
    assert metadata_bp1[0]["tags"]["Software"] == "fixture-software"


def test_load_original_metadata_prefers_attached_json_sidecar(tmp_path):
    """FFV1 MKVs can carry metadata JSON even when there is no ROI NPZ."""
    stale_path = tmp_path / "test_metadata_bp1.json"
    stale_path.write_text(json.dumps([{
        "tags": {"Software": "stale-output-json"},
        "is_encoded": False,
    }]))
    metadata_path = tmp_path / "movieA_compression_level_0_metadata_bp1.json"
    metadata_path.write_text(json.dumps([{
        "tags": {"Software": "ffv1-sidecar"},
        "is_encoded": False,
    }]))

    obj = _make_unsparz_for_metadata(tmp_path, sparse_bp1=None)
    obj.path_metadata_bp1 = [str(metadata_path)]
    obj.path_metadata_bp2 = None

    metadata_bp1, metadata_bp2 = obj.load_original_metadata()
    assert metadata_bp2 is None
    assert metadata_bp1[0]["tags"]["Software"] == "ffv1-sidecar"


def test_attached_json_metadata_preserves_video_slots(tmp_path):
    """A missing sidecar for video 0 must not shift video 1 metadata into slot 0."""
    metadata_path = tmp_path / "movieB_compression_level_0_metadata_bp1.json"
    metadata_path.write_text(json.dumps([{
        "tags": {"Software": "video-1-sidecar"},
        "is_encoded": False,
    }]))

    obj = _make_unsparz_for_metadata(tmp_path, sparse_bp1=None)
    obj.path_metadata_bp1 = [None, str(metadata_path)]
    obj.path_metadata_bp2 = None

    metadata_bp1, _metadata_bp2 = obj.load_original_metadata()
    assert metadata_bp1[0] == {}
    assert metadata_bp1[1]["tags"]["Software"] == "video-1-sidecar"


def test_write_mkv_metadata_sidecar_matches_video_stem(tmp_path):
    """Per-video JSON sidecars are named so MKV extraction can match them by stem."""
    obj = SPARZIP.__new__(SPARZIP)
    obj.extract_metadata_flag = True
    obj.output_path = str(tmp_path) + "/"
    obj.metadata_bp1 = [{"tags": {"Software": "bp1-meta"}}]
    obj.metadata_bp2 = None

    dataset_json = tmp_path / "movieA_metadata_bp1.json"
    dataset_json.write_text(json.dumps([{"tags": {"Software": "dataset-meta"}}]))

    path = obj._write_mkv_metadata_sidecar("movieA_compression_level_0", "bp1", 0)
    assert path.endswith("movieA_compression_level_0_metadata_bp1.json")
    assert os.path.exists(path)
    assert json.loads(dataset_json.read_text())[0]["tags"]["Software"] == "dataset-meta"

    matched = sparz_mod.find_metadata_sidecars_for_video(
        str(tmp_path / "movieA_compression_level_0.mkv"),
        search_dir=str(tmp_path),
    )
    assert matched["bp1"] == path
    assert matched["bp2"] is None


def test_collect_npz_paths_handles_glob(tmp_path):
    a = tmp_path / "a.npz"; a.write_bytes(b"")
    obj = UNSPARZ.__new__(UNSPARZ)
    obj.output_path = str(tmp_path) + "/"
    paths = obj._collect_npz_paths(str(tmp_path / "*.npz"))
    assert paths == [str(a)]


def test_collect_npz_paths_falls_back_to_output_path(tmp_path):
    a = tmp_path / "a.npz"; a.write_bytes(b"")
    obj = UNSPARZ.__new__(UNSPARZ)
    obj.output_path = str(tmp_path) + "/"
    paths = obj._collect_npz_paths(None)
    assert paths == [str(a)]


# ---------------------------------------------------------------------------
# Streaming TIFF Software metadata
# ---------------------------------------------------------------------------

def test_streaming_tiff_software_uses_software_kwarg(tmp_path):
    """Software string must be written via tifffile's software= kwarg, not extratag 305."""
    obj = _make_unsparz_for_tiff_writer()
    T, H, W = 2, 2, 2
    frames = np.zeros((T, H, W), dtype=np.uint16)

    def _chunks():
        yield 0, T, frames

    file_metadata = {
        "tags": {
            "Software": "TestSoftware",  # format_metadata adds " -> UNSPARZ" suffix
        },
    }

    out = tmp_path / "sw.tiff"
    obj._write_streaming_tiff(str(out), _chunks(), T, H, W, np.uint16, file_metadata)

    import tifffile as _tf
    with _tf.TiffFile(str(out)) as tf:
        page0 = tf.pages[0]
        sw_tag = page0.tags.get("Software")
        assert sw_tag is not None
        sw_value = sw_tag.value if hasattr(sw_tag, "value") else sw_tag
        # Must NOT be tifffile.py default; must contain our test string.
        assert "tifffile" not in sw_value.lower(), \
            f"Software tag fell through to tifffile default: {sw_value!r}"
        assert "TestSoftware" in sw_value


def test_streaming_tiff_software_warning_path_not_used(tmp_path, monkeypatch):
    """Confirm extratags passed to tifffile no longer include tag 305."""
    obj = _make_unsparz_for_tiff_writer()
    T, H, W = 1, 2, 2
    frames = np.zeros((T, H, W), dtype=np.uint16)

    def _chunks():
        yield 0, T, frames

    file_metadata = {"tags": {"Software": "TestSoftware"}}

    import tifffile as _tf
    real_writer_cls = _tf.TiffWriter

    seen = {"page0_extratags": None, "page0_software": None}

    class _SpyWriter:
        def __init__(self, *a, **kw):
            self._inner = real_writer_cls(*a, **kw)
        def __enter__(self):
            self._inner.__enter__()
            return self
        def __exit__(self, *a):
            return self._inner.__exit__(*a)
        def write(self, frame, **kwargs):
            if seen["page0_extratags"] is None:
                seen["page0_extratags"] = list(kwargs.get("extratags") or [])
                seen["page0_software"] = kwargs.get("software")
            return self._inner.write(frame, **kwargs)

    monkeypatch.setattr(_tf, "TiffWriter", _SpyWriter)
    out = tmp_path / "sw.tiff"
    obj._write_streaming_tiff(str(out), _chunks(), T, H, W, np.uint16, file_metadata)

    # tag 305 must NOT appear in extratags.
    codes = [et[0] for et in seen["page0_extratags"]]
    assert 305 not in codes
    # Software was passed via the dedicated kwarg.
    assert seen["page0_software"] is not None
    assert "TestSoftware" in seen["page0_software"]


class _FallbackMonkeypatch:
    """Minimal pytest-monkeypatch substitute for the no-pytest fallback runner."""
    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value, raising=True):
        self._undo.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def teardown(self):
        for target, name, original in reversed(self._undo):
            setattr(target, name, original)
        self._undo.clear()


class _FallbackCapsys:
    """Minimal pytest-capsys substitute capturing stdout."""
    def __init__(self):
        import io as _io
        self._buf = _io.StringIO()
        self._original = None

    def start(self):
        import sys as _sys
        self._original = _sys.stdout
        _sys.stdout = self._buf

    def stop(self):
        import sys as _sys
        if self._original is not None:
            _sys.stdout = self._original

    def readouterr(self):
        text = self._buf.getvalue()
        self._buf.seek(0)
        self._buf.truncate(0)

        class _R:
            pass
        r = _R()
        r.out = text
        r.err = ""
        return r


if __name__ == "__main__":
    if pytest is not None:
        raise SystemExit(pytest.main([__file__, "-v"]))
    # Fallback: run tests manually without pytest
    import inspect
    import tempfile
    failures = []
    test_funcs = [obj for name, obj in list(globals().items())
                  if name.startswith("test_") and callable(obj)]
    for fn in test_funcs:
        sig = inspect.signature(fn)
        kwargs = {}
        td = None
        mp = None
        cs = None
        if "tmp_path" in sig.parameters:
            td = tempfile.TemporaryDirectory()
            from pathlib import Path
            kwargs["tmp_path"] = Path(td.name)
        if "monkeypatch" in sig.parameters:
            mp = _FallbackMonkeypatch()
            kwargs["monkeypatch"] = mp
        if "capsys" in sig.parameters:
            cs = _FallbackCapsys()
            cs.start()
            kwargs["capsys"] = cs
        outcome = None
        err = None
        try:
            fn(**kwargs)
            outcome = "PASS"
        except Exception as e:
            outcome = "FAIL"
            err = e
        finally:
            if cs is not None:
                cs.stop()
            if mp is not None:
                mp.teardown()
            if td is not None:
                td.cleanup()
        # Print outcome AFTER capsys has been restored so it actually reaches stdout.
        if outcome == "PASS":
            print(f"PASS  {fn.__name__}")
        else:
            failures.append((fn.__name__, err))
            print(f"FAIL  {fn.__name__}: {err}")
    if failures:
        for name, err in failures:
            print(f"  -> {name}: {err}")
        raise SystemExit(1)
    print(f"\nAll {len(test_funcs)} tests passed.")
    raise SystemExit(0)
