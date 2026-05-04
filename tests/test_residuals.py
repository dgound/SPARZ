"""Helper-level tests for the SPARZ residual sidecar layer."""
import os
import sys

import numpy as np

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


def test_wire_sidecars_warns_on_missing_npz(tmp_path, capsys):
    obj = _make_unsparz_for_extract()
    records = [
        {"video": str(tmp_path / "a.mp4"), "npz": str(tmp_path / "a.npz"), "residual": None},
        {"video": str(tmp_path / "b.mp4"), "npz": None, "residual": None},
    ]
    obj._wire_sidecars_from_records(records, plane="bp1")
    out = capsys.readouterr().out if hasattr(capsys, "readouterr") else ""
    assert obj.path_sparse_bp1 is None  # partial NPZ coverage -> disabled
    if out:
        assert "1/2 NPZ" in out or "partial" in out.lower() or "warning" in out.lower()


def test_extract_one_mkv_uses_stem_matched_sidecars(tmp_path):
    """_extract_one_mkv picks sidecars whose stems match the extracted video.

    We cannot run ffmpeg/ffprobe in unit tests, so we stub them out: the codec
    probe returns ``None`` (which maps to ``.mkv``), the attachment extractor
    is a no-op, and we pre-create the extracted ``.mkv`` so the "already
    extracted" branch fires instead of invoking real ffmpeg.
    """
    obj = _make_unsparz_for_extract()
    mkv_dir = tmp_path / "mkvs"
    mkv_dir.mkdir()
    mkv_file = mkv_dir / "movieA_compression_level_0.mkv"
    mkv_file.write_bytes(b"")

    temp_dir = mkv_dir / "mkv_temp"
    temp_dir.mkdir()
    # Codec probe falls back to ``("mkv", "matroska")`` for unknown codecs,
    # so the extracted video lands here as ``.mkv`` (not ``.mp4``).
    video_path = temp_dir / "movieA_compression_level_0.mkv"
    video_path.write_bytes(b"")
    (temp_dir / "movieA.npz").write_bytes(b"")
    (temp_dir / ("movieA_compression_level_0" + RESIDUAL_SUFFIX)).write_bytes(b"")
    # Decoy sidecars with a different stem must NOT be picked up.
    (temp_dir / "movieB.npz").write_bytes(b"")
    (temp_dir / ("movieB_compression_level_0" + RESIDUAL_SUFFIX)).write_bytes(b"")

    # Stub out the ffmpeg pieces so no subprocess is launched.
    import types
    obj._extract_mkv_attachments = types.MethodType(lambda self, m, t: (0, 0), obj)

    import SPARZ as _sparz
    real_probe = _sparz.ffmpeg.probe
    _sparz.ffmpeg.probe = lambda *_a, **_kw: {"streams": []}
    try:
        record = obj._extract_one_mkv(str(mkv_file))
    finally:
        _sparz.ffmpeg.probe = real_probe

    assert record["video"] == str(video_path)
    assert record["npz"].endswith("movieA.npz")
    assert os.path.basename(record["residual"]).startswith("movieA")


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
                                 path_residual_bp1=None, path_residual_bp2=None):
    obj = UNSPARZ.__new__(UNSPARZ)
    obj.use_roi = use_roi
    obj.path_encoded_bp1 = list(path_encoded_bp1)
    obj.path_encoded_bp2 = list(path_encoded_bp2) if path_encoded_bp2 else None
    obj.path_sparse_bp1 = path_sparse_bp1
    obj.path_sparse_bp2 = path_sparse_bp2
    obj.path_residual_bp1 = path_residual_bp1
    obj.path_residual_bp2 = path_residual_bp2
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
