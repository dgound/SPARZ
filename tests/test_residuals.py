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
    load_residual_sidecar,
    apply_residuals,
    get_residual_path_for_video,
    find_matching_sidecars_for_video,
    RESIDUAL_SUFFIX,
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
        if "tmp_path" in sig.parameters:
            td = tempfile.TemporaryDirectory()
            from pathlib import Path
            kwargs["tmp_path"] = Path(td.name)
        try:
            fn(**kwargs)
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failures.append((fn.__name__, e))
            print(f"FAIL  {fn.__name__}: {e}")
        finally:
            if td is not None:
                td.cleanup()
    if failures:
        for name, err in failures:
            print(f"  -> {name}: {err}")
        raise SystemExit(1)
    print(f"\nAll {len(test_funcs)} tests passed.")
    raise SystemExit(0)
