#%%
import numpy as np
import tifffile
from tifffile import TiffFile
from skimage.feature import peak_local_max
import glob
import time
import sparse
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
import dask.array as da
import dask_image.imread
from dask import delayed
import dask
from dask import compute
import gc
from tqdm import tqdm
import os
import io
import re
import pandas as pd
import json
from dask import delayed
import ffmpeg
from dask.diagnostics import ProgressBar
# from concurrent.futures import ThreadPoolExecutor, as_completed
import zstandard as zstd
from statsmodels.stats.power import TTestIndPower
import av
import concurrent.futures
from typing import Any, Dict, List, Union
# import joblib

#%%
def serialize_metadata_value(value: Any) -> Any:
    """
    Convert metadata values to JSON-serializable format with type checking.
    
    Args:
        value: The metadata value to serialize
        
    Returns:
        JSON-serializable version of the value or None if not serializable
    """
    # Handle None
    if value is None:
        return None
    
    # Handle basic JSON-serializable types
    if isinstance(value, (str, int, float, bool)):
        return value
    
    # Handle numpy types
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, np.ndarray):
        # Special handling for LUTs and other important arrays
        # LUTs are typically (3, 256) for RGB lookup tables or similar structures
        # We should preserve these as they're important for image processing
        
        # Always convert arrays to lists for JSON serialization
        # Store with metadata about the original type
        return {
            '__numpy_array__': True,
            'shape': value.shape,
            'dtype': str(value.dtype),
            'data': value.tolist()
        }
    
    # Handle lists and tuples
    elif isinstance(value, (list, tuple)):
        serialized_list = []
        for item in value:
            serialized_item = serialize_metadata_value(item)
            if serialized_item is not None:
                serialized_list.append(serialized_item)
        return serialized_list
    
    # Handle dictionaries
    elif isinstance(value, dict):
        serialized_dict = {}
        for k, v in value.items():
            # Ensure key is string
            key_str = str(k)
            serialized_value = serialize_metadata_value(v)
            if serialized_value is not None:
                serialized_dict[key_str] = serialized_value
        return serialized_dict
    
    # Handle bytes
    elif isinstance(value, bytes):
        try:
            # Try to decode as UTF-8 string
            return value.decode('utf-8')
        except UnicodeDecodeError:
            # If not UTF-8, convert to base64 for preservation
            import base64
            return {
                '__bytes__': True,
                'data': base64.b64encode(value).decode('ascii')
            }
    
    # Handle other types by converting to string representation
    else:
        try:
            str_value = str(value)
            # Only keep reasonable length strings
            if len(str_value) <= 10000:  # Arbitrary limit
                return f"<{type(value).__name__}>: {str_value}"
            else:
                return f"<{type(value).__name__}_length_{len(str_value)}>"
        except:
            return f"<non_serializable_{type(value).__name__}>"

def serialize_metadata(metadata_list: List[Dict]) -> List[Dict]:
    """
    Serialize a list of metadata dictionaries for JSON storage.
    
    Args:
        metadata_list: List of metadata dictionaries from SPARZIP
        
    Returns:
        List of JSON-serializable metadata dictionaries
    """
    serialized_list = []
    
    for i, metadata in enumerate(metadata_list):
        if not isinstance(metadata, dict):
            print(f"Warning: metadata[{i}] is not a dictionary, skipping")
            continue
            
        serialized_metadata = {}
        
        for key, value in metadata.items():
            # Ensure key is string
            key_str = str(key)
            
            # Check and serialize the value
            serialized_value = serialize_metadata_value(value)
            
            if serialized_value is not None:
                serialized_metadata[key_str] = serialized_value
            else:
                print(f"Warning: Could not serialize metadata[{i}]['{key}'] of type {type(value)}")
        
        serialized_list.append(serialized_metadata)
    
    return serialized_list

def restore_numeric_values(metadata: Dict) -> Dict:
    """
    Restore numeric values that may have been converted during JSON serialization.
    Call this before using metadata in format_metadata_for_tifffile().
    Also restores numpy arrays and bytes that were serialized.
    """
    restored_metadata = {}
    
    for key, value in metadata.items():
        if isinstance(value, dict):
            # Check if it's a serialized numpy array
            if value.get('__numpy_array__') == True:
                # Restore the numpy array from the serialized format
                import numpy as np
                shape = tuple(value['shape'])
                dtype = np.dtype(value['dtype'])
                data = np.array(value['data'], dtype=dtype)
                restored_metadata[key] = data.reshape(shape)
            # Check if it's serialized bytes
            elif value.get('__bytes__') == True:
                # Restore bytes from base64
                import base64
                restored_metadata[key] = base64.b64decode(value['data'])
            else:
                # Recursively restore nested dictionaries
                restored_metadata[key] = restore_numeric_values(value)
        elif isinstance(value, list):
            # Restore lists that might contain metadata
            restored_list = []
            for item in value:
                if isinstance(item, dict):
                    # Check if this dict is a serialized numpy array or bytes
                    if item.get('__numpy_array__') == True:
                        import numpy as np
                        shape = tuple(item['shape'])
                        dtype = np.dtype(item['dtype'])
                        data = np.array(item['data'], dtype=dtype)
                        restored_list.append(data.reshape(shape))
                    elif item.get('__bytes__') == True:
                        import base64
                        restored_list.append(base64.b64decode(item['data']))
                    else:
                        restored_list.append(restore_numeric_values(item))
                else:
                    restored_list.append(item)
            restored_metadata[key] = restored_list
        elif isinstance(value, str) and value.startswith('<') and value.endswith('>') and not value.startswith('<?xml'):
            # Handle old serialization format placeholders
            # These are from NPZ files created before the serialization fix
            # Silently skip them without printing warnings for known patterns
            if 'numpy_array_shape' in value or 'bytes_length' in value:
                # This is expected from old NPZ files, don't warn
                continue
            else:
                # Only warn for unexpected placeholder formats
                print(f'Skipping non-serializable metadata: {key} = {value}')
                continue
        else:
            # Keep the value as-is
            restored_metadata[key] = value
    
    return restored_metadata

#%%
# ============================================================
# Residual sidecar format
# ============================================================
# Binary container for the optional exact-reconstruction residual
# layer used by near-lossless video codecs.
#
# Two on-disk versions exist; readers support both, writers default to v2.
#
# Common header (little-endian):
#   magic                8 bytes   b"SPRZRS01"
#   version              uint8     1 = single-blob, 2 = chunk-indexed
#   mode                 uint8     0=full, 1=background
#   residual_dtype       uint8     0=int16, 1=int32
#   flags                uint8     bit0=delta, bit1=byte-shuffle
#   shape                3 x int64 (T, H, W)
#   orig_dtype_len       uint16    + UTF-8 bytes (e.g. "uint16")
#   codec_len            uint16    + UTF-8 bytes
#   compression_level    int32
#
# v1 body: a single zstd frame containing residual bytes (after optional
# delta + shuffle over the whole array). Used by helper-level encoders for
# small in-memory data.
#
# v2 body: a chunked stream supporting streaming write/read without ever
# materialising the full residual.
#   chunk_size           int32     frames per chunk (last may be smaller)
#   n_chunks             int32
#
#   If flags bit2 ("indexed") is set, an index table follows immediately so
#   readers can jump to any chunk in O(1):
#       n_chunks repetitions of:
#           start_frame      uint32
#           n_frames         uint32
#           compressed_off   uint64   absolute byte offset into the file
#           compressed_len   uint64
#       then the concatenated zstd-compressed payloads in chunk order.
#
#   Without bit2 (legacy v2), payloads are interleaved with their lengths:
#       n_chunks repetitions of (n_frames uint32, compressed_len uint64,
#                                payload bytes).
#
#   Each payload (in either layout) holds the chunk bytes after optional
#   per-chunk delta + per-chunk shuffle as indicated by bits 0/1.
RESIDUAL_MAGIC = b"SPRZRS01"
RESIDUAL_VERSION = 1
RESIDUAL_VERSION_V2 = 2
RESIDUAL_MODE_FULL = 0
RESIDUAL_MODE_BACKGROUND = 1
RESIDUAL_DTYPE_INT16 = 0
RESIDUAL_DTYPE_INT32 = 1
RESIDUAL_FLAG_DELTA = 0x01
RESIDUAL_FLAG_SHUFFLE = 0x02
RESIDUAL_FLAG_INDEXED = 0x04
RESIDUAL_INDEX_ENTRY_SIZE = 24  # 4 + 4 + 8 + 8
RESIDUAL_SUFFIX = "_residual.zst"


def _residual_shuffle_bytes(data: bytes, itemsize: int) -> bytes:
    arr = np.frombuffer(data, dtype=np.uint8)
    n = len(arr) // itemsize
    if n == 0:
        return data
    return arr[:n * itemsize].reshape(n, itemsize).T.tobytes()


def _residual_unshuffle_bytes(data: bytes, itemsize: int) -> bytes:
    arr = np.frombuffer(data, dtype=np.uint8)
    n = len(arr) // itemsize
    if n == 0:
        return data
    return arr[:n * itemsize].reshape(itemsize, n).T.tobytes()


def encode_residual_to_bytes(residual, original_dtype, mode, codec, compression_level,
                             use_delta=True, use_shuffle=True, zstd_level=19):
    """Encode a (T,H,W) residual array into a sidecar byte string."""
    if mode not in ("full", "background"):
        raise ValueError(f"unknown residual mode: {mode}")
    residual = np.asarray(residual)
    if residual.ndim != 3:
        raise ValueError("residual must be 3D (T, H, W)")

    res32 = residual.astype(np.int32, copy=False)
    if res32.size and int(res32.min()) >= -32768 and int(res32.max()) <= 32767:
        res = res32.astype(np.int16, copy=False)
        dtype_code = RESIDUAL_DTYPE_INT16
        itemsize = 2
    else:
        res = res32
        dtype_code = RESIDUAL_DTYPE_INT32
        itemsize = 4

    if use_delta and res.shape[0] > 1:
        delta = np.empty_like(res)
        delta[0] = res[0]
        delta[1:] = res[1:] - res[:-1]
        body = delta.tobytes()
    else:
        use_delta = False
        body = res.tobytes()

    if use_shuffle:
        body = _residual_shuffle_bytes(body, itemsize)

    cctx = zstd.ZstdCompressor(level=zstd_level)
    compressed = cctx.compress(body)

    flags = 0
    if use_delta:
        flags |= 0x01
    if use_shuffle:
        flags |= 0x02
    mode_code = RESIDUAL_MODE_BACKGROUND if mode == "background" else RESIDUAL_MODE_FULL

    orig_dtype_bytes = np.dtype(original_dtype).name.encode("utf-8")
    codec_bytes = str(codec).encode("utf-8")

    header = bytearray()
    header += RESIDUAL_MAGIC
    header += bytes([RESIDUAL_VERSION, mode_code, dtype_code, flags])
    header += np.array(res.shape, dtype=np.int64).tobytes()
    header += len(orig_dtype_bytes).to_bytes(2, "little")
    header += orig_dtype_bytes
    header += len(codec_bytes).to_bytes(2, "little")
    header += codec_bytes
    header += int(compression_level).to_bytes(4, "little", signed=True)

    return bytes(header) + compressed


def decode_residual_from_bytes(buf):
    """Inverse of encode_residual_to_bytes. Returns (residual_array, header_dict)."""
    pos = 0
    magic = bytes(buf[pos:pos + 8]); pos += 8
    if magic != RESIDUAL_MAGIC:
        raise ValueError(f"bad residual magic: {magic!r}")
    version = buf[pos]; pos += 1
    mode_code = buf[pos]; pos += 1
    dtype_code = buf[pos]; pos += 1
    flags = buf[pos]; pos += 1
    shape = tuple(int(x) for x in np.frombuffer(buf[pos:pos + 24], dtype=np.int64))
    pos += 24
    n = int.from_bytes(buf[pos:pos + 2], "little"); pos += 2
    orig_dtype = bytes(buf[pos:pos + n]).decode("utf-8"); pos += n
    n = int.from_bytes(buf[pos:pos + 2], "little"); pos += 2
    codec = bytes(buf[pos:pos + n]).decode("utf-8"); pos += n
    compression_level = int.from_bytes(buf[pos:pos + 4], "little", signed=True); pos += 4

    use_delta = bool(flags & 0x01)
    use_shuffle = bool(flags & 0x02)
    mode = "background" if mode_code == RESIDUAL_MODE_BACKGROUND else "full"
    res_dtype = np.int16 if dtype_code == RESIDUAL_DTYPE_INT16 else np.int32
    itemsize = np.dtype(res_dtype).itemsize

    dctx = zstd.ZstdDecompressor()
    body = dctx.decompress(bytes(buf[pos:]))

    if use_shuffle:
        body = _residual_unshuffle_bytes(body, itemsize)

    res = np.frombuffer(body, dtype=res_dtype).reshape(shape).copy()
    if use_delta and res.shape[0] > 1:
        res = np.cumsum(res, axis=0).astype(res_dtype, copy=False)

    header = {
        "version": int(version),
        "mode": mode,
        "shape": shape,
        "original_dtype": orig_dtype,
        "residual_dtype": np.dtype(res_dtype).name,
        "codec": codec,
        "compression_level": compression_level,
        "use_delta": use_delta,
        "use_shuffle": use_shuffle,
    }
    return res, header


def save_residual_sidecar(path, residual, original_dtype, mode, codec, compression_level,
                          use_delta=True, use_shuffle=True, zstd_level=19):
    data = encode_residual_to_bytes(
        residual, original_dtype, mode, codec, compression_level,
        use_delta=use_delta, use_shuffle=use_shuffle, zstd_level=zstd_level,
    )
    with open(path, "wb") as f:
        f.write(data)


def load_residual_sidecar(path):
    """Read a residual sidecar fully into memory, supporting both v1 and v2."""
    with open(path, "rb") as f:
        first = f.read(9)
    if first[:8] != RESIDUAL_MAGIC:
        raise ValueError(f"bad residual magic: {first[:8]!r}")
    version = first[8]
    if version == RESIDUAL_VERSION:
        with open(path, "rb") as f:
            return decode_residual_from_bytes(f.read())
    if version == RESIDUAL_VERSION_V2:
        chunks = []
        header = None
        for start, end, chunk, header in iter_residual_chunks_from_path(path):
            chunks.append(chunk)
        if not chunks:
            T, H, W = header["shape"]
            res_dtype = _residual_dtype_from_code(header["dtype_code"])
            return np.zeros((T, H, W), dtype=res_dtype), header
        return np.concatenate(chunks, axis=0), header
    raise ValueError(f"unsupported residual version: {version}")


def apply_residuals(decoded_frames, residual, original_dtype, context=None):
    """Add residual to decoded frames and clip back to original dtype range.

    Raises ValueError on shape mismatch — exact reconstruction requires
    identical shapes; silent truncation would corrupt later frames.
    """
    if decoded_frames.shape != residual.shape:
        raise ValueError(
            f"residual/decoded shape mismatch: decoded={decoded_frames.shape} "
            f"residual={residual.shape}"
            + (f" (context: {context})" if context else "")
        )
    target = np.dtype(original_dtype)
    out = decoded_frames.astype(np.int64, copy=False) + residual.astype(np.int64, copy=False)
    if np.issubdtype(target, np.integer):
        info = np.iinfo(target)
        np.clip(out, info.min, info.max, out=out)
    return out.astype(target, copy=False)


def _residual_dtype_from_code(code):
    return np.int16 if code == RESIDUAL_DTYPE_INT16 else np.int32


def _read_residual_header(f):
    """Read the common header from a residual file/buffer-like object positioned at start."""
    magic = f.read(8)
    if magic != RESIDUAL_MAGIC:
        raise ValueError(f"bad residual magic: {magic!r}")
    version = f.read(1)[0]
    mode_code = f.read(1)[0]
    dtype_code = f.read(1)[0]
    flags = f.read(1)[0]
    shape = tuple(int(x) for x in np.frombuffer(f.read(24), dtype=np.int64))
    n = int.from_bytes(f.read(2), "little")
    orig_dtype = f.read(n).decode("utf-8")
    n = int.from_bytes(f.read(2), "little")
    codec = f.read(n).decode("utf-8")
    compression_level = int.from_bytes(f.read(4), "little", signed=True)
    return {
        "version": int(version),
        "mode_code": mode_code,
        "mode": "background" if mode_code == RESIDUAL_MODE_BACKGROUND else "full",
        "dtype_code": dtype_code,
        "residual_dtype": np.dtype(_residual_dtype_from_code(dtype_code)).name,
        "flags": flags,
        "use_delta": bool(flags & 0x01),
        "use_shuffle": bool(flags & 0x02),
        "shape": shape,
        "original_dtype": orig_dtype,
        "codec": codec,
        "compression_level": compression_level,
    }


def _build_header_bytes(version, mode, residual_dtype_code, flags, shape,
                       original_dtype, codec, compression_level):
    mode_code = RESIDUAL_MODE_BACKGROUND if mode == "background" else RESIDUAL_MODE_FULL
    orig_dtype_bytes = np.dtype(original_dtype).name.encode("utf-8")
    codec_bytes = str(codec).encode("utf-8")
    out = bytearray()
    out += RESIDUAL_MAGIC
    out += bytes([int(version), mode_code, int(residual_dtype_code), int(flags)])
    out += np.array(shape, dtype=np.int64).tobytes()
    out += len(orig_dtype_bytes).to_bytes(2, "little") + orig_dtype_bytes
    out += len(codec_bytes).to_bytes(2, "little") + codec_bytes
    out += int(compression_level).to_bytes(4, "little", signed=True)
    return bytes(out)


def _encode_chunk_payload(chunk, residual_dtype, use_delta, use_shuffle):
    res_dtype = np.dtype(residual_dtype)
    chunk = np.asarray(chunk, dtype=res_dtype)
    if use_delta and chunk.shape[0] > 1:
        delta = np.empty_like(chunk)
        delta[0] = chunk[0]
        delta[1:] = chunk[1:] - chunk[:-1]
        body = delta.tobytes()
    else:
        body = chunk.tobytes()
    if use_shuffle:
        body = _residual_shuffle_bytes(body, res_dtype.itemsize)
    return body


def _decode_chunk_payload(payload, residual_dtype, n_frames, frame_shape, use_delta, use_shuffle):
    res_dtype = np.dtype(residual_dtype)
    H, W = frame_shape
    if use_shuffle:
        payload = _residual_unshuffle_bytes(payload, res_dtype.itemsize)
    chunk = np.frombuffer(payload, dtype=res_dtype).reshape((n_frames, H, W)).copy()
    if use_delta and chunk.shape[0] > 1:
        chunk = np.cumsum(chunk, axis=0).astype(res_dtype, copy=False)
    return chunk


def save_residual_sidecar_streaming(path, shape, original_dtype, mode, codec, compression_level,
                                    chunk_iter, chunk_size,
                                    use_delta=True, use_shuffle=True, zstd_level=19,
                                    residual_dtype=np.int32):
    """Stream a residual to disk as the chunked, indexed v2 format.

    The writer reserves an index table up front, writes payloads while
    recording byte offsets, then back-patches the index. Readers can then
    jump directly to any chunk in O(1).

    Parameters
    ----------
    chunk_iter : iterable of (start, end, chunk_array_int)
        Yields per-chunk residual arrays of shape ``(end - start, H, W)``.
        Frame indices must cover ``[0, T)`` exactly, with no gaps.
    chunk_size : int
        Frames per chunk advertised in the header (the last chunk may be smaller).
    residual_dtype : numpy dtype
        ``np.int16`` or ``np.int32``. Caller is responsible for selecting a
        dtype that the data fits — overflow raises during cast.
    """
    if mode not in ("full", "background"):
        raise ValueError(f"unknown residual mode: {mode}")
    T, H, W = shape
    res_dtype = np.dtype(residual_dtype)
    if res_dtype == np.int16:
        dtype_code = RESIDUAL_DTYPE_INT16
    elif res_dtype == np.int32:
        dtype_code = RESIDUAL_DTYPE_INT32
    else:
        raise ValueError(f"residual_dtype must be int16 or int32; got {res_dtype}")

    flags = RESIDUAL_FLAG_INDEXED
    if use_delta:
        flags |= RESIDUAL_FLAG_DELTA
    if use_shuffle:
        flags |= RESIDUAL_FLAG_SHUFFLE

    header = _build_header_bytes(RESIDUAL_VERSION_V2, mode, dtype_code, flags,
                                 shape, original_dtype, codec, compression_level)
    n_chunks_planned = (T + chunk_size - 1) // chunk_size if T else 0

    cctx = zstd.ZstdCompressor(level=zstd_level)
    index_entries = []
    expected_start = 0

    with open(path, "wb") as f:
        f.write(header)
        f.write(int(chunk_size).to_bytes(4, "little", signed=True))
        n_chunks_pos = f.tell()
        f.write(int(n_chunks_planned).to_bytes(4, "little", signed=True))

        # Reserve the index table; back-patched after payloads are written.
        index_pos = f.tell()
        f.write(b"\x00" * (n_chunks_planned * RESIDUAL_INDEX_ENTRY_SIZE))

        for start, end, chunk in chunk_iter:
            if start != expected_start:
                raise ValueError(f"chunk gap: expected start {expected_start}, got {start}")
            n_frames = end - start
            if chunk.shape != (n_frames, H, W):
                raise ValueError(
                    f"chunk shape mismatch: expected ({n_frames},{H},{W}) got {chunk.shape}"
                )
            payload = _encode_chunk_payload(chunk, res_dtype, use_delta, use_shuffle)
            compressed = cctx.compress(payload)

            offset = f.tell()
            f.write(compressed)
            index_entries.append((start, n_frames, offset, len(compressed)))
            expected_start = end

        if expected_start != T:
            raise ValueError(f"chunk iter covered {expected_start} frames, expected {T}")

        end_of_payloads = f.tell()

        # Patch n_chunks (in case fewer chunks than planned were emitted).
        if len(index_entries) != n_chunks_planned:
            f.seek(n_chunks_pos)
            f.write(int(len(index_entries)).to_bytes(4, "little", signed=True))

        # Fill the actual index entries.
        f.seek(index_pos)
        for start, n_frames, off, ln in index_entries:
            f.write(int(start).to_bytes(4, "little", signed=False))
            f.write(int(n_frames).to_bytes(4, "little", signed=False))
            f.write(int(off).to_bytes(8, "little", signed=False))
            f.write(int(ln).to_bytes(8, "little", signed=False))

        # Restore EOF position so readers do not see stale trailing slots
        # interpreted as additional payload.
        f.seek(end_of_payloads)


def iter_residual_chunks_from_path(path):
    """Yield ``(start, end, chunk_array, header)`` for each chunk in a residual file.

    Works for v1 (a single blob covering the whole shape), legacy v2
    (sequential, no index) and indexed v2 (jumps via the index table).
    """
    with open(path, "rb") as f:
        header = _read_residual_header(f)
        if header["version"] == RESIDUAL_VERSION_V2:
            chunk_size = int.from_bytes(f.read(4), "little", signed=True)
            n_chunks = int.from_bytes(f.read(4), "little", signed=True)
            header["chunk_size"] = chunk_size
            header["n_chunks"] = n_chunks
            res_dtype = _residual_dtype_from_code(header["dtype_code"])
            shape = header["shape"]
            H, W = shape[1], shape[2]
            dctx = zstd.ZstdDecompressor()
            indexed = bool(header["flags"] & RESIDUAL_FLAG_INDEXED)
            if indexed:
                index = _read_index_table(f, n_chunks)
                header["index"] = index
                for start, n_frames, off, clen in index:
                    f.seek(off)
                    compressed = f.read(clen)
                    payload = dctx.decompress(compressed)
                    chunk = _decode_chunk_payload(payload, res_dtype, n_frames, (H, W),
                                                  header["use_delta"], header["use_shuffle"])
                    yield start, start + n_frames, chunk, header
            else:
                start = 0
                for _ in range(n_chunks):
                    n_frames = int.from_bytes(f.read(4), "little", signed=False)
                    clen = int.from_bytes(f.read(8), "little", signed=False)
                    compressed = f.read(clen)
                    payload = dctx.decompress(compressed)
                    chunk = _decode_chunk_payload(payload, res_dtype, n_frames, (H, W),
                                                  header["use_delta"], header["use_shuffle"])
                    yield start, start + n_frames, chunk, header
                    start += n_frames
        elif header["version"] == RESIDUAL_VERSION:
            # v1: rest of file is a single zstd frame containing the whole residual
            compressed = f.read()
            res_dtype = _residual_dtype_from_code(header["dtype_code"])
            itemsize = np.dtype(res_dtype).itemsize
            dctx = zstd.ZstdDecompressor()
            body = dctx.decompress(compressed)
            if header["use_shuffle"]:
                body = _residual_unshuffle_bytes(body, itemsize)
            res = np.frombuffer(body, dtype=res_dtype).reshape(header["shape"]).copy()
            if header["use_delta"] and res.shape[0] > 1:
                res = np.cumsum(res, axis=0).astype(res_dtype, copy=False)
            header.setdefault("chunk_size", res.shape[0])
            header.setdefault("n_chunks", 1)
            yield 0, res.shape[0], res, header
        else:
            raise ValueError(f"unsupported residual version: {header['version']}")


def _read_index_table(f, n_chunks):
    """Read the index table (n_chunks entries) from a file positioned at its start."""
    raw = f.read(n_chunks * RESIDUAL_INDEX_ENTRY_SIZE)
    entries = []
    for i in range(n_chunks):
        base = i * RESIDUAL_INDEX_ENTRY_SIZE
        start = int.from_bytes(raw[base:base + 4], "little", signed=False)
        n_frames = int.from_bytes(raw[base + 4:base + 8], "little", signed=False)
        off = int.from_bytes(raw[base + 8:base + 16], "little", signed=False)
        ln = int.from_bytes(raw[base + 16:base + 24], "little", signed=False)
        entries.append((start, n_frames, off, ln))
    return entries


def read_residual_header(path):
    """Cheap header read that does not decode any chunks. Includes the index for indexed v2."""
    with open(path, "rb") as f:
        h = _read_residual_header(f)
        if h["version"] == RESIDUAL_VERSION_V2:
            h["chunk_size"] = int.from_bytes(f.read(4), "little", signed=True)
            h["n_chunks"] = int.from_bytes(f.read(4), "little", signed=True)
            if h["flags"] & RESIDUAL_FLAG_INDEXED:
                h["index"] = _read_index_table(f, h["n_chunks"])
        return h


def read_residual_index(path):
    """Return the index table for indexed v2 residual files, or None otherwise."""
    h = read_residual_header(path)
    if h["version"] != RESIDUAL_VERSION_V2:
        return None
    if not (h["flags"] & RESIDUAL_FLAG_INDEXED):
        return None
    return h["index"]


def read_residual_chunk_at(path, index_entry, header=None):
    """Read and decode a single chunk by index entry; does not touch other chunks.

    ``index_entry`` is ``(start, n_frames, offset, length)``. Returns the
    decoded chunk array.
    """
    if header is None:
        header = read_residual_header(path)
    start, n_frames, offset, length = index_entry
    res_dtype = _residual_dtype_from_code(header["dtype_code"])
    H, W = header["shape"][1], header["shape"][2]
    with open(path, "rb") as f:
        f.seek(offset)
        compressed = f.read(length)
    if len(compressed) != length:
        raise IOError(f"short read: wanted {length} bytes at offset {offset}, got {len(compressed)}")
    dctx = zstd.ZstdDecompressor()
    payload = dctx.decompress(compressed)
    return _decode_chunk_payload(payload, res_dtype, n_frames, (H, W),
                                  header["use_delta"], header["use_shuffle"])


def get_residual_path_for_video(video_path):
    base, _ = os.path.splitext(video_path)
    return base + RESIDUAL_SUFFIX


def _container_for_mkv_video_codec(codec_name):
    """Pick ``(file_ext, ffmpeg_format_name)`` for re-muxing an MKV's video stream.

    The MKV container itself accepts almost everything, but a stream-copy
    extraction has to land in a container that can hold the codec. ``.mp4`` is
    safe for h264/hevc/av1; ffv1 needs MKV (or AVI); ProRes needs MOV. Anything
    unknown stays in MKV, which is the most permissive option.
    """
    name = (codec_name or "").lower()
    if name in ("h264", "avc", "avc1", "libx264"):
        return "mp4", "mp4"
    if name in ("hevc", "h265", "libx265"):
        return "mp4", "mp4"
    if name in ("av1", "libaom-av1", "libsvtav1"):
        return "mp4", "mp4"
    if name == "ffv1":
        return "mkv", "matroska"
    if name in ("prores", "prores_ks"):
        return "mov", "mov"
    # Safe default — MKV holds anything ffmpeg can stream-copy.
    return "mkv", "matroska"


def find_matching_sidecars_for_video(video_path, search_dir=None):
    """Return {'npz': path|None, 'residual': path|None} for sidecars matching this video.

    The .npz uses the original input stem (without the `_compression_level_N`
    suffix); the residual sits next to the video and includes the level suffix.
    """
    video_dir = search_dir if search_dir is not None else os.path.dirname(video_path) or "."
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    input_stem = re.sub(r"_compression_level_-?\d+$", "", video_base)
    npz_path = os.path.join(video_dir, input_stem + ".npz")
    res_path = os.path.join(video_dir, video_base + RESIDUAL_SUFFIX)
    return {
        "npz": npz_path if os.path.exists(npz_path) else None,
        "residual": res_path if os.path.exists(res_path) else None,
    }


class _LazyVideoProbe:
    """Shape/dtype probe for an encoded video without decoding any frames.

    Stand-in for a Dask array in the streaming UNSPARZ path: it exposes
    ``shape``, ``dtype``, and ``ndim`` so existing code that just queries
    metadata keeps working, but it never materialises frames. The streaming
    pipeline opens its own PyAV iterator on the underlying file when it
    actually needs to decode.

    For containers where ``stream.frames`` is not populated we fall back to
    iterating the decode pass once just to count — that is one full CPU pass
    over the file but does not allocate per-frame buffers.
    """

    __slots__ = ("path", "_shape", "_dtype")

    def __init__(self, path):
        self.path = path
        self._shape, self._dtype = self._probe(path)

    @staticmethod
    def _probe(path):
        container = av.open(path)
        try:
            stream = container.streams.video[0]
            T = stream.frames or 0
            ctx = stream.codec_context
            H = ctx.height or 0
            W = ctx.width or 0
            if H == 0 or W == 0 or T == 0:
                first = next(container.decode(stream), None)
                if first is None:
                    raise RuntimeError(f"empty video: {path}")
                arr = first.to_ndarray(format="gray16le")
                if H == 0 or W == 0:
                    H, W = int(arr.shape[0]), int(arr.shape[1])
                if T == 0:
                    count = 1  # already consumed first
                    for _ in container.decode(stream):
                        count += 1
                    T = count
        finally:
            container.close()
        return (int(T), int(H), int(W)), np.dtype(np.uint16)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return 3


class SPARZIP:
    def __init__(self, path_image_files1:str,
                 stem:str,
                 output_path:str,
                 path_image_files2:str = None,
                 peaks_process:str = None,
                 relative_threshold:float = 0.45,
                 epsilon:int = 12,
                 kernel_size:int = 9,
                #  compression_level:int=0,
                 batch_size:int=100,
                 stack_size:int=250,
                 reflect_bp2:bool = False,
                 find_peaks:bool = True,
                 align_planes:bool=False,
                 num_workers:int=4,
                 num_dask_workers:int=2,
                 extract_metadata:bool=False,
                 create_single_file:bool=False,
                 save_metadata_to_json:bool=False,
                 save_residuals:bool=False,
                 residual_mode:str="auto",
                 residual_chunk_size:int=16,
                 ):
      
        """
        Class for compressing the data into a sparse matrix and into a mp4 file.

        Parameters
        ----------
        path_image_files1 : str
            path to the first image file
        path_image_files2 : str
            path to the second image file
        stem : str
            stem of the output file
        relative_threshold : float, optional
            relative threshold for peak finding, by default 0.5
        epsilon : int, optional
            epsilon for peak finding, by default 12
        kernel_size : int, optional
            kernel size for peak finding, by default 3
        comp_level : int, optional
            compression level for the output file, by default 0
        reflect_bp2 : bool, optional
            whether to reflect the second image, by default False
        align_planes : bool, optional
            whether to align the planes, by default False
        extract_metadata : bool, optional
            whether to extract comprehensive TIFF metadata for lossless preservation, by default False
            (Only relevant for TIFF files. DAT files do not contain metadata.)
        create_single_file : bool, optional
            whether to package the video and NPZ files into a single MKV container, by default False
        save_metadata_to_json : bool, optional
            whether to save metadata to a separate JSON file, by default False (embeds in NPZ)

        """
        # self.codec = codec
        self.single_plane = False
        if path_image_files2 is None:
            self.single_plane = True
        self.path_image_files1 = sorted(glob.glob(path_image_files1))
        if self.single_plane==False:
            self.path_image_files2 = sorted(glob.glob(path_image_files2))
        self.bp1, self.bp2 = self.load_images(path_image_files1, path_image_files2)
        # Optimized dtype checking - check metadata first without computing arrays
        try:
            # Check dtype from first array metadata (no computation needed)
            first_array_dtype = self.bp1[0].dtype
            if first_array_dtype == np.float32:
                self.bp1, self.bp2 = self.to_16bit()
        except (AttributeError, TypeError):
            # Fallback: compute only if metadata access fails
            dtype_first_array = self.bp1[0].compute().dtype
            if dtype_first_array == np.float32:
                self.bp1, self.bp2 = self.to_16bit()
        self.stem = stem
        if output_path[-1] != '/':
            self.output_path = output_path+"/"
        else:
            self.output_path = output_path
        self.rel_threshold = relative_threshold
        self.epsilon = epsilon
        self.kernel_size = kernel_size
        # self.compression_level = compression_level
        # Dynamic batch sizing based on available memory
        self.batch_size = self.get_optimal_batch_size(batch_size)
        self.stack_size = stack_size
        self.find_roi = find_peaks
        self.num_workers = num_workers
        self.num_dask_workers = num_dask_workers
        self.peak_process = peaks_process
        self.extract_metadata_flag = extract_metadata
        self.create_single_file = create_single_file
        self.save_metadata_to_json = save_metadata_to_json
        self.save_residuals = save_residuals
        if residual_mode not in ("auto", "full", "background"):
            raise ValueError(f"residual_mode must be 'auto', 'full', or 'background'; got {residual_mode!r}")
        self.residual_mode = residual_mode
        self.residual_chunk_size = max(1, int(residual_chunk_size))
        if reflect_bp2:
            if self.single_plane:
                print('Skipping reflection on single plane data.')
            else:
                self.bp2 = [np.flip(x, 2) for x in self.bp2]
        if align_planes:
            if self.single_plane:
                print('Cannot align single plane data. Skipping.')
            else:
                self.bp2 = self.align_planes(self.bp1, self.bp2)

        self.processed_bp1,self.processed_bp2 = self.process_images()
        
        # Extract and store metadata from source files (only if enabled)
        if self.extract_metadata_flag:
            self.metadata_bp1, self.metadata_bp2 = self.extract_metadata()
        else:
            # Initialize empty metadata
            self.metadata_bp1, self.metadata_bp2 = [], []

    def get_optimal_batch_size(self, base_batch_size):
        """
        Calculate optimal batch size based on available system memory.
        
        Args:
            base_batch_size: User-requested batch size
            
        Returns:
            Optimized batch size that fits within memory constraints
        """
        try:
            import psutil
            
            # Get available memory (in bytes)
            available_memory = psutil.virtual_memory().available
            
            # Estimate memory per batch item based on typical image sizes
            if hasattr(self, 'bp1') and self.bp1:
                try:
                    # Estimate based on first array if available
                    first_array = self.bp1[0]
                    if hasattr(first_array, 'nbytes'):
                        estimated_item_size = first_array.nbytes
                    elif hasattr(first_array, 'dtype') and hasattr(first_array, 'shape'):
                        estimated_item_size = first_array.dtype.itemsize * np.prod(first_array.shape)
                    else:
                        estimated_item_size = 1024 * 1024  # 1MB fallback
                except:
                    estimated_item_size = 1024 * 1024  # 1MB fallback
            else:
                estimated_item_size = 1024 * 1024  # 1MB fallback for typical microscopy image
            
            # Use at most 10% of available memory for batch processing
            max_memory_for_batch = available_memory * 0.1
            
            # Calculate optimal batch size
            optimal_size = int(max_memory_for_batch / estimated_item_size)
            
            # Ensure minimum batch size of 1 and don't exceed user's request
            optimal_size = max(1, min(optimal_size, base_batch_size))
            
            if optimal_size < base_batch_size:
                print(f"Reducing batch size from {base_batch_size} to {optimal_size} due to memory constraints")
            
            return optimal_size
            
        except ImportError:
            print("psutil not available, using default batch size")
            return base_batch_size
        except Exception as e:
            print(f"Error calculating optimal batch size: {e}, using default")
            return base_batch_size
        
    def load_images(self, path_image_files1, path_image_files2=None, multipage=False):
        if self.peak_process not in [None, 'median']:
            raise ValueError('Peaks process must be either None or median.')


    @delayed
    def load_dat_file(self,dat_file,dimX, dimY):
        raw_data = np.memmap(dat_file, dtype='uint16')
        img_nums = len(raw_data) // (dimX * dimY)
        return np.reshape(raw_data, (img_nums, dimY, dimX))

    def load_images(self, path_image_files1:str, path_image_files2:str):
        print ('Lazily loading images...')
        # files1 = sorted(glob.glob(path_image_files1))
        files1=self.path_image_files1
        if not files1:
            raise ValueError(f"No files found at {path_image_files1}")
        multipage = False
        ext = os.path.splitext(files1[0])[1]
        if not ext:
            raise ValueError(f"No extension found for file {files1[0]}")
        if ext == '.tiff' or ext == '.tif':
            with TiffFile(files1[0]) as tif:
                multipage = len(tif.pages) > 1
            if multipage:
                p1 = [dask_image.imread.imread(f) for f in files1]
            else:
                p1 = []
                for i in range(0, len(files1), self.stack_size):
                    images = [dask_image.imread.imread(f) for f in files1[i:i+self.stack_size]]
                    p1.append(da.concatenate(images,axis=0))
        elif ext == '.dat':
            dir1=os.path.dirname(files1[0])
            json_path = os.path.join(dir1, 'data.json')
            if not os.path.exists(json_path):
                raise ValueError(f"data.json not found in directory: {dir1}")
            with open(json_path, 'r') as f:
                data_tmp = pd.DataFrame.from_dict(json.load(f))
            dimX = data_tmp.loc['Image'].value['RecordDimX']
            dimY = data_tmp.loc['Image'].value['RecordDimY']
            p1 = []
            for dat_file in files1:
                raw_data = np.memmap(dat_file, dtype='uint16')
                img_nums = len(raw_data) // (dimX * dimY)
                p1.append(da.from_delayed(self.load_dat_file(dat_file, dimX, dimY), shape=(img_nums, dimY, dimX), dtype='uint16').rechunk((1, dimY, dimX)))
                # Explicit cleanup of memory mapping to prevent memory leaks
                del raw_data
        else:
            raise ValueError(f'Unsupported file extension: {ext}.Supported extensions are .tiff, .tif and SRX .dat')
        if self.single_plane:
            print ('Working with single plane data.')
            return p1, None
        print('Working with biplane data.')
        # files2 = sorted(glob.glob(path_image_files2))
        files2 = self.path_image_files2
        if not files2:
            raise ValueError(f"No files found at {path_image_files2}")
        ext = os.path.splitext(files2[0])[1]
        if not ext:
            raise ValueError(f"No extension found for file {files2[0]}")
        if ext == '.tiff' or ext == '.tif':
            if multipage:
                p2 = [dask_image.imread.imread(f) for f in files2]
            else:
                p2 = []
                for i in range(0, len(files2), self.stack_size):
                    images = [dask_image.imread.imread(f) for f in files2[i:i+self.stack_size]]
                    p2.append(da.concatenate(images,axis=0))
        elif ext == '.dat':
            dir1=os.path.dirname(files1[0])
            json_path = os.path.join(dir1, 'data.json')
            if not os.path.exists(json_path):
                raise ValueError(f"data.json not found in directory: {dir1}")
            p2 = []
            for dat_file in files2:
                raw_data = np.memmap(dat_file, dtype='uint16')
                img_nums = len(raw_data) // (dimX * dimY)
                p2.append(da.from_array(np.reshape(raw_data, (img_nums, dimY, dimX)), chunks=(1, dimY, dimX)) )
                # Explicit cleanup of memory mapping to prevent memory leaks
                del raw_data
        else:
            raise ValueError(f'Unsupported file extension: {ext}')
        return p1, p2

    def to_16bit(self):
        print('Float32 data detected.')
        print('Converting to 16 bit...')
        # z1 = self.bp1.map_blocks(lambda x:((x - x.min())/ (x.max() - x.min())*65535).astype('uint16'), dtype='uint16')
        z1 = [blck.map_blocks(lambda x:((x - x.min())/ (x.max() - x.min())*65535).astype('uint16'), dtype='uint16') for blck in self.bp1]
        if self.single_plane:
            return z1, None
        # z2 = self.bp2.map_blocks(lambda x:(( x - x.min())/ (x.max() - x.min()) * 65535).astype('uint16'), dtype='uint16')
        z2 = [blck.map_blocks(lambda x:((x - x.min())/ (x.max() - x.min())*65535).astype('uint16'), dtype='uint16') for blck in self.bp2]
        return z1, z2 
    # @njit
    def find_peaks(self, image:np.ndarray, kernel_size:int, min_distance:int = 1):

        #find the local maxima in the image
        coordinates = peak_local_max(image, 
                                    threshold_rel= self.rel_threshold, 
                                    min_distance=min_distance, 
                                    footprint=np.ones((kernel_size, kernel_size))
                                    )

        return coordinates
    # @njit
    def intersection(self, peaks1:np.ndarray, peaks2:np.ndarray, bp1:np.ndarray, epsilon:int):
        # Compute element-wise differences between peaks
        diffs = np.abs(peaks1[:, None, :] - peaks2)
        
        # Find indices of peaks in peaks1 that are within epsilon of any peak in peaks2
        common_inds_1 = np.where(np.any(np.all(np.abs(diffs) <= epsilon, axis=2), axis=1))[0]
        
        # Find indices of peaks in peaks2 that are within epsilon of any peak in peaks1
        # common_inds_2 = np.where(np.any(np.all(np.abs(diffs) <= epsilon, axis=2), axis=0))[0]
        
        # Create binary matrix where common peaks are set to 1 and all other elements are 0
        common_peaks = np.zeros(bp1.shape)
        common_peaks[peaks1[common_inds_1, 0], peaks1[common_inds_1, 1]] = 1
        
        return common_peaks
    # @njit
    def union(self, peaks1:np.ndarray, peaks2:np.ndarray, bp1):
        merged = np.vstack([peaks1, peaks2])

        merged_peaks = np.zeros(bp1)
        merged_peaks[merged[:, 0], merged[:, 1]] = 1    
        return merged_peaks
    # @njit
    def add_kernel(self, image:np.ndarray,kernel_size:int):

        kernel = np.ones((kernel_size, kernel_size))

        pad_size = kernel.shape[0]//2
        arr = np.pad(image, pad_size, mode='constant')

        # Find the indices of the ones in the input array
        ones_indices = np.argwhere(arr == 1)

        # Initialize a new array with the same shape as the input array
        new_arr = np.zeros(arr.shape)

        # Loop through the indices of the ones and center the kernel on each one
        for i, j in ones_indices:
            start_i = i - kernel.shape[0]//2
            end_i = i + kernel.shape[0]//2 + 1
            start_j = j - kernel.shape[1]//2
            end_j = j + kernel.shape[1]//2 + 1

            # Center the kernel on the current one
            new_arr[start_i:end_i, start_j:end_j] = kernel

        # Remove the padding from the new array
        return new_arr[pad_size:-pad_size, pad_size:-pad_size]

    def align_planes(self,plane1:np.ndarray, plane2:np.ndarray):
        print('Aligning planes, please wait...')
        # frames = plane1.shape[0]
        # # print (frames)
        # if frames < 50:
        #     shifts, _, _ = phase_cross_correlation(plane1, plane2,
        #                                                 upsample_factor=100)

        
        # else:
            # sample = np.random.randint(int(frames*0.48), int(frames*0.52),150)
        plane1_c = da.concatenate(plane1, axis=0)
        plane2_c = da.concatenate(plane2, axis=0)
        b1 = plane1_c[int(plane1_c.shape[0]*0.48): int(plane1_c.shape[0]*0.52)]
        b2 = plane2_c[int(plane2_c.shape[0]*0.48): int(plane2_c.shape[0]*0.52)]
        shifts, _, _ = phase_cross_correlation(b1, b2,
                                                    upsample_factor=100)
        #We don't want to shift frame numbers, so we set the first value to 0
        shifts[0] = 0
        print(f'Applying mean shift {shifts} to all frames.')
        del plane1_c, plane2_c
        gc.collect()
            
        return [p.map_blocks(lambda x: shift(x, shifts, mode='constant')) for p in plane2]


    def get_raw_frame(self,index:int=None,start_frame:int=None,end_frame:int=None,plane:str='bp1'):
        if index is None:
            print ('No index provided. Using index 0.')
            index = 0
        if start_frame is None:
            # raise ValueError('Please provide a start frame.')
            print ('No starting frame provided. Using frame 0.')
            start_frame = 0
        if end_frame is None:
            print('No end frame provided. Using starting_frame+1.')
            end_frame = start_frame+1
        if plane == 'bp1':
            return self.bp1[index].blocks[start_frame:end_frame,0].compute()
        else:
            return self.bp2[index].blocks[start_frame:end_frame,0].compute()

    def get_processed_frame(self,index:int=None,start_frame:int=None,end_frame:int=None,plane:str='bp1'):
        if index is None:
            print ('No index provided. Using index 0.')
            index = 0
        if start_frame is None:
            # raise ValueError('Please provide a start frame.')
            print ('No starting frame provided. Using frame 0.')
            start_frame = 0
        if end_frame is None:
            print('No end frame provided. Using starting_frame+1.')
            end_frame = start_frame+1
        if plane == 'bp1':
            return self.processed_bp1[index].blocks[start_frame:end_frame,0].compute().todense()
        else:
            return self.processed_bp2[index].blocks[start_frame:end_frame,0].compute().todense()

    def median_patch(self, block):
        new_block = np.copy(block)
        for i in range(block.shape[0]):
            frame = block[i]
            peaks = self.find_peaks(frame, self.kernel_size, min_distance=1)
            for (r, c) in peaks:
                half_k = self.kernel_size // 2
                r_start = max(r - half_k, 0)
                r_end = min(r + half_k + 1, frame.shape[0])
                c_start = max(c - half_k, 0)
                c_end = min(c + half_k + 1, frame.shape[1])
                median_val = np.median(frame[r_start:r_end, c_start:c_end])
                new_block[i, r_start:r_end, c_start:c_end] = median_val
        return new_block

    def _process_frame_to_sparse_single(self, frame):
        """
        Fused operation for single plane: find peaks, create mask, apply kernel, extract sparse.
        Reduces dask task graph from 5 operations to 1 per block.
        """
        frame_2d = frame[0, :, :] if frame.ndim == 3 else frame

        # Find peaks
        peaks = peak_local_max(frame_2d,
                              threshold_rel=self.rel_threshold,
                              min_distance=1,
                              footprint=np.ones((self.kernel_size, self.kernel_size)))

        if len(peaks) == 0:
            # Return empty sparse matrix
            return sparse.COO(np.zeros_like(frame_2d, dtype='int16'))

        # Create mask at peak locations
        mask = np.zeros(frame_2d.shape, dtype='int16')
        mask[peaks[:, 0], peaks[:, 1]] = 1

        # Expand with kernel
        kernel_mask = self.add_kernel(mask, self.kernel_size)

        # Extract values where kernel is non-zero
        masked = np.where(kernel_mask, frame_2d, 0).astype('int16')

        return sparse.COO(masked)

    def _process_biplane_frame(self, frame1, frame2):
        """
        Fused operation for biplane: find peaks in both, union, apply kernel.
        Returns tuple of (sparse1, sparse2).
        """
        f1 = frame1[0, :, :] if frame1.ndim == 3 else frame1
        f2 = frame2[0, :, :] if frame2.ndim == 3 else frame2

        # Find peaks in both planes
        peaks1 = peak_local_max(f1, threshold_rel=self.rel_threshold,
                               min_distance=1, footprint=np.ones((self.kernel_size, self.kernel_size)))
        peaks2 = peak_local_max(f2, threshold_rel=self.rel_threshold,
                               min_distance=1, footprint=np.ones((self.kernel_size, self.kernel_size)))

        # Union of peaks
        if len(peaks1) == 0 and len(peaks2) == 0:
            return sparse.COO(np.zeros_like(f1, dtype='int16')), sparse.COO(np.zeros_like(f2, dtype='int16'))

        if len(peaks1) > 0 and len(peaks2) > 0:
            merged = np.vstack([peaks1, peaks2])
        elif len(peaks1) > 0:
            merged = peaks1
        else:
            merged = peaks2

        # Create union mask
        mask = np.zeros(f1.shape, dtype='int16')
        mask[merged[:, 0], merged[:, 1]] = 1

        # Expand with kernel
        kernel_mask = self.add_kernel(mask, self.kernel_size)

        # Extract values from both planes
        masked1 = np.where(kernel_mask, f1, 0).astype('int16')
        masked2 = np.where(kernel_mask, f2, 0).astype('int16')

        return sparse.COO(masked1), sparse.COO(masked2)

    def _process_biplane_to_sparse_bp1(self, frame1, frame2):
        """
        Fused biplane operation returning sparse for plane 1.
        Computes union of peaks from both planes, applies to plane 1.
        """
        sparse1, _ = self._process_biplane_frame(frame1, frame2)
        return sparse1

    def _process_biplane_to_sparse_bp2(self, frame1, frame2):
        """
        Fused biplane operation returning sparse for plane 2.
        Computes union of peaks from both planes, applies to plane 2.
        """
        _, sparse2 = self._process_biplane_frame(frame1, frame2)
        return sparse2

    def process_images(self):
        print('Processing images...')

        if self.single_plane:
            print('Single plane - using fused processing')
            if self.peak_process == 'median':
                print('Applying median patch...')
                self.bp1 = [block.map_blocks(self.median_patch, dtype=block.dtype) for block in self.bp1]

            # Fused single-plane processing: peaks -> mask -> kernel -> sparse in one operation
            result = [blck.map_blocks(self._process_frame_to_sparse_single, dtype=object)
                     for blck in self.bp1]
            print('Done.')
            return result, None

        # Biplane processing - using fused operations
        print('Biplane processing - using fused operations')
        assert len(self.bp1) == len(self.bp2), 'Error: Both biplanes must have the same number of images.'

        if self.peak_process == 'median':
            print('Applying median patch...')
            self.bp1 = [block.map_blocks(self.median_patch, dtype=block.dtype) for block in self.bp1]
            self.bp2 = [block.map_blocks(self.median_patch, dtype=block.dtype) for block in self.bp2]

        # Fused biplane processing: process both planes together for peak union
        sp1_list = []
        sp2_list = []

        for i in range(len(self.bp1)):
            # Use da.map_blocks with both arrays - fused peak finding and sparse conversion
            sp1 = da.map_blocks(
                self._process_biplane_to_sparse_bp1,
                self.bp1[i], self.bp2[i],
                dtype=object,
                drop_axis=None
            )
            sp2 = da.map_blocks(
                self._process_biplane_to_sparse_bp2,
                self.bp1[i], self.bp2[i],
                dtype=object,
                drop_axis=None
            )
            sp1_list.append(sp1)
            sp2_list.append(sp2)

        print('Done.')
        return sp1_list, sp2_list

    def extract_metadata(self):
        """
        Extract comprehensive TIFF metadata including OME-XML from input files.
        Only processes TIFF files - DAT files don't contain metadata.
        """
        print('Extracting metadata from source files...')
        metadata_bp1 = []
        
        for file_path in self.path_image_files1:
            metadata = {}
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext in ['.tiff', '.tif']:
                try:
                    with TiffFile(file_path) as tif:
                        if tif.pages:
                            first_page = tif.pages[0]
                            # Basic image properties
                            metadata['shape'] = first_page.shape
                            metadata['dtype'] = str(first_page.dtype)
                            metadata['is_multipage'] = len(tif.pages) > 1
                            metadata['page_count'] = len(tif.pages)
                            
                            # Extract TIFF tags from first page (global metadata)
                            metadata['tags'] = {}
                            for tag in first_page.tags:
                                try:
                                    if hasattr(tag, 'name') and hasattr(tag, 'value'):
                                        # Store commonly used tags
                                        if tag.name in ['ImageDescription', 'Software', 'DateTime', 
                                                       'XResolution', 'YResolution', 'ResolutionUnit']:
                                            if isinstance(tag.value, (str, int, float, bool)):
                                                metadata['tags'][tag.name] = tag.value
                                            elif isinstance(tag.value, (tuple, list)):
                                                metadata['tags'][tag.name] = list(tag.value)
                                except (AttributeError, ValueError, TypeError):
                                    continue
                            
                            # Extract individual IFD metadata from each page
                            metadata['individual_ifds'] = []
                            for page_idx, page in enumerate(tif.pages):
                                ifd_metadata = {'page_index': page_idx}
                                
                                # Extract all tags from this IFD
                                ifd_tags = {}
                                for tag in page.tags:
                                    try:
                                        if hasattr(tag, 'name') and hasattr(tag, 'value'):
                                            # Store all tags for individual IFDs
                                            if isinstance(tag.value, (str, int, float, bool)):
                                                ifd_tags[tag.name] = tag.value
                                            elif isinstance(tag.value, (tuple, list)) and len(tag.value) <= 10:
                                                ifd_tags[tag.name] = list(tag.value)
                                            # Also store by tag number for custom tags
                                            if hasattr(tag, 'code'):
                                                ifd_tags[f'tag_{tag.code}'] = tag.value
                                    except (AttributeError, ValueError, TypeError):
                                        continue
                                
                                ifd_metadata['tags'] = ifd_tags
                                metadata['individual_ifds'].append(ifd_metadata)
                            
                            # Detect if individual IFD metadata varies significantly
                            metadata['requires_individual_ifd_writing'] = self._detect_individual_ifd_variation(metadata['individual_ifds'])
                            
                            # ImageJ metadata
                            if tif.is_imagej:
                                try:
                                    metadata['imagej_metadata'] = tif.imagej_metadata
                                    metadata['is_imagej'] = True
                                except:
                                    metadata['is_imagej'] = False
                            else:
                                metadata['is_imagej'] = False
                            
                            # OME-XML metadata (for OME-TIFF files)
                            # Check multiple sources for OME-XML
                            metadata['is_ome'] = False
                            if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                                metadata['ome_xml'] = tif.ome_metadata
                                metadata['is_ome'] = True
                            elif 'ImageDescription' in metadata['tags']:
                                # Check if ImageDescription contains OME-XML
                                desc = metadata['tags']['ImageDescription']
                                if isinstance(desc, str) and ('<?xml' in desc or '<OME' in desc):
                                    metadata['ome_xml'] = desc
                                    metadata['is_ome'] = True
                            
                            # Also check if this is an OME-TIFF file
                            if hasattr(tif, 'is_ome') and tif.is_ome:
                                metadata['is_ome'] = True
                                if not metadata.get('ome_xml') and hasattr(tif, 'ome_metadata'):
                                    metadata['ome_xml'] = tif.ome_metadata
                            
                            # Shaped metadata (includes additional structured metadata)
                            if hasattr(tif, 'shaped_metadata') and tif.shaped_metadata:
                                try:
                                    metadata['shaped_metadata'] = tif.shaped_metadata
                                except:
                                    pass
                                    
                except Exception as e:
                    print(f"Warning: Could not extract metadata from {file_path}: {e}")
                    metadata = {'error': str(e)}
            elif ext == '.dat':
                # DAT files don't contain metadata
                metadata = {'file_type': 'dat', 'has_metadata': False}
            else:
                metadata = {'file_type': 'non_tiff', 'has_metadata': False}
                
            metadata_bp1.append(metadata)
        
        # Extract metadata for bp2 if it exists
        metadata_bp2 = []
        if not self.single_plane:
            for file_path in self.path_image_files2:
                metadata = {}
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext in ['.tiff', '.tif']:
                    try:
                        with TiffFile(file_path) as tif:
                            if tif.pages:
                                first_page = tif.pages[0]
                                metadata['shape'] = first_page.shape
                                metadata['dtype'] = str(first_page.dtype)
                                metadata['is_multipage'] = len(tif.pages) > 1
                                metadata['page_count'] = len(tif.pages)
                                
                                # Extract TIFF tags from first page (global metadata)
                                metadata['tags'] = {}
                                for tag in first_page.tags:
                                    try:
                                        if hasattr(tag, 'name') and hasattr(tag, 'value'):
                                            if tag.name in ['ImageDescription', 'Software', 'DateTime', 
                                                           'XResolution', 'YResolution', 'ResolutionUnit']:
                                                if isinstance(tag.value, (str, int, float, bool)):
                                                    metadata['tags'][tag.name] = tag.value
                                                elif isinstance(tag.value, (tuple, list)):
                                                    metadata['tags'][tag.name] = list(tag.value)
                                    except (AttributeError, ValueError, TypeError):
                                        continue
                                
                                # Extract individual IFD metadata from each page
                                metadata['individual_ifds'] = []
                                for page_idx, page in enumerate(tif.pages):
                                    ifd_metadata = {'page_index': page_idx}
                                    
                                    # Extract all tags from this IFD
                                    ifd_tags = {}
                                    for tag in page.tags:
                                        try:
                                            if hasattr(tag, 'name') and hasattr(tag, 'value'):
                                                # Store all tags for individual IFDs
                                                if isinstance(tag.value, (str, int, float, bool)):
                                                    ifd_tags[tag.name] = tag.value
                                                elif isinstance(tag.value, (tuple, list)) and len(tag.value) <= 10:
                                                    ifd_tags[tag.name] = list(tag.value)
                                                # Also store by tag number for custom tags
                                                if hasattr(tag, 'code'):
                                                    ifd_tags[f'tag_{tag.code}'] = tag.value
                                        except (AttributeError, ValueError, TypeError):
                                            continue
                                    
                                    ifd_metadata['tags'] = ifd_tags
                                    metadata['individual_ifds'].append(ifd_metadata)
                                
                                # Detect if individual IFD metadata varies significantly
                                metadata['requires_individual_ifd_writing'] = self._detect_individual_ifd_variation(metadata['individual_ifds'])
                                
                                # ImageJ metadata
                                if tif.is_imagej:
                                    try:
                                        metadata['imagej_metadata'] = tif.imagej_metadata
                                        metadata['is_imagej'] = True
                                    except:
                                        metadata['is_imagej'] = False
                                else:
                                    metadata['is_imagej'] = False
                                
                                # OME-XML metadata (for OME-TIFF files)
                                # Check multiple sources for OME-XML
                                metadata['is_ome'] = False
                                if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                                    metadata['ome_xml'] = tif.ome_metadata
                                    metadata['is_ome'] = True
                                elif 'ImageDescription' in metadata['tags']:
                                    # Check if ImageDescription contains OME-XML
                                    desc = metadata['tags']['ImageDescription']
                                    if isinstance(desc, str) and ('<?xml' in desc or '<OME' in desc):
                                        metadata['ome_xml'] = desc
                                        metadata['is_ome'] = True
                                
                                # Also check if this is an OME-TIFF file
                                if hasattr(tif, 'is_ome') and tif.is_ome:
                                    metadata['is_ome'] = True
                                    if not metadata.get('ome_xml') and hasattr(tif, 'ome_metadata'):
                                        metadata['ome_xml'] = tif.ome_metadata
                                
                                # Shaped metadata
                                if hasattr(tif, 'shaped_metadata') and tif.shaped_metadata:
                                    try:
                                        metadata['shaped_metadata'] = tif.shaped_metadata
                                    except:
                                        pass
                                        
                    except Exception as e:
                        print(f"Warning: Could not extract metadata from {file_path}: {e}")
                        metadata = {'error': str(e)}
                elif ext == '.dat':
                    # DAT files don't contain metadata
                    metadata = {'file_type': 'dat', 'has_metadata': False}
                else:
                    metadata = {'file_type': 'non_tiff', 'has_metadata': False}
                    
                metadata_bp2.append(metadata)
        
        return metadata_bp1, metadata_bp2 if not self.single_plane else None

    def format_metadata_for_tifffile(self, metadata, frame_index=None):
        """Format extracted metadata for tifffile.TiffWriter"""
        if not metadata or 'error' in metadata:
            return None, []
        
        description = None
        extratags = []
        
        # For the first frame, use global metadata
        if frame_index is None or frame_index == 0:
            # Add OME-XML if present (takes precedence)
            if metadata.get('is_ome') and 'ome_xml' in metadata:
                description = metadata['ome_xml']
                # Add software tag (TIFF tag 305)
                extratags.append((305, 's', 0, "SPARZIP with OME-XML", True))
                return description, extratags
            
            # Add ImageJ metadata if present
            description_parts = []
            if metadata.get('is_imagej') and 'imagej_metadata' in metadata:
                try:
                    imagej_meta = metadata['imagej_metadata']
                    if isinstance(imagej_meta, dict):
                        for key, value in imagej_meta.items():
                            if isinstance(value, (str, int, float)):
                                description_parts.append(f"{key}={value}")
                except:
                    pass
            
            # Add TIFF tags to description and extratags
            if 'tags' in metadata:
                tags = metadata['tags']
                
                # ImageDescription is the most important
                if 'ImageDescription' in tags:
                    existing_desc = tags['ImageDescription']
                    if existing_desc and isinstance(existing_desc, str):
                        description = existing_desc
                        if description_parts:
                            description += f" | {' | '.join(description_parts)}"
                
                # Add other important tags as extratags
                for tag_name, tag_value in tags.items():
                    if tag_name == 'Software' and isinstance(tag_value, str):
                        extratags.append((305, 's', 0, tag_value, True))  # Software tag
                    elif tag_name == 'DateTime' and isinstance(tag_value, str):
                        extratags.append((306, 's', 0, tag_value, True))  # DateTime tag
                    elif tag_name not in ['ImageDescription', 'Software', 'DateTime'] and isinstance(tag_value, (str, int, float)):
                        description_parts.append(f"{tag_name}={tag_value}")
            
            # Create description from parts if not already set
            if description is None and description_parts:
                description = ' | '.join(description_parts)
            
            # Add default software tag if not present
            if not any(tag[0] == 305 for tag in extratags):
                extratags.append((305, 's', 0, "SPARZIP", True))
        
        else:
            # For subsequent frames, use individual IFD metadata if available
            if 'individual_ifds' in metadata and frame_index < len(metadata['individual_ifds']):
                ifd_meta = metadata['individual_ifds'][frame_index]
                ifd_tags = ifd_meta.get('tags', {})
                
                # Add IFD-specific ImageDescription
                if 'ImageDescription' in ifd_tags:
                    description = ifd_tags['ImageDescription']
                
                # Add IFD-specific custom tags
                for tag_name, tag_value in ifd_tags.items():
                    if tag_name.startswith('tag_') and isinstance(tag_value, (str, int, float)):
                        try:
                            tag_code = int(tag_name.split('_')[1])
                            if tag_code > 50000:  # Custom tags usually > 50000
                                if isinstance(tag_value, str):
                                    extratags.append((tag_code, 's', 0, str(tag_value), True))
                                elif isinstance(tag_value, int):
                                    extratags.append((tag_code, 'i', 1, tag_value, True))
                                elif isinstance(tag_value, float):
                                    # Convert float to string for Picasso compatibility
                                    extratags.append((tag_code, 's', 0, str(tag_value), True))
                        except (ValueError, IndexError):
                            continue
        
        return description, extratags
    
    # def compress_joblib(self,path,mat):
        # joblib.dump(mat, path, compress=('lzma', 9))


    def deflate(self):
        print('Deflating images...')
        show_progress_bar = False
        try:
            get_ipython()
            show_progress_bar = True
        except NameError:
            show_progress_bar = False
        with dask.config.set(scheduler='threads'):
            # Prepare a list to store delayed operations
            saves = []
            for i in range(len(self.processed_bp1)):
                # Directly append delayed save_npz operations to the list
                flnm1 = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files1[i]))[1])[0]
                
                # Get metadata for this file
                metadata_bp1_entry = self.metadata_bp1[i] if hasattr(self, 'metadata_bp1') and self.metadata_bp1 and i < len(self.metadata_bp1) else None
                
                # Use new method that includes metadata
                saves.append(delayed(self.save_npz_with_metadata)(self.output_path+flnm1+'.npz', self.processed_bp1[i], metadata_bp1_entry))
                
                if self.single_plane == False:
                    flnm2 = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files2[i]))[1])[0]
                    
                    # Get metadata for BP2 file
                    metadata_bp2_entry = self.metadata_bp2[i] if hasattr(self, 'metadata_bp2') and self.metadata_bp2 and i < len(self.metadata_bp2) else None
                    
                    saves.append(delayed(self.save_npz_with_metadata)(self.output_path+flnm2+'.npz', self.processed_bp2[i], metadata_bp2_entry))
            # Perform the save_npz operations
            if show_progress_bar:
                progress_bar = tqdm(total=len(saves), desc="Creating sparse matrices", position=0, leave=True)
            
            for i in range(0, len(saves), self.batch_size):
                batch = saves[i:i+self.batch_size]
                dask.compute(*batch, scheduler='threads',num_workers=self.num_workers)
                if show_progress_bar:
                    progress_bar.update(self.batch_size)
            if show_progress_bar:
                progress_bar.close()
        
        # Save metadata for lossless restoration
        self.save_metadata()
        
        end = time.time()
        



    def determine_ctu_size(self, image_width, image_height):
        # Example logic for determining CTU size
        min_dimension = min(image_width, image_height)
        if min_dimension <= 64:
            return 16  # Smallest CTU size for very small images
        elif min_dimension <= 128:
            return 32
        else:
            return 64  # Default CTU size for larger images
        

    def encode(self, codec:str='x265', compression_lvl:int=0, custom_dict:dict=None, custom_file_extension:str=None):
        if codec =='x265':
            w, h = self.bp1[0].shape[1], self.bp1[0].shape[2]
            ctu_size = self.determine_ctu_size(w, h)
            compression_levels = {0: {
                                                'vcodec': 'libx265',
                                                'pix_fmt': 'gray12le',
                                                'x265-params': f'lossless=1:ctu={ctu_size}',
                                            },
                                1:{
                                                'vcodec': 'libx265',
                                                'crf': '0', 
                                                'pix_fmt': 'gray12le',
                                                'x265-params': f'ctu={ctu_size}',
                                            },
                                2:{
                                                'vcodec': 'libx265',
                                                'crf': '5',
                                                'pix_fmt': 'gray12le',
                                                'x265-params': f'ctu={ctu_size}',
                                            },

                                3:{
                                                'vcodec': 'libx265',
                                                'crf': '15',
                                                'pix_fmt': 'gray12le',
                                                'x265-params': f'ctu={ctu_size}',
                                            }                                      
                                            
                                            
                            }
        elif codec == 'av1':
            compression_levels = {0: {
                                                'vcodec': 'libaom-av1',
                                                'pix_fmt': 'gray16le', 
                                                'strict': 'experimental',
                                                'crf': '0',
                                                'fps_mode': 'vfr'
                                }
,
                                1:{
                                                'vcodec': 'libaom-av1',
                                                'crf': '5',
                                                'pix_fmt': 'gray16le',
                                                'fps_mode': 'vfr'
                                            },
                                2:{
                                                'vcodec': 'libaom-av1',
                                                'crf': '15',
                                                'pix_fmt': 'gray16le',
                                                'fps_mode': 'vfr'
                                            },

                                3:{
                                                'vcodec': 'libaom-av1',
                                                'crf': '25',
                                                'pix_fmt': 'gray16le',
                                                'fps_mode': 'vfr'
                                            }                                      
                                            
                                            
                            }
        elif codec == 'x264':
            compression_levels = {0: {
                                                'vcodec': 'libx264',
                                                'pix_fmt': 'gray16le',
                                                'crf': '0'
                                },
                                1:{
                                                'vcodec': 'libx264',
                                                'crf': '5',
                                                'pix_fmt': 'gray16le'
                                },
                                2:{
                                                'vcodec': 'libx264',
                                                'crf': '15',
                                                'pix_fmt': 'gray16le'
                                },
                                3:{
                                                'vcodec': 'libx264',
                                                'crf': '25',
                                                'pix_fmt': 'gray16le'
                                }
                }

        elif codec =='ffv1':
            compression_levels = {
                                0: {  # Maximum compression
                                                'vcodec': 'ffv1',
                                                'pix_fmt': 'gray16le',
                                                'level': '3',
                                                'coder': '2',  # Range coder v2 (best)
                                                'context': '1',  # Large context
                                                'slices': '1',  # Single slice
                                                'slicecrc': '0',
                                                'g': '1',
                                },
                                1: {  # High compression
                                                'vcodec': 'ffv1',
                                                'pix_fmt': 'gray16le',
                                                'level': '3',
                                                'coder': '1',  # Range coder
                                                'context': '1',
                                                'slices': '1',
                                                'slicecrc': '0',
                                },
                                2: {  # Balanced
                                                'vcodec': 'ffv1',
                                                'pix_fmt': 'gray16le',
                                                'level': '3',
                                                'coder': '1',
                                                'context': '1',
                                                'slices': '4',
                                },
                                3: {  # Fast, decent compression
                                                'vcodec': 'ffv1',
                                                'pix_fmt': 'gray16le',
                                                'level': '3',
                                                'coder': '0',  # Golomb-Rice (fast)
                                                'context': '0',
                                                'slices': '4',
                                }
                }
        elif codec == 'prores':
            compression_levels = {
            0: {
                'vcodec': 'prores_ks',
                'pix_fmt': 'yuv444p10le',
                'profile:v': '4444',
            },
            1: {
                'vcodec': 'prores_ks',
                'pix_fmt': 'yuv422p10le',
                'profile:v': 'hq',
            },
            2: {
                'vcodec': 'prores_ks',
                'pix_fmt': 'yuv422p10le',
                'profile:v': 'standard',
            },
            3: {
                'vcodec': 'prores_ks',
                'pix_fmt': 'yuv422p10le',
                'profile:v': 'lt',
            }
        }
        
        elif codec =='user':
            assert custom_dict is not None, 'Error: Custom dictionary not provided.'
            assert custom_file_extension is not None, 'Error: Custom file extension not provided.'
            assert isinstance(custom_dict, dict), 'Error: Custom dictionary must be a dictionary.'
            assert 'vcodec' in custom_dict.keys(), 'Error: Custom dictionary must contain a vcodec key.'
            print ('Applying user defined settings.')

        else:
            raise ValueError(f'Unsupported codec: {codec}. Supported codecs are x265 and av1.')
        print('Compressing video...')
        show_progress_bar = False
        try:
            get_ipython()
            show_progress_bar = True
        except NameError:
            show_progress_bar = False
        # Process videos using delayed
        writes = []
        for k in range(len(self.processed_bp1)):
            input_file_name1 = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files1[k]))[1])[0]

            if codec =='prores':
                video_name1 = f'{self.output_path}{input_file_name1}_compression_level_{compression_lvl}.mov'
            elif codec =='ffv1':
                video_name1 = f'{self.output_path}{input_file_name1}_compression_level_{compression_lvl}.avi'
            elif codec=='user':
                video_name1 = f'{self.output_path}{input_file_name1}_compression_level_{compression_lvl}.{custom_file_extension}'
            else:
                video_name1 = f'{self.output_path}{input_file_name1}_compression_level_{compression_lvl}.mp4'
                

            if codec!='user':
                writer_args = compression_levels[compression_lvl]
            else:
                writer_args = custom_dict

            if self.single_plane:
                writes.append(delayed(self.write_frames_to_video)(
                    self.bp1[k], video_name1, writer_args
                ))
            else:
                writes.append(delayed(self.write_frames_to_video)(
                    self.bp1[k], video_name1, writer_args
                ))
                input_file_name2 = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files2[k]))[1])[0]
                
                if codec=='prores':
                    video_name2 = f'{self.output_path}{input_file_name2}_compression_level_{compression_lvl}.mov'
                elif codec=='ffv1':
                    video_name2 = f'{self.output_path}{input_file_name2}_compression_level_{compression_lvl}.avi'
                elif codec=='user':
                    video_name2 = f'{self.output_path}{input_file_name2}_compression_level_{compression_lvl}.{custom_file_extension}'
                else:
                    video_name2 = f'{self.output_path}{input_file_name2}_compression_level_{compression_lvl}.mp4'
                # video_name2 = f'{self.output_path}{self.stem}_bp2_compression_level_{compression_lvl}_part_{k}.mp4'
                writes.append(delayed(self.write_frames_to_video)(
                    self.bp2[k], video_name2, writer_args
                ))

        # Execute the delayed writes sequentially (one file at a time)
        # FFmpeg is not optimized for parallel encoding - sequential reduces memory and CPU contention
        print(f'Encoding {len(writes)} video file(s) sequentially...')
        for i, write_task in enumerate(writes):
            print(f'  Encoding file {i+1}/{len(writes)}...')
            if show_progress_bar:
                with ProgressBar():
                    compute(write_task, scheduler='threads', num_workers=self.num_workers)
            else:
                compute(write_task, scheduler='threads', num_workers=self.num_workers)
            gc.collect()

        # Save metadata for lossless restoration
        self.save_metadata()

    def write_frames_to_video(self, bp1_block, video_name, writer_args, chunk_size=200):
        """
        Memory-optimized video encoding using streaming chunks instead of loading entire video into memory.
        Enhanced with better error handling, validation, and moov atom fix.
        
        Args:
            bp1_block: Dask array of video frames
            video_name: Output video file path
            writer_args: FFmpeg encoder arguments
            chunk_size: Number of frames to process at once (default: 50)
        """
        import subprocess
        
        # Pre-flight checks and directory preparation
        output_dir = os.path.dirname(video_name)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Validate input array
        if bp1_block is None or bp1_block.shape[0] == 0:
            raise ValueError(f"Input array is empty or None for {video_name}")
        
        # Get frame dimensions
        H, W = bp1_block.shape[1], bp1_block.shape[2]
        T = bp1_block.shape[0]
        
        # Detect if this is a Dask array
        is_dask_array = hasattr(bp1_block, 'compute') and callable(getattr(bp1_block, 'compute'))
        
        # Validate data type by checking first frame
        if is_dask_array:
            sample_frame = bp1_block[0:1].compute()
        else:
            sample_frame = bp1_block[0:1]
        
        expected_dtype = np.uint16
        if sample_frame.dtype != expected_dtype:
            print(f"Warning: Expected dtype {expected_dtype}, got {sample_frame.dtype}. Converting...")
        
        # Build FFmpeg command with enhanced parameters
        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray16le', 
               '-s', f'{W}x{H}', '-r', '25', '-i', 'pipe:0']
        
        # Add encoder arguments
        for key, value in writer_args.items():
            if key not in ['s']:  # Skip dimensions as already set
                cmd.extend([f'-{key}', str(value)])
        
        # Add movflags for MP4 to ensure moov atom is written properly
        if video_name.endswith('.mp4'):
            cmd.extend(['-movflags', '+faststart'])
        
        # Use 'error' level for better diagnostics while still being relatively quiet
        cmd.extend(['-loglevel', 'error', video_name])
        
        # Start FFmpeg process with larger buffer
        process = subprocess.Popen(cmd, 
                                 stdin=subprocess.PIPE, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 bufsize=10**8)  # 100MB buffer
        
        stdin_closed = False
        try:
            # Stream frames in chunks to reduce memory usage
            bytes_written = 0
            for i in range(0, T, chunk_size):
                end_idx = min(i + chunk_size, T)
                
                # Get chunk of frames
                if is_dask_array:
                    chunk = bp1_block[i:end_idx].compute()
                else:
                    chunk = bp1_block[i:end_idx]
                
                # Ensure correct dtype
                if chunk.dtype != expected_dtype:
                    chunk = chunk.astype(expected_dtype)
                
                # Write to FFmpeg stdin
                chunk_bytes = chunk.tobytes()
                try:
                    process.stdin.write(chunk_bytes)
                    process.stdin.flush()  # Explicit flush after each chunk
                    bytes_written += len(chunk_bytes)
                except BrokenPipeError:
                    # FFmpeg closed the pipe - get error message
                    stdin_closed = True
                    stdout, stderr = process.wait(timeout=2), process.stderr.read()
                    error_msg = stderr.decode() if stderr else "No error message"
                    raise RuntimeError(f"FFmpeg closed input pipe early: {error_msg}")
                
                # Progress indicator for large files
                if (i + chunk_size) % 500 == 0:
                    print(f"  Encoded {min(i + chunk_size, T)}/{T} frames...")
            
            # Close stdin to signal end of input (if not already closed)
            if not stdin_closed:
                process.stdin.close()
                stdin_closed = True
            
            # Wait for FFmpeg to finish with timeout
            return_code = process.wait(timeout=60)
            
            # Read any remaining output
            stdout_data = process.stdout.read() if process.stdout else b''
            stderr_data = process.stderr.read() if process.stderr else b''
            
            # Check return code
            if return_code != 0:
                error_msg = stderr_data.decode() if stderr_data else "No error message from FFmpeg"
                raise RuntimeError(f"FFmpeg failed with return code {return_code} for {video_name}. Error: {error_msg}")
            
            # Validate output file exists and has reasonable size
            if not os.path.exists(video_name):
                raise FileNotFoundError(f"FFmpeg completed but output file not found: {video_name}")
            
            file_size = os.path.getsize(video_name)
            if file_size < 1000:  # Less than 1KB is suspiciously small
                raise ValueError(f"Output file is suspiciously small ({file_size} bytes): {video_name}")
            
            # Success message
            print(f"  Successfully encoded {T} frames ({bytes_written / (1024*1024):.1f} MB) to {os.path.basename(video_name)}")
            
        except subprocess.TimeoutExpired:
            # Timeout waiting for process
            process.kill()
            process.wait(timeout=5)  # Give it time to die
            raise TimeoutError(f"FFmpeg timed out encoding {video_name}")
            
        except Exception as e:
            # For any other exception, ensure process is terminated
            if process.poll() is None:  # Process is still running
                try:
                    if not stdin_closed and process.stdin:
                        try:
                            process.stdin.close()
                        except:
                            pass  # Ignore errors closing stdin
                    process.terminate()  # Try graceful termination first
                    process.wait(timeout=2)
                except:
                    process.kill()  # Force kill if terminate doesn't work
                    try:
                        process.wait(timeout=5)
                    except:
                        pass  # Process might be stuck
            
            # Re-raise the original exception with context
            if "FFmpeg" not in str(e):
                raise RuntimeError(f"Video encoding failed for {video_name}: {str(e)}") from e
            else:
                raise  # Re-raise if it's already an FFmpeg error
        finally:
            # Ensure all pipes are closed
            if process.stdin and not stdin_closed:
                try:
                    process.stdin.close()
                except:
                    pass
            if process.stdout:
                try:
                    process.stdout.close()
                except:
                    pass
            if process.stderr:
                try:
                    process.stderr.close()
                except:
                    pass

    def validate_video_file(self, video_path):
        """
        Validate that a video file is properly formatted and readable.
        
        Args:
            video_path: Path to the video file to validate
            
        Returns:
            bool: True if valid, raises exception if invalid
        """
        import subprocess
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check file size
        file_size = os.path.getsize(video_path)
        if file_size < 1000:
            raise ValueError(f"Video file is too small ({file_size} bytes): {video_path}")
        
        # Use ffprobe to validate the video structure
        cmd = ['ffprobe', '-v', 'error', '-show_format', '-show_streams', video_path]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                raise ValueError(f"Video file appears corrupted. FFprobe error: {result.stderr}")
            
            # Check for moov atom in MP4 files
            if video_path.endswith('.mp4'):
                # Look for format information in the output
                if 'format' not in result.stdout.lower():
                    raise ValueError(f"MP4 file missing format information (possibly missing moov atom): {video_path}")
            
            return True
            
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Video validation timed out for: {video_path}")
        except FileNotFoundError:
            print("Warning: ffprobe not found. Skipping detailed video validation.")
            return True  # Assume valid if we can't validate

    def _shuffle_bytes(self, data: bytes, itemsize: int = 2) -> bytes:
        """
        Byte shuffle for better compression of numeric arrays.
        Groups bytes by position within each element (all high bytes, then all low bytes).
        """
        arr = np.frombuffer(data, dtype=np.uint8)
        n_elements = len(arr) // itemsize
        if n_elements == 0:
            return data
        reshaped = arr[:n_elements * itemsize].reshape(n_elements, itemsize)
        shuffled = reshaped.T.flatten()
        return shuffled.tobytes()

    def _unshuffle_bytes(self, data: bytes, itemsize: int = 2) -> bytes:
        """Reverse the byte shuffle operation."""
        arr = np.frombuffer(data, dtype=np.uint8)
        n_elements = len(arr) // itemsize
        if n_elements == 0:
            return data
        reshaped = arr[:n_elements * itemsize].reshape(itemsize, n_elements)
        unshuffled = reshaped.T.flatten()
        return unshuffled.tobytes()

    def zstd_compress(self, compression_level, effect_size, power, compute_dict, use_delta=True, use_shuffle=True):
        """
        Compress data with zstd, optionally using delta encoding and byte shuffling.

        Args:
            compression_level: Zstd compression level (0-22)
            effect_size: Effect size for power analysis (dictionary training)
            power: Statistical power for sample size calculation
            compute_dict: Whether to train and use a zstd dictionary
            use_delta: Apply delta encoding across frames (default True)
            use_shuffle: Apply byte shuffle for better compression (default True)
        """
        # Perform power analysis to find sample size for dictionary training
        analysis = TTestIndPower()
        sample_size = int(analysis.solve_power(effect_size=effect_size, power=power, alpha=0.05))

        print(f'Compressing with zstd level {compression_level} (delta={use_delta}, shuffle={use_shuffle})...')

        for k in range(len(self.bp1)):
            input_file_name1 = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files1[k]))[1])[0]

            # Dictionary training if requested
            bp1_dict = None
            bp2_dict = None
            if compute_dict:
                bp1_samples = np.random.choice(self.bp1[k].shape[0], min(sample_size, self.bp1[k].shape[0]), replace=False)
                bp1_samples = self.bp1[k][bp1_samples].flatten().compute()
                bp1_sample_bytes = [sample.tobytes() for sample in bp1_samples]
                print(f'Training dictionary with {len(bp1_sample_bytes)} samples...')
                bp1_dict = zstd.train_dictionary(dict_size=131072, samples=bp1_sample_bytes)

                if not self.single_plane:
                    bp2_samples = np.random.choice(self.bp2[k].shape[0], min(sample_size, self.bp2[k].shape[0]), replace=False)
                    bp2_samples = self.bp2[k][bp2_samples].flatten().compute()
                    bp2_sample_bytes = [sample.tobytes() for sample in bp2_samples]
                    bp2_dict = zstd.train_dictionary(dict_size=131072, samples=bp2_sample_bytes)

            # Compress BP1
            output_file1 = f'{self.output_path}{input_file_name1}_level_{compression_level}.zst'
            self._zstd_compress_array(self.bp1[k], output_file1, compression_level,
                                      bp1_dict, use_delta, use_shuffle)

            # Compress BP2 if biplane
            if not self.single_plane:
                input_file_name2 = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files2[k]))[1])[0]
                output_file2 = f'{self.output_path}{input_file_name2}_level_{compression_level}.zst'
                self._zstd_compress_array(self.bp2[k], output_file2, compression_level,
                                          bp2_dict, use_delta, use_shuffle)

    def _zstd_compress_array(self, data_array, output_file, compression_level, zstd_dict, use_delta, use_shuffle):
        """
        Compress a single dask array to a zst file with optional delta encoding and shuffle.
        """
        try:
            with open(output_file, 'wb') as f:
                # Build flags byte (backward compatible with old format)
                flags = 0
                if zstd_dict is not None:
                    flags |= 0x01  # bit 0: dictionary present
                if use_delta:
                    flags |= 0x02  # bit 1: delta encoded
                if use_shuffle:
                    flags |= 0x04  # bit 2: shuffled

                # Write flags byte
                f.write(np.array([flags], dtype=np.uint8).tobytes())

                # Write dictionary if present
                if zstd_dict is not None:
                    f.write(np.array([len(zstd_dict)], dtype=np.int32).tobytes())
                    f.write(zstd_dict.as_bytes())

                # Write shape and dtype
                f.write(np.array(data_array.shape, dtype=np.int32).tobytes())
                f.write(np.array(str(data_array.dtype), dtype='S20').tobytes())

                # Setup compressor
                if zstd_dict is not None:
                    cctx = zstd.ZstdCompressor(dict_data=zstd_dict, level=compression_level)
                else:
                    cctx = zstd.ZstdCompressor(level=compression_level)

                compressor = cctx.stream_writer(f, write_size=32768)

                # Determine itemsize for shuffle
                itemsize = data_array.dtype.itemsize

                # Process frames with delta encoding and progress bar
                prev_frame = None
                original_size = 0
                n_frames = data_array.shape[0]
                pbar = tqdm(total=n_frames, desc=f'  Compressing {os.path.basename(output_file)}', unit='frame')
                for chunk in data_array:
                    frame = chunk.compute()
                    original_size += frame.nbytes
                    pbar.update(1)

                    if use_delta and prev_frame is not None:
                        # Delta encode: store difference from previous frame as int16
                        delta = (frame.astype(np.int32) - prev_frame.astype(np.int32)).astype(np.int16)
                        data_to_compress = delta
                        itemsize_for_shuffle = 2  # int16
                    else:
                        data_to_compress = frame
                        itemsize_for_shuffle = itemsize

                    prev_frame = frame.copy()

                    # Convert to bytes
                    frame_bytes = data_to_compress.tobytes()

                    # Apply byte shuffle
                    if use_shuffle:
                        frame_bytes = self._shuffle_bytes(frame_bytes, itemsize=itemsize_for_shuffle)

                    compressor.write(frame_bytes)

                pbar.close()
                compressor.flush(zstd.FLUSH_FRAME)

            # Print compression stats
            compressed_size = os.path.getsize(output_file)
            ratio = original_size / compressed_size if compressed_size > 0 else 0
            print(f'  {os.path.basename(output_file)}: {original_size/1e6:.2f}MB -> {compressed_size/1e6:.2f}MB ({ratio:.1f}x)')

        except Exception as e:
            print(f"Error during compression of {output_file}: {e}")

    def _detect_individual_ifd_variation(self, individual_ifds):
        """
        Detect if individual IFD metadata varies significantly between frames.
        Returns True if frame-by-frame writing is needed.
        """
        if not individual_ifds or len(individual_ifds) <= 1:
            return False
        
        # Get reference metadata from first frame
        first_frame_tags = individual_ifds[0].get('tags', {})
        
        # List of tags that are allowed to vary between frames (these don't count as "significant variation")
        frame_varying_tags = {
            'StripOffsets', 'StripByteCounts', 'TileOffsets', 'TileByteCounts',
            'tag_273', 'tag_279', 'tag_324', 'tag_325',  # Numeric equivalents
            'PageNumber', 'tag_297'
        }
        
        # Check for significant variations
        variation_count = 0
        checked_tags = set()
        
        for frame_idx, ifd in enumerate(individual_ifds[1:], 1):  # Skip first frame
            frame_tags = ifd.get('tags', {})
            
            # Check for tags that exist in first frame but not in this frame
            for tag_name, tag_value in first_frame_tags.items():
                if tag_name in frame_varying_tags:
                    continue
                    
                if tag_name not in checked_tags:
                    checked_tags.add(tag_name)
                    
                    if tag_name not in frame_tags:
                        variation_count += 1
                    elif frame_tags[tag_name] != tag_value:
                        variation_count += 1
            
            # Check for tags that exist in this frame but not in first frame
            for tag_name in frame_tags:
                if tag_name in frame_varying_tags or tag_name in first_frame_tags:
                    continue
                if tag_name not in checked_tags:
                    checked_tags.add(tag_name)
                    variation_count += 1
        
        # If we found significant variations, recommend frame-by-frame writing
        significant_variation = variation_count > 0
        if significant_variation:
            print(f"Found {variation_count} significant IFD variations across {len(individual_ifds)} frames")
        
        return significant_variation

    def save_npz_with_metadata(self, filepath, sparse_matrix, metadata_entry=None):
        """
        Save sparse matrix with zstd compression and delta-encoded coordinates.
        """
        try:
            # Convert sparse matrix to COO format and compute if needed
            if hasattr(sparse_matrix, 'compute'):
                sparse_data = sparse_matrix.compute()
            else:
                sparse_data = sparse_matrix

            # Ensure it's in COO format
            if hasattr(sparse_data, 'tocoo') and not isinstance(sparse_data, sparse.COO):
                sparse_data = sparse_data.tocoo()

            # Delta encode coordinates for better compression
            coords = sparse_data.coords
            delta_coords = self._delta_encode_coords(coords)

            data_to_save = {
                'data': sparse_data.data,
                'delta_coords': delta_coords,
                'shape': np.array(sparse_data.shape),
                'encoding': 'delta',
            }

            # Embed metadata only if the flag is not set and metadata exists
            if metadata_entry and not self.save_metadata_to_json:
                serialized_metadata = serialize_metadata([metadata_entry])[0]
                data_to_save['metadata'] = json.dumps(serialized_metadata, ensure_ascii=False)

            # Save with zstd compression
            self._save_with_zstd(filepath, data_to_save)

        except Exception as e:
            print(f'Error saving NPZ to {filepath}: {e}')
            # Fallback to standard sparse save
            sparse.save_npz(filepath, sparse_matrix)

    def _delta_encode_coords(self, coords):
        """
        Delta encode coordinates for better compression.
        Peaks often persist across frames, so frame indices have runs.
        Delta encoding reduces entropy significantly.
        """
        if coords.shape[1] == 0:
            return coords.astype(np.int32)

        # Sort by frame index first for better delta compression
        sort_idx = np.lexsort((coords[2], coords[1], coords[0]))
        sorted_coords = coords[:, sort_idx]

        # Delta encode: first value is absolute, rest are deltas
        delta = np.zeros_like(sorted_coords, dtype=np.int32)
        delta[:, 0] = sorted_coords[:, 0]
        delta[:, 1:] = np.diff(sorted_coords, axis=1)

        return delta

    def get_encoded_video_path(self, input_stem, compression_level, codec, custom_file_extension=None):
        """Resolve the output video path for a given input stem and codec."""
        if codec == "prores":
            ext = "mov"
        elif codec == "ffv1":
            ext = "avi"
        elif codec == "user":
            ext = custom_file_extension or "mp4"
        else:
            ext = "mp4"
        return f"{self.output_path}{input_stem}_compression_level_{compression_level}.{ext}"

    def get_residual_path(self, video_path):
        return get_residual_path_for_video(video_path)

    def iter_decoded_video_chunks(self, video_path, chunk_size):
        """Stream-decode a video into uint16 chunks of up to chunk_size frames each.

        Yields ``(start, end, chunk_uint16)`` tuples covering the whole video.
        Never materialises more than ``chunk_size`` frames at once.
        """
        container = av.open(video_path)
        try:
            video_stream = container.streams.video[0]
            buf = []
            start = 0
            for frame in container.decode(video_stream):
                buf.append(frame.to_ndarray(format="gray16le"))
                if len(buf) == chunk_size:
                    arr = np.stack(buf, axis=0).astype(np.uint16, copy=False)
                    end = start + arr.shape[0]
                    yield start, end, arr
                    start = end
                    buf = []
            if buf:
                arr = np.stack(buf, axis=0).astype(np.uint16, copy=False)
                end = start + arr.shape[0]
                yield start, end, arr
        finally:
            container.close()

    def iter_original_chunks(self, dask_block, chunk_size):
        """Yield ``(start, end, chunk_array)`` slices of an original Dask block."""
        T = int(dask_block.shape[0])
        for t0 in range(0, T, chunk_size):
            t1 = min(T, t0 + chunk_size)
            sl = dask_block[t0:t1]
            arr = sl.compute() if hasattr(sl, "compute") else np.asarray(sl)
            yield t0, t1, np.asarray(arr)

    def _load_sparse_roi_coords(self, npz_path):
        """Return ``{'coords': (3,N), 'shape': (T,H,W)}`` for a sparse ROI NPZ.

        Does NOT densify. Returns ``None`` if the file cannot be read.
        """
        try:
            with open(npz_path, "rb") as f:
                file_bytes = f.read()
            if file_bytes[:4] == b"\x28\xb5\x2f\xfd":
                dctx = zstd.ZstdDecompressor()
                npz_data = np.load(io.BytesIO(dctx.decompress(file_bytes)), allow_pickle=True)
            else:
                npz_data = np.load(npz_path, allow_pickle=True)
            if "encoding" in npz_data and str(npz_data["encoding"]) == "delta":
                shape = tuple(int(x) for x in npz_data["shape"])
                coords = np.cumsum(npz_data["delta_coords"], axis=1)
                return {"coords": np.asarray(coords), "shape": shape}
            if "data" in npz_data and "coords" in npz_data and "shape" in npz_data:
                shape = tuple(int(x) for x in npz_data["shape"])
                return {"coords": np.asarray(npz_data["coords"]), "shape": shape}
            coo = sparse.load_npz(npz_path)
            if hasattr(coo, "coords") and hasattr(coo, "shape"):
                return {"coords": np.asarray(coo.coords), "shape": tuple(coo.shape)}
            return None
        except Exception as e:
            print(f"  Warning: could not load sparse ROI {npz_path}: {e}")
            return None

    def load_sparse_roi_for_chunk(self, roi_info, frame_start, frame_end, frame_shape):
        """Build a small bool ROI mask for frames in ``[frame_start, frame_end)``."""
        H, W = frame_shape
        nT = frame_end - frame_start
        if roi_info is None:
            return np.zeros((nT, H, W), dtype=bool)
        coords = roi_info["coords"]
        if coords.size == 0:
            return np.zeros((nT, H, W), dtype=bool)
        sel = (coords[0] >= frame_start) & (coords[0] < frame_end)
        if not sel.any():
            return np.zeros((nT, H, W), dtype=bool)
        cc = coords[:, sel]
        mask = np.zeros((nT, H, W), dtype=bool)
        mask[cc[0] - frame_start, cc[1], cc[2]] = True
        return mask

    def compute_residual_chunk(self, original_chunk, decoded_chunk, mode, roi_chunk_mask=None,
                               context=None):
        """Compute the int32 residual chunk; raises on shape mismatch."""
        if original_chunk.shape != decoded_chunk.shape:
            raise ValueError(
                f"original/decoded chunk shape mismatch: "
                f"original={original_chunk.shape} decoded={decoded_chunk.shape}"
                + (f" (context: {context})" if context else "")
            )
        residual = original_chunk.astype(np.int32) - decoded_chunk.astype(np.int32)
        if mode == "background" and roi_chunk_mask is not None:
            if roi_chunk_mask.shape != residual.shape:
                raise ValueError(
                    f"ROI mask chunk shape {roi_chunk_mask.shape} != residual {residual.shape}"
                    + (f" (context: {context})" if context else "")
                )
            residual = np.where(roi_chunk_mask, np.int32(0), residual)
        return residual

    def compute_residuals(self, original_frames, decoded_frames, mode, sparse_mask_dense=None):
        """Whole-array convenience wrapper that delegates to compute_residual_chunk."""
        return self.compute_residual_chunk(original_frames, decoded_frames, mode,
                                           roi_chunk_mask=(sparse_mask_dense.astype(bool)
                                                           if sparse_mask_dense is not None else None))

    def save_residual_sidecar(self, path, residual, original_dtype, mode, codec, compression_level):
        save_residual_sidecar(path, residual, original_dtype, mode, codec, compression_level)

    def _resolve_residual_mode(self):
        """Translate residual_mode + find_roi into a concrete mode.

        - ``auto`` resolves to ``background`` when ROI is on, else ``full``.
        - ``full`` is always honoured, even with ROI on (the residual will
          carry the ROI pixels redundantly, which the user explicitly asked for).
        - ``background`` requires ``find_roi=True``; otherwise we raise rather
          than silently rewriting the user's request.
        """
        req = self.residual_mode
        if req == "auto":
            return "background" if self.find_roi else "full"
        if req == "full":
            return "full"
        if req == "background":
            if not self.find_roi:
                raise ValueError(
                    'residual_mode="background" requires find_roi=True; '
                    'no ROI mask is available to skip when ROI detection is off'
                )
            return "background"
        raise ValueError(f"unknown residual_mode: {req!r}")

    def _save_residuals_for_run(self, codec, compression_level, custom_file_extension=None):
        """After encode(), compute and save residual sidecars for each output video."""
        if codec == "zstd":
            print("Warning: residuals not supported for zstd codec; skipping.")
            return
        if codec == "ffv1":
            print("Skipping residuals for ffv1 (codec is already 16-bit lossless).")
            return
        if codec == "user":
            print("Warning: residuals not supported for user codec; skipping.")
            return

        mode = self._resolve_residual_mode()
        print(f"Computing {mode} residuals from re-decoded videos (chunk_size={self.residual_chunk_size})...")

        for k in range(len(self.bp1)):
            in1 = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files1[k]))[1])[0]
            video1 = self.get_encoded_video_path(in1, compression_level, codec, custom_file_extension)
            self._save_one_residual(video1, k, plane="bp1", mode=mode,
                                    codec=codec, compression_level=compression_level,
                                    input_stem=in1)
            if not self.single_plane:
                in2 = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files2[k]))[1])[0]
                video2 = self.get_encoded_video_path(in2, compression_level, codec, custom_file_extension)
                self._save_one_residual(video2, k, plane="bp2", mode=mode,
                                        codec=codec, compression_level=compression_level,
                                        input_stem=in2)

    def _make_residual_chunk_iter_factory(self, video_path, original_block, mode, roi_info,
                                          chunk_size, H, W):
        """Return a zero-arg factory that produces a fresh joined residual iterator.

        Each call rewinds both the decoded video and the original Dask block
        and re-emits ``(start, end, residual_chunk_int32)`` tuples. The factory
        is what allows the writer to make two passes (dtype scan + write).

        Mismatched frame counts in either direction are raised as
        ``ValueError`` via ``itertools.zip_longest``.
        """
        from itertools import zip_longest
        sentinel = object()

        def factory():
            decoded_iter = self.iter_decoded_video_chunks(video_path, chunk_size)
            original_iter = self.iter_original_chunks(original_block, chunk_size)
            for d_item, o_item in zip_longest(decoded_iter, original_iter, fillvalue=sentinel):
                if d_item is sentinel:
                    or_t0, or_t1, _ = o_item
                    raise ValueError(
                        f"decoded video has fewer frames than original for {video_path}: "
                        f"original still has chunk [{or_t0},{or_t1})"
                    )
                if o_item is sentinel:
                    do_t0, do_t1, _ = d_item
                    raise ValueError(
                        f"decoded video has more frames than original for {video_path}: "
                        f"decoded extra chunk [{do_t0},{do_t1})"
                    )
                do_t0, do_t1, decoded_chunk = d_item
                or_t0, or_t1, orig_chunk = o_item
                if (do_t0, do_t1) != (or_t0, or_t1):
                    raise ValueError(
                        f"chunk index drift: decoded=[{do_t0},{do_t1}) "
                        f"original=[{or_t0},{or_t1}) for {video_path}"
                    )
                if decoded_chunk.shape[1:] != (H, W):
                    raise ValueError(
                        f"decoded frame size {decoded_chunk.shape[1:]} != original ({H},{W}) "
                        f"for {video_path}"
                    )
                roi_chunk_mask = None
                if mode == "background":
                    roi_chunk_mask = self.load_sparse_roi_for_chunk(roi_info, do_t0, do_t1, (H, W))
                yield do_t0, do_t1, self.compute_residual_chunk(
                    orig_chunk, decoded_chunk, mode,
                    roi_chunk_mask=roi_chunk_mask,
                    context=f"{os.path.basename(video_path)} chunk [{do_t0},{do_t1})",
                )

        return factory

    def _scan_residual_dtype(self, chunk_iter_factory):
        """Two-pass helper: scan all chunks once and pick int16 if the data fits."""
        global_min = None
        global_max = None
        for _t0, _t1, chunk in chunk_iter_factory():
            if chunk.size == 0:
                continue
            cmin = int(chunk.min())
            cmax = int(chunk.max())
            global_min = cmin if global_min is None else min(global_min, cmin)
            global_max = cmax if global_max is None else max(global_max, cmax)
        if global_min is None:
            return np.int16  # empty residual, anything works
        if global_min >= -32768 and global_max <= 32767:
            return np.int16
        return np.int32

    def _save_one_residual(self, video_path, k, plane, mode, codec, compression_level, input_stem):
        if not os.path.exists(video_path):
            print(f"  Warning: video not found, skipping residual: {video_path}")
            return

        original_block = self.bp1[k] if plane == "bp1" else self.bp2[k]
        T = int(original_block.shape[0])
        H = int(original_block.shape[1])
        W = int(original_block.shape[2])
        chunk_size = self.residual_chunk_size

        roi_info = None
        if mode == "background":
            npz_path = os.path.join(self.output_path, input_stem + ".npz")
            if os.path.exists(npz_path):
                roi_info = self._load_sparse_roi_coords(npz_path)
                if roi_info is not None and roi_info["shape"] != (T, H, W):
                    raise ValueError(
                        f"ROI sparse {npz_path} shape {roi_info['shape']} != "
                        f"original ({T},{H},{W}); cannot mask residual"
                    )

        out_path = self.get_residual_path(video_path)
        orig_dtype = original_block.dtype

        chunk_factory = self._make_residual_chunk_iter_factory(
            video_path, original_block, mode, roi_info, chunk_size, H, W,
        )

        # Pass 1: pick int16 vs int32 from the actual residual range.
        residual_dtype = self._scan_residual_dtype(chunk_factory)
        print(f"  Selected residual dtype {np.dtype(residual_dtype).name} for {os.path.basename(video_path)}")

        # Pass 2: write the sidecar with that dtype.
        try:
            save_residual_sidecar_streaming(
                out_path, shape=(T, H, W), original_dtype=orig_dtype, mode=mode,
                codec=codec, compression_level=compression_level,
                chunk_iter=chunk_factory(), chunk_size=chunk_size,
                residual_dtype=residual_dtype,
            )
        except Exception as e:
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            print(f"  ERROR: failed to save residual {out_path}: {e}")
            raise
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  Saved residual: {os.path.basename(out_path)} ({size_mb:.2f} MB)")

    def find_matching_sidecars_for_video(self, video_path):
        return find_matching_sidecars_for_video(video_path, search_dir=self.output_path)

    def _save_with_zstd(self, filepath, data_dict, level=19):
        """Save data dictionary with zstd compression."""
        import io

        # First save to NPZ in memory
        buffer = io.BytesIO()
        np.savez(buffer, **data_dict)
        npz_bytes = buffer.getvalue()

        # Compress with zstd (level 19 for high compression)
        cctx = zstd.ZstdCompressor(level=level)
        compressed = cctx.compress(npz_bytes)

        # Write to file
        with open(filepath, 'wb') as f:
            f.write(compressed)

        compression_ratio = 100 * len(compressed) / len(npz_bytes) if len(npz_bytes) > 0 else 0
        print(f'  Saved {os.path.basename(filepath)}: {len(npz_bytes)/1e6:.2f}MB -> {len(compressed)/1e6:.2f}MB ({compression_ratio:.1f}%)')
    
    def save_metadata(self):
        """
        Save extracted metadata to JSON files for later restoration if requested.
        Only saves metadata if extract_metadata=True was set during initialization.
        """
        if not self.extract_metadata_flag:
            print('Metadata extraction was disabled - no metadata to save.')
            return

        if not self.save_metadata_to_json:
            # This is now the default behavior, metadata is embedded in NPZ
            return

        print('Saving metadata to separate JSON files...')
        
        # Serialize metadata
        serialized_bp1 = serialize_metadata(self.metadata_bp1)
        
        # Save BP1 metadata
        metadata_file_bp1 = os.path.join(self.output_path, f'{self.stem}_metadata_bp1.json')
        try:
            with open(metadata_file_bp1, 'w') as f:
                json.dump(serialized_bp1, f, indent=4)
            print(f'Saved BP1 metadata to: {metadata_file_bp1}')
        except Exception as e:
            print(f'Error saving BP1 metadata to {metadata_file_bp1}: {e}')
            
        # Save BP2 metadata if it exists
        if self.metadata_bp2:
            serialized_bp2 = serialize_metadata(self.metadata_bp2)
            metadata_file_bp2 = os.path.join(self.output_path, f'{self.stem}_metadata_bp2.json')
            try:
                with open(metadata_file_bp2, 'w') as f:
                    json.dump(serialized_bp2, f, indent=4)
                print(f'Saved BP2 metadata to: {metadata_file_bp2}')
            except Exception as e:
                print(f'Error saving BP2 metadata to {metadata_file_bp2}: {e}')

    def _print_compression_stats(self):
        """Print breakdown of compressed file sizes."""
        npz_files = glob.glob(f'{self.output_path}*.npz')
        video_files = (glob.glob(f'{self.output_path}*.mp4') +
                      glob.glob(f'{self.output_path}*.avi') +
                      glob.glob(f'{self.output_path}*.mov'))
        residual_files = glob.glob(f'{self.output_path}*{RESIDUAL_SUFFIX}')
        mkv_files = glob.glob(f'{self.output_path}*.mkv')

        def _size(files):
            return sum(os.path.getsize(f) for f in files) if files else 0

        npz_size = _size(npz_files)
        video_size = _size(video_files)
        residual_size = _size(residual_files)
        mkv_size = _size(mkv_files)

        if self.create_single_file and mkv_size > 0:
            total_size = mkv_size
        else:
            total_size = video_size + npz_size + residual_size

        # Estimate original size from dask arrays
        original_size = 0
        try:
            for arr in self.bp1:
                if hasattr(arr, 'nbytes'):
                    original_size += arr.nbytes
                elif hasattr(arr, 'dtype') and hasattr(arr, 'shape'):
                    original_size += arr.dtype.itemsize * np.prod(arr.shape)
        except Exception:
            pass

        print(f"\n=== Compression Statistics ===")
        if video_size > 0:
            print(f"  Video files:    {video_size / 1e6:.1f} MB ({len(video_files)} files)")
        if npz_size > 0:
            print(f"  NPZ files:      {npz_size / 1e6:.1f} MB ({len(npz_files)} files)")
        if residual_size > 0:
            print(f"  Residual files: {residual_size / 1e6:.1f} MB ({len(residual_files)} files)")
        if mkv_size > 0:
            print(f"  MKV files:      {mkv_size / 1e6:.1f} MB ({len(mkv_files)} files)")
        print(f"  Total:          {total_size / 1e6:.1f} MB")
        if original_size > 0 and total_size > 0:
            ratio = original_size / total_size
            print(f"  Compression ratio: {ratio:.1f}x")
        print("=" * 31)

    def run(self,codec:str='x265', compression_level:int=0,custom_dict:dict=None,custom_file_extension:str=None,compute_zstd_dict:bool=False):#,find_peaks:bool=True):
        if codec!='zstd' and compute_zstd_dict:
            print ('Warning: Dictionary computation is only supported for Zstandard compression. Ignoring compute_zstd_dict flag.')
            compute_zstd_dict = False
        
        if (self.find_roi) and (codec =='zstd'):
            print ('Warning: ROI detection is not supported for Zstandard compression. Ignoring find_roi flag.')
            self.find_roi = False

        # Determine if codec is truly lossless (no data loss)
        # ffv1 with gray16le preserves all 16 bits, so sparse matrix is redundant
        is_truly_lossless = (codec == 'ffv1')

        if self.find_roi and not is_truly_lossless:
            self.deflate()
            gc.collect()
        elif is_truly_lossless and self.find_roi:
            print('Skipping sparse matrix storage - ffv1 codec is truly lossless (16-bit preserved).')
        if codec in ['x265', 'av1', 'x264', 'ffv1', 'prores', 'user']:
            assert compression_level in [0,1,2,3], 'Error: Compression level for ffmpeg based compression must be 0, 1, 2 or 3.'
            self.encode(codec=codec, compression_lvl=compression_level,custom_dict=custom_dict,custom_file_extension=custom_file_extension)
        elif codec=='zstd':
            assert compression_level <= 22, 'Error: Valid compression levels for Zstandard compression are all negative integers through 22.'
            self.zstd_compress(compression_level=compression_level, effect_size=0.5, power=0.95, compute_dict=compute_zstd_dict)
        else:
            raise ValueError(f'Unsupported codec: {codec}.')

        # Optional exact-reconstruction residual sidecars for video codecs
        if self.save_residuals and codec in ('x265', 'av1', 'x264', 'prores', 'ffv1', 'zstd', 'user'):
            self._save_residuals_for_run(codec, compression_level, custom_file_extension=custom_file_extension)

        # Create single MKV file if requested
        if self.create_single_file:
            self.package_to_mkv(codec, compression_level)

        # Print compression statistics
        self._print_compression_stats()

    def package_to_mkv(self, codec, compression_level):
        """
        Package the video file and NPZ files into a single MKV container.
        Uses ffmpeg-python to create an MKV file with the video as the main track
        and NPZ files as attachments.
        """
        # Skip MKV packaging for zstd codec (creates .zst files, not video files)
        if codec == 'zstd':
            print('MKV packaging not applicable for zstd codec (creates .zst files, not video files).')
            return
            
        print('Packaging files into single MKV container...')
        
        # Determine video file extension based on codec
        if codec == 'prores':
            video_ext = 'mov'
        elif codec == 'ffv1':
            video_ext = 'avi'
        else:
            video_ext = 'mp4'  # Default for x265, av1, x264, etc.
        
        # Find all generated video files (following the actual naming pattern)
        video_files = []
        mkv_files_to_create = []
        
        for k in range(len(self.processed_bp1)):
            input_file_name1 = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files1[k]))[1])[0]
            video_name1 = f'{self.output_path}{input_file_name1}_compression_level_{compression_level}.{video_ext}'
            mkv_name1 = f'{self.output_path}{input_file_name1}_compression_level_{compression_level}.mkv'
            
            if os.path.exists(video_name1):
                # Validate video file before adding to list
                try:
                    self.validate_video_file(video_name1)
                    video_files.append(video_name1)
                    mkv_files_to_create.append(mkv_name1)
                except Exception as e:
                    print(f"Warning: Skipping invalid video file {video_name1}: {e}")
                    continue
            
            # Handle BP2 if not single plane
            if not self.single_plane:
                input_file_name2 = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files2[k]))[1])[0]
                video_name2 = f'{self.output_path}{input_file_name2}_compression_level_{compression_level}.{video_ext}'
                mkv_name2 = f'{self.output_path}{input_file_name2}_compression_level_{compression_level}.mkv'
                
                if os.path.exists(video_name2):
                    # Validate video file before adding to list
                    try:
                        self.validate_video_file(video_name2)
                        video_files.append(video_name2)
                        mkv_files_to_create.append(mkv_name2)
                    except Exception as e:
                        print(f"Warning: Skipping invalid video file {video_name2}: {e}")
                        continue
        
        if not video_files:
            print('Warning: No video files found. Skipping MKV packaging.')
            return
        
        # Create MKV files for each video file, attaching only matching sidecars
        import subprocess
        successful_count = 0
        failed_count = 0

        for video_file, mkv_file in zip(video_files, mkv_files_to_create):
            try:
                print(f'  Creating MKV: {os.path.basename(mkv_file)}...')

                # Per-video sidecar selection: only attach files matching this stem
                sidecars = self.find_matching_sidecars_for_video(video_file)
                attachments = [p for p in (sidecars.get('npz'), sidecars.get('residual')) if p]

                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_file,
                ]

                for i, attach_path in enumerate(attachments):
                    cmd.extend(['-attach', attach_path])
                    cmd.extend(['-metadata:s:t:{}'.format(i), 'mimetype=application/octet-stream'])
                    cmd.extend(['-metadata:s:t:{}'.format(i), 'filename={}'.format(os.path.basename(attach_path))])

                cmd.extend([
                    '-c', 'copy',
                    '-map', '0:0',
                    '-loglevel', 'error',
                    mkv_file
                ])

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                if result.returncode != 0:
                    print(f'  ERROR: FFmpeg failed creating {os.path.basename(mkv_file)}')
                    print(f'  FFmpeg error: {result.stderr}')
                    failed_count += 1

                    if 'moov atom not found' in result.stderr:
                        print(f'  DIAGNOSIS: Input video file {os.path.basename(video_file)} is corrupted (missing moov atom)')
                    elif 'Invalid data found' in result.stderr:
                        print(f'  DIAGNOSIS: Input video file {os.path.basename(video_file)} contains invalid data')

                    continue

                if not os.path.exists(mkv_file):
                    print(f'  ERROR: MKV file was not created: {mkv_file}')
                    failed_count += 1
                    continue

                mkv_size = os.path.getsize(mkv_file)
                video_size = os.path.getsize(video_file)

                if mkv_size < video_size:
                    print(f'  WARNING: MKV file seems too small ({mkv_size} bytes vs video {video_size} bytes)')

                successful_count += 1
                attach_names = ', '.join(os.path.basename(p) for p in attachments) or 'none'
                print(f'  SUCCESS: Created {os.path.basename(mkv_file)} ({mkv_size / (1024*1024):.1f} MB)')
                print(f'    - Video: {os.path.basename(video_file)}')
                print(f'    - Attachments: {attach_names}')
                
            except subprocess.TimeoutExpired:
                print(f'  ERROR: Timeout creating MKV for {os.path.basename(video_file)}')
                failed_count += 1
                continue
                
            except Exception as e:
                print(f'  ERROR: Unexpected error creating MKV for {os.path.basename(video_file)}: {e}')
                failed_count += 1
                continue
        
        # Summary
        print(f'\nMKV Packaging Summary:')
        print(f'  Successful: {successful_count}')
        print(f'  Failed: {failed_count}')
        
        if failed_count > 0:
            print(f'\nWARNING: {failed_count} MKV file(s) could not be created.')
            print('  Check the error messages above for details.')
            print('  The original video and NPZ files are still available.')


class UNSPARZ:
    def __init__(self, path_sparse_bp1:str,
                 path_encoded_bp1:str,
                 stem:str,
                 output_path:str,
                 path_sparse_bp2:str=None,
                 path_encoded_bp2:str=None,
                 use_roi:bool=True,
                 chunk_size:int=10,
                 num_workers:int=4,
                 num_dask_workers:int=2,
                 output_format: str = 'tiff',
                 use_residuals:bool=True,
                 path_residual_bp1:str=None,
                 path_residual_bp2:str=None,
                 streaming:bool=True):

        # Store original parameters
        original_path_sparse_bp1 = path_sparse_bp1
        original_path_sparse_bp2 = path_sparse_bp2

        # Initialize temp directory tracking
        self.temp_dirs_to_cleanup = []

        # Residual configuration
        self.use_residuals = use_residuals
        self.path_residual_bp1 = path_residual_bp1
        self.path_residual_bp2 = path_residual_bp2
        # Set use_roi early so _validate_mkv_sidecars sees the right value.
        self.use_roi = use_roi

        # Check if we have MKV files (single-file format)
        self._mkv_extracted = False
        if self._looks_like_mkv_input(path_encoded_bp1):
            mkv_files = self._resolve_mkv_files(path_encoded_bp1)
            if not mkv_files:
                raise ValueError("MKV files not found")
            print(f'Detected MKV single-file format. Extracting components...')
            self.path_encoded_bp1, self.path_encoded_bp2 = self.extract_from_mkv(mkv_files, path_encoded_bp2)
            # extract_from_mkv populated path_sparse_<plane> / path_residual_<plane>
            path_sparse_bp1 = self.path_sparse_bp1
            path_sparse_bp2 = self.path_sparse_bp2
            self._mkv_extracted = True
            # Fail fast on missing required sidecars rather than passing None downstream.
            self._validate_mkv_sidecars()
        else:
            # Traditional separate files
            self.path_encoded_bp1 = sorted(glob.glob(path_encoded_bp1))
            self.path_encoded_bp2 = sorted(glob.glob(path_encoded_bp2)) if path_encoded_bp2 is not None else None

        self.output_format = output_format.lower()
        if self.output_format not in ['tiff', 'dat']:
            raise ValueError("output_format must be either 'tiff' or 'dat'")

        if not self.path_encoded_bp1:
            raise ValueError("No encoded video files were resolved from path_encoded_bp1")

        first_ext = os.path.splitext(self.path_encoded_bp1[0])[1]
        # zstd-encoded image stacks have no streaming-decode path; they always
        # use the legacy eager Dask flow.
        self.streaming = bool(streaming) and first_ext != '.zst'

        if first_ext == '.zst':
            if self.use_roi:
                print('ROI detection is not supported for Zstandard compressed files. Ignoring use_roi flag.')
                self.use_roi = False
            self.encoded_bp1, self.encoded_bp2 = self.decode_zst(self.path_encoded_bp1, self.path_encoded_bp2)
        elif self.streaming:
            # Streaming path: only probe shape/dtype; never decode frames in __init__.
            self.encoded_bp1 = [_LazyVideoProbe(p) for p in self.path_encoded_bp1]
            self.encoded_bp2 = ([_LazyVideoProbe(p) for p in self.path_encoded_bp2]
                                if self.path_encoded_bp2 else None)
        else:
            self.encoded_bp1, self.encoded_bp2 = self.decode(self.path_encoded_bp1, self.path_encoded_bp2)

        # Track ROI sparse paths for later use (streaming uses them per chunk).
        self._path_sparse_bp1_resolved = path_sparse_bp1
        self._path_sparse_bp2_resolved = path_sparse_bp2

        # Optional exact-reconstruction residual sidecars.
        if self.use_residuals and first_ext != '.zst':
            self._autodetect_residual_paths()
            if not self.streaming:
                # Eager Dask apply only; streaming applies per chunk in run().
                self._apply_residuals_to_encoded()

        self.shapes = [x.shape[:3] for x in self.encoded_bp1]
        if self.use_roi and not self.streaming:
            self.sparse_bp1, self.sparse_bp2 = self.load_sparse(path_sparse_bp1, path_sparse_bp2, self.shapes)
            self.processed_bp1, self.processed_bp2 = self.process_frames()
        else:
            # Streaming: ROI is loaded coords-only per video at run() time.
            self.sparse_bp1 = None
            self.sparse_bp2 = None
            self.processed_bp1 = None
            self.processed_bp2 = None
        self.stem = stem
        self.num_workers = num_workers
        self.num_dask_workers = num_dask_workers
        if output_path[-1] != '/':
            self.output_path = output_path+"/"
        else:
            self.output_path = output_path
        self.chunk_size = chunk_size
        
        # Extract metadata from encoded files for preservation
        self.metadata_bp1, self.metadata_bp2 = self.load_original_metadata()
        
        # Restore numeric values
        if self.metadata_bp1:
            self.metadata_bp1 = [restore_numeric_values(meta) for meta in self.metadata_bp1]
        if self.metadata_bp2:
            self.metadata_bp2 = [restore_numeric_values(meta) for meta in self.metadata_bp2]

    def _extract_mkv_attachments(self, mkv_file, temp_dir):
        """Extract every attachment in a single MKV into ``temp_dir``.

        Returns ``(npz_count, residual_count)`` for the labelling-only summary.
        Each attachment is written under its declared filename so the standard
        sidecar matcher can find it later.
        """
        import subprocess
        try:
            probe = ffmpeg.probe(mkv_file)
        except ffmpeg.Error as e:
            print(f'Error probing MKV {mkv_file}: {e}')
            if getattr(e, "stderr", None):
                print(f'FFmpeg stderr: {e.stderr.decode()}')
            return 0, 0

        attachment_count = 0
        npz_extracted = 0
        residual_extracted = 0
        for stream in probe.get('streams', []):
            if stream.get('codec_type') != 'attachment':
                continue
            stream_index = stream['index']
            attachment_filename = stream.get('tags', {}).get('filename',
                                                             f'attachment_{attachment_count}.bin')
            attachment_path = os.path.join(temp_dir, attachment_filename)
            kind = ('residual' if attachment_filename.endswith(RESIDUAL_SUFFIX)
                    else ('npz' if attachment_filename.endswith('.npz') else 'attachment'))

            if os.path.exists(attachment_path):
                print(f'{kind} already extracted: {attachment_filename}')
                attachment_count += 1
                if kind == 'npz':
                    npz_extracted += 1
                elif kind == 'residual':
                    residual_extracted += 1
                continue

            cmd = [
                'ffmpeg', '-dump_attachment:t:{}'.format(attachment_count), attachment_path,
                '-i', mkv_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            success = result.returncode == 0 and os.path.exists(attachment_path)
            if not success:
                cmd2 = [
                    'ffmpeg', '-y', '-i', mkv_file,
                    '-map', f'0:{stream_index}',
                    '-c', 'copy', attachment_path
                ]
                result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=10)
                success = result2.returncode == 0 and os.path.exists(attachment_path)
                if not success:
                    cmd3 = [
                        'ffmpeg', '-i', mkv_file,
                        '-dump_attachment:t', attachment_path
                    ]
                    result3 = subprocess.run(cmd3, capture_output=True, text=True, timeout=10)
                    success = result3.returncode == 0 and os.path.exists(attachment_path)
                    if not success:
                        print(f'Failed to extract {attachment_filename}')
                        if result2.stderr:
                            print(f'Error: {result2.stderr[:200]}')

            if success:
                print(f'Extracted {kind}: {attachment_filename}')
                attachment_count += 1
                if kind == 'npz':
                    npz_extracted += 1
                elif kind == 'residual':
                    residual_extracted += 1
        return npz_extracted, residual_extracted

    def _extract_one_mkv(self, mkv_file):
        """Extract one MKV into a per-MKV temp dir and locate its sidecars.

        Returns ``{'video', 'npz', 'residual', 'temp_dir', 'mkv_dir'}``. Sidecar
        entries may be ``None`` when no matching file is present.

        The output container is chosen from the MKV's video codec
        (h264/hevc/av1 → .mp4, ffv1 → .mkv, prores → .mov, anything else → .mkv)
        because plain ``.mp4`` cannot hold ffv1/prores in a stream-copy mux.
        """
        print(f'Extracting from MKV: {mkv_file}')
        mkv_dir = os.path.dirname(mkv_file)
        temp_dir = os.path.join(mkv_dir or '.', 'mkv_temp')
        os.makedirs(temp_dir, exist_ok=True)
        if temp_dir not in self.temp_dirs_to_cleanup:
            self.temp_dirs_to_cleanup.append(temp_dir)

        # Probe codec to pick a compatible container/extension.
        codec_name = None
        try:
            probe = ffmpeg.probe(mkv_file)
            video_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'video']
            if video_streams:
                codec_name = video_streams[0].get('codec_name')
        except Exception:
            codec_name = None
        ext, container_format = _container_for_mkv_video_codec(codec_name)
        if codec_name:
            print(f'  Source codec: {codec_name} → extracting to .{ext}')

        video_name = os.path.splitext(os.path.basename(mkv_file))[0] + '.' + ext
        video_path = os.path.join(temp_dir, video_name)

        try:
            if os.path.exists(video_path):
                print(f'Video already extracted: {video_name}')
            else:
                ffmpeg.input(mkv_file).output(
                    video_path, vcodec='copy', format=container_format
                ).run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
                print(f'Extracted video: {video_name}')
            n_npz, n_res = self._extract_mkv_attachments(mkv_file, temp_dir)
            print(f'Extracted {n_npz} NPZ + {n_res} residual sidecar(s) from '
                  f'{os.path.basename(mkv_file)}')
        except ffmpeg.Error as e:
            print(f'Error extracting from MKV {mkv_file}: {e}')
            if getattr(e, "stderr", None):
                print(f'FFmpeg stderr: {e.stderr.decode()}')
        except Exception as e:
            print(f'Unexpected error extracting from MKV {mkv_file}: {e}')

        # Match sidecars by stem in the temp dir first, then the MKV dir.
        sidecars = find_matching_sidecars_for_video(video_path, search_dir=temp_dir)
        if not sidecars["npz"] and mkv_dir:
            fallback = find_matching_sidecars_for_video(video_path, search_dir=mkv_dir)
            if fallback["npz"]:
                sidecars["npz"] = fallback["npz"]
            if fallback["residual"] and not sidecars["residual"]:
                sidecars["residual"] = fallback["residual"]

        return {
            "video": video_path,
            "npz": sidecars["npz"],
            "residual": sidecars["residual"],
            "temp_dir": temp_dir,
            "mkv_dir": mkv_dir,
        }

    def _extract_mkv_group(self, mkv_files):
        """Extract a group of MKV files (one plane). Returns a list of records."""
        return [self._extract_one_mkv(m) for m in mkv_files]

    def _wire_sidecars_from_records(self, records, plane):
        """Assign per-plane path_sparse / path_residual based on stem-matched records."""
        npz_paths = [r["npz"] for r in records]
        res_paths = [r["residual"] for r in records]
        n_npz = sum(1 for p in npz_paths if p)
        n_res = sum(1 for p in res_paths if p)
        print(f'  {plane}: matched {n_npz} NPZ and {n_res} residual sidecar(s) '
              f'to {len(records)} video(s)')

        if n_npz == len(records) and len(records) > 0:
            sparse_paths = npz_paths
        elif n_npz > 0:
            print(f'  Warning: only {n_npz}/{len(records)} NPZ sidecars found for {plane}; '
                  'use_roi will be unavailable for this plane')
            sparse_paths = None
        else:
            sparse_paths = None
        setattr(self, f'path_sparse_{plane}', sparse_paths)

        # Only override an explicitly-provided path_residual_<plane> when we
        # found at least one residual. None entries (gaps) survive into the
        # apply step which already tolerates them.
        if n_res > 0 and getattr(self, f'path_residual_{plane}') is None:
            setattr(self, f'path_residual_{plane}', res_paths)

    @staticmethod
    def _looks_like_mkv_input(path_or_glob):
        """Return True if path_or_glob looks like it points at MKV files."""
        if path_or_glob is None:
            return False
        if isinstance(path_or_glob, (list, tuple)):
            return any(str(p).endswith('.mkv') for p in path_or_glob)
        s = str(path_or_glob)
        if '.mkv' in s:
            return bool(glob.glob(s))
        # ``*.mp4`` glob may be present alongside MKV-packaged versions.
        alt = s.replace('.mp4', '.mkv').replace('*.mp4', '*.mkv')
        return bool(glob.glob(alt))

    @staticmethod
    def _resolve_mkv_files(path_or_glob):
        """Resolve a string or list to a sorted list of MKV files."""
        if path_or_glob is None:
            return []
        if isinstance(path_or_glob, (list, tuple)):
            return sorted([p for p in path_or_glob if str(p).endswith('.mkv')])
        s = str(path_or_glob)
        if '.mkv' in s:
            return sorted(glob.glob(s))
        alt = s.replace('.mp4', '.mkv').replace('*.mp4', '*.mkv')
        return sorted(glob.glob(alt))

    def extract_from_mkv(self, mkv_files, path_encoded_bp2):
        """Backward-compatible bp1+bp2 extractor.

        Extracts the bp1 group from ``mkv_files``; if ``path_encoded_bp2`` resolves
        to MKV files, extracts that bp2 group separately. Wires
        ``self.path_sparse_<plane>`` / ``self.path_residual_<plane>`` per plane.
        Returns ``(bp1_videos, bp2_videos_or_None)``.
        """
        bp1_records = self._extract_mkv_group(mkv_files)
        self._wire_sidecars_from_records(bp1_records, plane='bp1')
        bp1_videos = [r["video"] for r in bp1_records]

        bp2_videos = None
        if path_encoded_bp2:
            bp2_mkv_files = self._resolve_mkv_files(path_encoded_bp2)
            if bp2_mkv_files:
                bp2_records = self._extract_mkv_group(bp2_mkv_files)
                self._wire_sidecars_from_records(bp2_records, plane='bp2')
                bp2_videos = [r["video"] for r in bp2_records]

        return bp1_videos, bp2_videos

    def _validate_mkv_sidecars(self):
        """Verify that MKV-extracted sidecars cover what reconstruction will need.

        Raises ``ValueError`` with a clear plane/path message rather than
        passing ``None`` or partial lists down to ``load_sparse`` /
        ``_apply_residuals_to_encoded`` / streaming reconstruction.
        """
        n1 = len(self.path_encoded_bp1)
        n2 = len(self.path_encoded_bp2) if self.path_encoded_bp2 else 0

        def _check_roi(plane, n_videos, sparse_paths, video_paths):
            if not sparse_paths:
                raise ValueError(
                    f"MKV reconstruction with use_roi=True needs the {plane} sparse "
                    f"ROI sidecar(s) (.npz), but none were extracted. Set use_roi=False "
                    f"to reconstruct background-only, or re-package the MKV with the "
                    f"sidecar attached."
                )
            resolved = self._resolve_paths(sparse_paths)
            if len(resolved) < n_videos:
                missing_for = video_paths[len(resolved):]
                raise ValueError(
                    f"{plane} has {n_videos} video(s) but only {len(resolved)} ROI "
                    f"sidecar(s) were extracted; missing for: {missing_for}"
                )
            missing = [p for p in resolved if not os.path.exists(p)]
            if missing:
                raise ValueError(f"Missing {plane} ROI sidecar(s): {missing}")

        if self.use_roi:
            _check_roi("bp1", n1, self.path_sparse_bp1, self.path_encoded_bp1)
            if self.path_encoded_bp2:
                _check_roi("bp2", n2, self.path_sparse_bp2, self.path_encoded_bp2)

        def _check_residuals(plane, n_videos, residual_list, video_paths):
            if residual_list is None:
                return  # nothing advertised, nothing to validate
            if isinstance(residual_list, str):
                # Glob-shaped explicit override; presence checked when applied.
                return
            if not isinstance(residual_list, (list, tuple)):
                return
            if not any(residual_list):
                return  # no residuals advertised
            if len(residual_list) < n_videos:
                raise ValueError(
                    f"{plane} advertises residuals but only {len(residual_list)} "
                    f"residual sidecar(s) extracted for {n_videos} video(s)"
                )
            missing = [
                video_paths[i] for i, r in enumerate(residual_list) if not r
            ]
            if missing:
                raise ValueError(
                    f"{plane} MKV archive(s) advertise residuals but the matching "
                    f"sidecar is missing for video(s): {missing}"
                )
            absent_on_disk = [r for r in residual_list if r and not os.path.exists(r)]
            if absent_on_disk:
                raise ValueError(f"Missing {plane} residual sidecar(s): {absent_on_disk}")

        _check_residuals("bp1", n1, self.path_residual_bp1, self.path_encoded_bp1)
        if self.path_encoded_bp2:
            _check_residuals("bp2", n2, self.path_residual_bp2, self.path_encoded_bp2)

    def cleanup_temp_directories(self):
        """
        Clean up temporary directories created during MKV extraction.
        """
        if hasattr(self, 'temp_dirs_to_cleanup') and self.temp_dirs_to_cleanup:
            import shutil
            for temp_dir in self.temp_dirs_to_cleanup:
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        print(f'Cleaned up temporary directory: {temp_dir}')
                except Exception as e:
                    print(f'Warning: Failed to cleanup temporary directory {temp_dir}: {e}')
            self.temp_dirs_to_cleanup = []

    def load_metadata_from_npz(self, npz_file_path):
        """
        Load metadata from NPZ file that was saved with sparse matrix.
        Compatible with both original (45e86e9) and current formats.
        """
        try:
            # Load the NPZ file
            npz_data = np.load(npz_file_path, allow_pickle=True)
            
            # Check for metadata in different formats
            if 'metadata' in npz_data:
                # Original format from 45e86e9 - use 'metadata' key
                metadata_json_str = npz_data['metadata'].item()
                metadata = json.loads(metadata_json_str)
                return metadata
            elif 'metadata_json' in npz_data:
                # Current format - use 'metadata_json' key
                metadata_json_str = npz_data['metadata_json'].item()
                metadata = json.loads(metadata_json_str)
                return metadata
            else:
                return {}
                
        except Exception as e:
            print(f'Error loading metadata from {npz_file_path}: {e}')
            return {}

    def load_original_metadata(self):
        """
        Load original TIFF metadata from JSON files or embedded in NPZ files.
        """
        print('Loading metadata...')
        
        # --- Try loading from JSON files first ---
        metadata_file_bp1 = os.path.join(self.output_path, f'{self.stem}_metadata_bp1.json')
        metadata_file_bp2 = os.path.join(self.output_path, f'{self.stem}_metadata_bp2.json')

        # If exact stem match not found, try auto-detection
        if not os.path.exists(metadata_file_bp1):
            print(f'Exact metadata file not found: {metadata_file_bp1}')
            print('Attempting to auto-detect metadata files...')
            
            # Look for any *_metadata_bp1.json files in the output directory
            pattern_bp1 = os.path.join(self.output_path, '*_metadata_bp1.json')
            json_files_bp1 = glob.glob(pattern_bp1)
            
            if json_files_bp1:
                if len(json_files_bp1) > 1:
                    print(f'Warning: Multiple BP1 metadata files found: {json_files_bp1}')
                    print(f'Using first match: {json_files_bp1[0]}')
                metadata_file_bp1 = json_files_bp1[0]  # Use first match
                print(f'Auto-detected BP1 metadata file: {metadata_file_bp1}')
                
                # Try to find corresponding BP2 file with same prefix
                detected_stem = os.path.basename(metadata_file_bp1).replace('_metadata_bp1.json', '')
                metadata_file_bp2 = os.path.join(self.output_path, f'{detected_stem}_metadata_bp2.json')
                
                if os.path.exists(metadata_file_bp2):
                    print(f'Auto-detected BP2 metadata file: {metadata_file_bp2}')
                else:
                    # Also try pattern matching for BP2
                    pattern_bp2 = os.path.join(self.output_path, '*_metadata_bp2.json')
                    json_files_bp2 = glob.glob(pattern_bp2)
                    if json_files_bp2:
                        metadata_file_bp2 = json_files_bp2[0]
                        print(f'Auto-detected BP2 metadata file: {metadata_file_bp2}')

        metadata_bp1 = None
        metadata_bp2 = None

        if os.path.exists(metadata_file_bp1):
            try:
                with open(metadata_file_bp1, 'r') as f:
                    metadata_bp1 = json.load(f)
                print(f'Loaded metadata for BP1 from: {metadata_file_bp1}')
                if os.path.exists(metadata_file_bp2):
                    with open(metadata_file_bp2, 'r') as f:
                        metadata_bp2 = json.load(f)
                    print(f'Loaded metadata for BP2 from: {metadata_file_bp2}')
                return metadata_bp1, metadata_bp2
            except Exception as e:
                print(f'Error loading metadata from JSON file: {e}. Falling back to other methods.')

        # --- If JSON not found, try loading from NPZ files ---
        print('No JSON metadata found. Trying to load from NPZ files...')
        metadata_bp1 = []
        try:
            if hasattr(self, 'path_sparse_bp1') and self.path_sparse_bp1:
                bp1_pattern = self.path_sparse_bp1
            else:
                bp1_pattern = os.path.join(self.output_path, '*.npz')
            bp1_files = sorted(glob.glob(bp1_pattern))
            
            for npz_file in bp1_files:
                if '_bp2_' in npz_file or npz_file.endswith('_bp2.npz'):
                    continue
                metadata_entry = self.load_metadata_from_npz(npz_file)
                if metadata_entry:
                    metadata_bp1.append(metadata_entry)
            
            if metadata_bp1:
                print(f'Loaded BP1 metadata from {len(metadata_bp1)} NPZ files')
            
            # BP2 logic
            if self.path_encoded_bp2:
                metadata_bp2 = []
                bp2_files = [f for f in bp1_files if '_bp2_' in f or f.endswith('_bp2.npz')]
                for npz_file in sorted(bp2_files):
                    metadata_entry = self.load_metadata_from_npz(npz_file)
                    if metadata_entry:
                        metadata_bp2.append(metadata_entry)
                if metadata_bp2:
                    print(f'Loaded BP2 metadata from {len(metadata_bp2)} NPZ files')

        except Exception as e:
            print(f'Error loading metadata from NPZ: {e}. Falling back to encoded metadata.')

        # --- Fallback to basic encoded metadata ---
        if not metadata_bp1:
            print('No metadata found in JSON or NPZ files. Using basic encoded file metadata.')
            return self.extract_encoded_metadata()

        return metadata_bp1, metadata_bp2

    def extract_encoded_metadata(self):
        """Extract metadata from encoded video files if possible, or create basic metadata"""
        print('Extracting metadata from encoded files...')
        metadata_bp1 = []
        
        # For encoded files, we can't extract original TIFF metadata
        # But we can create basic metadata structure
        for file_path in self.path_encoded_bp1:
            metadata = {
                'source_file': os.path.basename(file_path),
                'is_encoded': True,
                'file_type': 'encoded_video'
            }
            metadata_bp1.append(metadata)
        
        metadata_bp2 = []
        if self.path_encoded_bp2:
            for file_path in self.path_encoded_bp2:
                metadata = {
                    'source_file': os.path.basename(file_path),
                    'is_encoded': True,
                    'file_type': 'encoded_video'
                }
                metadata_bp2.append(metadata)
        
        return metadata_bp1, metadata_bp2 if self.path_encoded_bp2 else None

    def format_metadata_for_tifffile(self, metadata, frame_index=None):
        """Format extracted metadata for tifffile.TiffWriter - UNSPARZ version"""
        if not metadata or 'error' in metadata:
            return None, []
        
        description = None
        extratags = []
        
        # For the first frame, use global metadata
        if frame_index is None or frame_index == 0:
            # First priority: OME-XML from original metadata (regardless of is_encoded flag)
            if metadata.get('is_ome') and 'ome_xml' in metadata:
                description = metadata['ome_xml']
                extratags.append((305, 's', 0, "UNSPARZ with OME-XML", True))
                return description, extratags
            
            # Second priority: For encoded files without OME-XML, add basic info
            if metadata.get('is_encoded') and not metadata.get('is_ome'):
                description = f"Reconstructed from {metadata['source_file']} | Processed by UNSPARZ"
                extratags.append((305, 's', 0, "UNSPARZ", True))
            else:
                # Same logic as SPARZIP for original TIFF files
                description_parts = []
                
                # Add ImageJ metadata if present
                if metadata.get('is_imagej') and 'imagej_metadata' in metadata:
                    try:
                        imagej_meta = metadata['imagej_metadata']
                        if isinstance(imagej_meta, dict):
                            for key, value in imagej_meta.items():
                                if isinstance(value, (str, int, float)):
                                    description_parts.append(f"{key}={value}")
                    except:
                        pass
                
                # Add TIFF tags to description and extratags
                if 'tags' in metadata:
                    tags = metadata['tags']
                    
                    # ImageDescription is the most important
                    if 'ImageDescription' in tags:
                        existing_desc = tags['ImageDescription']
                        if existing_desc and isinstance(existing_desc, str):
                            description = existing_desc
                            if description_parts:
                                description += f" | {' | '.join(description_parts)}"
                    
                    # Add other important tags as extratags
                    for tag_name, tag_value in tags.items():
                        if tag_name == 'Software' and isinstance(tag_value, str):
                            extratags.append((305, 's', 0, f"{tag_value} -> UNSPARZ", True))  # Software tag
                        elif tag_name == 'DateTime' and isinstance(tag_value, str):
                            extratags.append((306, 's', 0, tag_value, True))  # DateTime tag
                        elif tag_name not in ['ImageDescription', 'Software', 'DateTime'] and isinstance(tag_value, (str, int, float)):
                            description_parts.append(f"{tag_name}={tag_value}")
                
                # Create description from parts if not already set
                if description is None and description_parts:
                    description = ' | '.join(description_parts)
                
                # Add default software tag if not present
                if not any(tag[0] == 305 for tag in extratags):
                    extratags.append((305, 's', 0, "UNSPARZ", True))
        
        else:
            # For subsequent frames, use individual IFD metadata if available
            if 'individual_ifds' in metadata and frame_index < len(metadata['individual_ifds']):
                ifd_meta = metadata['individual_ifds'][frame_index]
                ifd_tags = ifd_meta.get('tags', {})
                
                # Add IFD-specific ImageDescription
                if 'ImageDescription' in ifd_tags:
                    description = ifd_tags['ImageDescription']
                
                # Add IFD-specific custom tags
                for tag_name, tag_value in ifd_tags.items():
                    if tag_name.startswith('tag_') and isinstance(tag_value, (str, int, float)):
                        try:
                            tag_code = int(tag_name.split('_')[1])
                            if tag_code > 50000:  # Custom tags usually > 50000
                                if isinstance(tag_value, str):
                                    extratags.append((tag_code, 's', 0, str(tag_value), True))
                                elif isinstance(tag_value, int):
                                    extratags.append((tag_code, 'i', 1, tag_value, True))
                                elif isinstance(tag_value, float):
                                    # Convert float to string for Picasso compatibility
                                    extratags.append((tag_code, 's', 0, str(tag_value), True))
                        except (ValueError, IndexError):
                            continue
        
        return description, extratags

    def extract_resolution_from_metadata(self, metadata):
        """Extract resolution information from metadata for tifffile.imwrite"""
        resolution = None
        resolution_unit = None
        
        if metadata and 'tags' in metadata:
            tags = metadata['tags']
            
            # Get X and Y resolution
            x_res = tags.get('XResolution')
            y_res = tags.get('YResolution')
            res_unit = tags.get('ResolutionUnit', 1)  # Default to no unit
            
            if x_res and y_res:
                # Handle both tuple/list format [numerator, denominator] and direct values
                if isinstance(x_res, (list, tuple)) and len(x_res) >= 2:
                    x_resolution = x_res[0] / x_res[1] if x_res[1] != 0 else x_res[0]
                else:
                    x_resolution = float(x_res)
                
                if isinstance(y_res, (list, tuple)) and len(y_res) >= 2:
                    y_resolution = y_res[0] / y_res[1] if y_res[1] != 0 else y_res[0]
                else:
                    y_resolution = float(y_res)
                
                resolution = (x_resolution, y_resolution)
                # tifffile expects numeric values for resolutionunit, not strings
                resolution_unit = {1: 1, 2: 2, 3: 3}.get(res_unit, 1)
        
        return resolution, resolution_unit

    def write_tiff_with_individual_ifds(self, filename, all_frames, file_metadata):
        """
        Write TIFF file with individual IFD metadata for each frame using TiffWriter.
        Used when requires_individual_ifd_writing is True.
        """
        print(f"Writing TIFF with metadata: {filename}")
        
        # Extract global resolution information
        resolution, resolution_unit = self.extract_resolution_from_metadata(file_metadata)
        
        # Get individual IFD metadata
        individual_ifds = file_metadata.get('individual_ifds', [])
        
        # Determine if BigTIFF is needed
        estimated_size = all_frames.nbytes if hasattr(all_frames, 'nbytes') else 0
        use_bigtiff = estimated_size > 4 * 1024 * 1024 * 1024  # 4GB threshold
        
        if use_bigtiff:
            print(f"Writing BigTIFF with individual IFDs (file size > 4GB): {filename}")
        
        with tifffile.TiffWriter(filename, bigtiff=use_bigtiff) as tif:
            for frame_idx, frame_data in enumerate(all_frames):
                # Prepare extratags for frame-specific metadata
                frame_extratags = []
                
                if frame_idx < len(individual_ifds):
                    ifd_data = individual_ifds[frame_idx]
                    frame_tags = ifd_data.get('tags', {})
                    
                    # Convert frame-specific tags to extratags format
                    for tag_name, tag_value in frame_tags.items():
                        if tag_name.startswith('tag_') and tag_name[4:].isdigit():
                            # Numeric tag
                            tag_code = int(tag_name[4:])
                            
                            # Skip basic TIFF tags that are handled automatically
                            # Including 270 (ImageDescription) which must use description parameter
                            if tag_code in [256, 257, 258, 259, 262, 270, 273, 277, 278, 279, 282, 283, 296]:
                                continue
                            
                            # Determine the tag type and format extratag
                            # Note: Use only Picasso-compatible types (no floats)
                            if isinstance(tag_value, int):
                                # Use 'I' for 32-bit unsigned int (TIFF type 4)
                                frame_extratags.append((tag_code, 'I', 1, tag_value, True))
                            elif isinstance(tag_value, float):
                                # Convert float to string for Picasso compatibility
                                # (Picasso doesn't support TIFF type 11/12 for floats)
                                frame_extratags.append((tag_code, 's', 0, str(tag_value), True))
                            elif isinstance(tag_value, str):
                                frame_extratags.append((tag_code, 's', 0, tag_value, True))
                            elif isinstance(tag_value, (list, tuple)):
                                # Handle arrays/lists
                                if len(tag_value) > 0:
                                    if isinstance(tag_value[0], int):
                                        frame_extratags.append((tag_code, 'I', len(tag_value), tag_value, True))
                                    elif isinstance(tag_value[0], float):
                                        # Convert float array to string for Picasso compatibility
                                        str_value = ','.join(str(v) for v in tag_value)
                                        frame_extratags.append((tag_code, 's', 0, str_value, True))
                
                # Write frame with its specific metadata
                if frame_idx == 0:
                    # First frame gets global metadata too
                    description = None
                    if file_metadata.get('is_ome') and 'ome_xml' in file_metadata:
                        description = file_metadata['ome_xml']
                    elif 'tags' in file_metadata and 'ImageDescription' in file_metadata['tags']:
                        description = file_metadata['tags']['ImageDescription']
                    
                    tif.write(frame_data, 
                             photometric='minisblack',
                             description=description,
                             resolution=resolution if resolution else None,
                             resolutionunit=resolution_unit if resolution_unit else None,
                             extratags=frame_extratags if frame_extratags else None)
                else:
                    # Subsequent frames get individual metadata
                    tif.write(frame_data,
                             photometric='minisblack', 
                             extratags=frame_extratags if frame_extratags else None)
                
                if frame_idx % 500 == 0:  # Progress indicator
                    print(f"Written frame {frame_idx}/{len(all_frames)}")
        
        print(f"Completed TIFF writing: {filename}")

    def fix_ome_xml_for_output(self, ome_xml, filename, num_frames):
        """
        Update OME-XML metadata for the output file.
        Updates filename references and UUID to match the new file.
        """
        import re
        import uuid
        import os
        
        if not ome_xml:
            return None
        
        # Generate new UUID for this file
        new_uuid = str(uuid.uuid4())
        
        # Get the base filename without path
        base_filename = os.path.basename(filename)
        
        # Update the Image Name attribute
        ome_xml = re.sub(
            r'Name="[^"]*"',
            f'Name="{base_filename}"',
            ome_xml,
            count=1  # Only replace first occurrence (the Image Name)
        )
        
        # Update all FileName attributes in UUID tags
        ome_xml = re.sub(
            r'FileName="[^"]*"',
            f'FileName="{base_filename}"',
            ome_xml
        )
        
        # Update all UUID values to the new UUID
        ome_xml = re.sub(
            r'urn:uuid:[a-f0-9\-]+',
            f'urn:uuid:{new_uuid}',
            ome_xml
        )
        
        # Ensure we have the right number of TiffData elements
        tiff_data_count = ome_xml.count('<TiffData')
        if tiff_data_count != num_frames:
            print(f"Warning: OME-XML has {tiff_data_count} TiffData elements but writing {num_frames} frames")
            # If we have more TiffData than frames, that's OK (extra will be ignored)
            # If we have fewer, we might need to generate more, but for now we'll use what we have
        
        return ome_xml

    def write_tiff_file(self, filename, all_frames, file_metadata, debug_prefix=""):
        """
        Helper method to write TIFF files with proper metadata handling.
        Automatically chooses between bulk writing and frame-by-frame writing.
        """
        requires_individual_writing = file_metadata.get('requires_individual_ifd_writing', False)
        # print(f"{debug_prefix}Requires individual IFD writing: {requires_individual_writing}")
        
        if requires_individual_writing:
            # Use frame-by-frame writing for individual IFD metadata
            self.write_tiff_with_individual_ifds(filename, all_frames, file_metadata)
        else:
            # Use standard bulk writing
            resolution, resolution_unit = self.extract_resolution_from_metadata(file_metadata)
            
            # ALWAYS format metadata to get extratags
            description, extratags = self.format_metadata_for_tifffile(file_metadata, frame_index=0)
            
            # If OME-XML exists, fix it for the output file
            if file_metadata.get('is_ome') and 'ome_xml' in file_metadata:
                original_ome = file_metadata['ome_xml']
                # Fix the OME-XML for this specific output file
                fixed_ome = self.fix_ome_xml_for_output(original_ome, filename, len(all_frames))
                if fixed_ome:
                    description = fixed_ome
                else:
                    description = original_ome
            
            # Determine if BigTIFF is needed (>4GB uncompressed)
            estimated_size = all_frames.nbytes if hasattr(all_frames, 'nbytes') else 0
            use_bigtiff = estimated_size > 4 * 1024 * 1024 * 1024  # 4GB threshold
            
            if use_bigtiff:
                print(f"Writing BigTIFF (file size > 4GB): {filename}")
            
            # Now, write with description and extratags
            if resolution:
                tifffile.imwrite(filename, all_frames, photometric='minisblack', 
                               bigtiff=use_bigtiff, description=description, extratags=extratags,
                               resolution=resolution, resolutionunit=resolution_unit)
            else:
                tifffile.imwrite(filename, all_frames, photometric='minisblack', 
                               bigtiff=use_bigtiff, description=description, extratags=extratags)
                # print(f"{debug_prefix}Fallback written to: {filename}")

    # def load_sparse(self, path_sparse_bp1:str, path_sparse_bp2:str):
    #     print ('Loading sparse matrices...')
    #     files_bp1 = sorted(glob.glob(path_sparse_bp1))
    #     if path_sparse_bp2 is not None:
    #         files_bp2 = sorted(glob.glob(path_sparse_bp2))
    #         assert len(files_bp1) == len(files_bp2), 'Error: Both biplanes must have the same number of images.'
    #         bp1,bp2 = [],[]
    #         for i in range(len(files_bp1)):
    #             try:
    #                 bp1.append(joblib.load(files_bp1[i]))
    #                 bp2.append(joblib.load(files_bp2[i]))
    #             except Exception as e:
    #                 print(f'Error loading sparse matrices: {e}')
    #                 print(f'File: {files_bp1[i]}')
    #         return bp1, bp2
    #     bp1 =[]
    #     for i in range(len(files_bp1)):
    #         bp1.append(joblib.load(files_bp1[i]))
    #     return bp1, None

    def load_sparse_matrix_from_npz(self, npz_file_path):
        """
        Load sparse matrix from NPZ file, handling multiple formats.
        Compatible with zstd+delta (new), original (45e86e9), current, and legacy formats.
        """
        import io

        try:
            # Read file bytes to check format
            with open(npz_file_path, 'rb') as f:
                file_bytes = f.read()

            # Check if zstd compressed (magic bytes: 0x28 0xB5 0x2F 0xFD)
            if file_bytes[:4] == b'\x28\xb5\x2f\xfd':
                # Decompress with zstd
                dctx = zstd.ZstdDecompressor()
                decompressed = dctx.decompress(file_bytes)
                npz_data = np.load(io.BytesIO(decompressed), allow_pickle=True)
            else:
                # Standard NPZ file
                npz_data = np.load(npz_file_path, allow_pickle=True)

            # Check for new format with delta encoding
            if 'encoding' in npz_data and str(npz_data['encoding']) == 'delta':
                data = npz_data['data']
                delta_coords = npz_data['delta_coords']
                shape = tuple(npz_data['shape'])

                # Decode delta-encoded coordinates
                coords = self._delta_decode_coords(delta_coords)

                sparse_matrix = sparse.COO(coords=coords, data=data, shape=shape)
                return sparse_matrix

            # Check for original format from 45e86e9 (data, coords, shape)
            elif 'data' in npz_data and 'coords' in npz_data and 'shape' in npz_data:
                data = npz_data['data']
                coords = npz_data['coords']
                shape = tuple(npz_data['shape'])

                sparse_matrix = sparse.COO(coords=coords, data=data, shape=shape)
                return sparse_matrix

            # Check for current format with separate row/col arrays
            elif 'data' in npz_data and 'row' in npz_data and 'col' in npz_data and 'shape' in npz_data:
                data = npz_data['data']
                row = npz_data['row']
                col = npz_data['col']
                shape = tuple(npz_data['shape'])

                sparse_matrix = sparse.COO(coords=[row, col], data=data, shape=shape)
                return sparse_matrix

            else:
                # Legacy format: fallback to sparse.load_npz
                return sparse.load_npz(npz_file_path)

        except Exception as e:
            print(f'Error loading sparse matrix from {npz_file_path}: {e}')
            # Final fallback
            return sparse.load_npz(npz_file_path)

    def _delta_decode_coords(self, delta_coords):
        """Decode delta-encoded coordinates."""
        return np.cumsum(delta_coords, axis=1)

    def load_residual_sidecar(self, path):
        """Load and decode a residual sidecar file. Returns (residual_array, header_dict)."""
        return load_residual_sidecar(path)

    def apply_residuals(self, decoded_frames, residual, original_dtype):
        return apply_residuals(decoded_frames, residual, original_dtype)

    def _resolve_residual_paths(self, attr_name, expected_videos):
        """Normalize self.<attr_name> into a list aligned with expected_videos length, or None."""
        val = getattr(self, attr_name)
        if val is None:
            return None
        if isinstance(val, str):
            matched = sorted(glob.glob(val))
            return matched if matched else None
        if isinstance(val, (list, tuple)):
            return list(val)
        return None

    def _autodetect_residual_paths(self):
        """Populate self.path_residual_bp1/bp2 from sibling files when not explicitly set."""
        # bp1
        if self.path_residual_bp1 is None:
            candidates = []
            for vp in self.path_encoded_bp1:
                rp = get_residual_path_for_video(vp)
                candidates.append(rp if os.path.exists(rp) else None)
            if any(candidates):
                self.path_residual_bp1 = candidates
        else:
            self.path_residual_bp1 = self._resolve_residual_paths('path_residual_bp1', self.path_encoded_bp1)

        # bp2
        if self.path_encoded_bp2 is not None and self.path_residual_bp2 is None:
            candidates = []
            for vp in self.path_encoded_bp2:
                rp = get_residual_path_for_video(vp)
                candidates.append(rp if os.path.exists(rp) else None)
            if any(candidates):
                self.path_residual_bp2 = candidates
        elif self.path_encoded_bp2 is not None:
            self.path_residual_bp2 = self._resolve_residual_paths('path_residual_bp2', self.path_encoded_bp2)

    def _build_chunked_residual_dask(self, residual_path, encoded_dask):
        """Build a Dask array of residual frames lazily from a v1/v2 sidecar.

        For indexed v2 files, each delayed task seeks directly to its chunk's
        compressed payload and decompresses only that one chunk — O(1) I/O per
        chunk, not O(N) per chunk. Legacy v2 (no index) and v1 fall back to a
        single sequential read shared across delayed tasks.
        """
        header = read_residual_header(residual_path)
        T_h, H_h, W_h = header["shape"]
        if encoded_dask.shape != (T_h, H_h, W_h):
            raise ValueError(
                f"residual shape {(T_h, H_h, W_h)} does not match encoded video "
                f"shape {tuple(encoded_dask.shape)} for {residual_path}; "
                f"refusing to silently truncate"
            )
        res_dtype = _residual_dtype_from_code(header["dtype_code"])

        if header["version"] == RESIDUAL_VERSION_V2 and (header["flags"] & RESIDUAL_FLAG_INDEXED):
            index = header["index"]
            delayed_chunks = []
            for entry in index:
                start, n_frames, _off, _ln = entry
                d = delayed(read_residual_chunk_at)(residual_path, entry, header)
                delayed_chunks.append(
                    da.from_delayed(d, shape=(n_frames, H_h, W_h), dtype=res_dtype)
                )
            if not delayed_chunks:
                return da.zeros((0, H_h, W_h), dtype=res_dtype)
            return da.concatenate(delayed_chunks, axis=0)

        # Legacy / unindexed: read the whole sidecar once (each chunk task would
        # otherwise have to scan from offset 0). One full read amortises across
        # all chunks instead of one full read per chunk.
        full, _ = load_residual_sidecar(residual_path)
        return da.from_array(full, chunks=(max(1, header.get("chunk_size") or T_h), H_h, W_h))

    def _apply_residuals_to_encoded(self):
        """Replace decoded dask arrays with residual-corrected lazy versions.

        For each encoded video that has a residual sidecar, build a Dask
        residual from per-chunk delayed reads and produce a lazy
        ``clip(encoded + residual)`` Dask array of the original dtype.
        Raises on shape mismatch to preserve exact reconstruction.
        """
        def _apply_to_list(encoded_list, residual_paths, label):
            if not residual_paths:
                return
            for k in range(len(encoded_list)):
                if k >= len(residual_paths) or not residual_paths[k]:
                    continue
                rp = residual_paths[k]
                try:
                    header = read_residual_header(rp)
                except Exception as e:
                    print(f'  Warning: could not read residual header {rp}: {e}')
                    continue
                arr = encoded_list[k]
                if tuple(arr.shape) != tuple(header["shape"]):
                    raise ValueError(
                        f"residual {rp} shape {tuple(header['shape'])} "
                        f"does not match {label}[{k}] encoded shape {tuple(arr.shape)}; "
                        f"exact reconstruction requires equal shapes"
                    )
                residual_da = self._build_chunked_residual_dask(rp, arr)
                target = np.dtype(header["original_dtype"])
                summed = arr.astype(np.int64) + residual_da.astype(np.int64)
                if np.issubdtype(target, np.integer):
                    info = np.iinfo(target)
                    summed = da.clip(summed, info.min, info.max)
                corrected = summed.astype(target)
                encoded_list[k] = corrected
                print(f'  Applied residual ({header["mode"]}, v{header["version"]}) from {os.path.basename(rp)}')

        if self.path_residual_bp1:
            _apply_to_list(self.encoded_bp1, self.path_residual_bp1, "bp1")
        if self.path_residual_bp2 and self.encoded_bp2 is not None:
            _apply_to_list(self.encoded_bp2, self.path_residual_bp2, "bp2")

    def load_sparse(self, sparse_bp1, sparse_bp2, shapes:tuple):
        """Load sparse ROI matrices. Accepts a glob string OR a list of NPZ paths."""
        print ('Loading sparse matrices...')
        files_bp1 = self._resolve_paths(sparse_bp1)
        if sparse_bp2 is not None:
            files_bp2 = self._resolve_paths(sparse_bp2)
            assert len(files_bp1) == len(files_bp2), 'Error: Both biplanes must have the same number of images.'
            bp1,bp2 = [],[]
            for i in range(len(files_bp1)):
                sparse_matrix_bp1 = self.load_sparse_matrix_from_npz(files_bp1[i])
                sparse_matrix_bp2 = self.load_sparse_matrix_from_npz(files_bp2[i])
                bp1.append(da.from_array(sparse_matrix_bp1, chunks=(1,shapes[i][1],shapes[i][2])))
                bp2.append(da.from_array(sparse_matrix_bp2, chunks=(1,shapes[i][1],shapes[i][2])))
            return bp1, bp2
        bp1 =[]
        for i in range(len(files_bp1)):
            sparse_matrix_bp1 = self.load_sparse_matrix_from_npz(files_bp1[i])
            bp1.append(da.from_array(sparse_matrix_bp1, chunks=(1,shapes[i][1],shapes[i][2])))
        return bp1, None


    def load_mp4(self, file_path):
        # Open the video file
        container = av.open(file_path)
        video_stream = container.streams.video[0]
        frame_count = video_stream.frames

        # Read all frames into a list of numpy arrays
        @delayed
        def read_frames():
            frames = []
            dtype = None
            for frame in container.decode(video_stream):
                np_frame = frame.to_ndarray(format='gray16le')
                if dtype is None:
                    dtype = np_frame.dtype  # Set dtype on first frame
                frames.append(np_frame)
            return np.array(frames), dtype
    
        @delayed
        def read_frames_h264():
            frames = []
            for frame in container.decode(video_stream):
                # Assuming conversion directly to 'gray16le' is handled elsewhere or not necessary
                y_plane = frame.planes[0]
                y_data = np.frombuffer(y_plane, np.uint16)
                y_data = y_data.reshape((frame.height, frame.width))
                y_data_16bit = np.left_shift(y_data, 6)
                frames.append(y_data_16bit)
            return np.array(frames), y_data_16bit.dtype
            # return np.stack(frames, axis=0) 


        # Get delayed frames and dtype
        if video_stream.codec.name == 'h264':
            print('H264 codec detected.')
            frames_dtype = read_frames()
        else:
            print(f'{video_stream.codec.name} codec detected.')
            frames_dtype = read_frames()

        # Calculate the shape and dtype of the frames for creating a Dask array
        # Since the dtype and frames are in a single tuple, we need to compute them to extract properly
        frames, dtype = frames_dtype.compute()

        # Create a Dask array from the numpy array of frames
        dask_frames = da.from_array(frames, chunks=(1, *frames[0].shape))
        return dask_frames

    def decode(self, path_bp1, path_bp2=None):
        """Decode encoded videos. Accepts a glob string or an already-resolved list of paths."""
        print('Decoding images...')
        files_bp1 = self._resolve_paths(path_bp1)
        bp1 = [self.load_mp4(file) for file in files_bp1]

        bp2 = None
        if path_bp2:
            files_bp2 = self._resolve_paths(path_bp2)
            assert len(files_bp1) == len(files_bp2), 'Error: Both biplanes must have the same number of images.'
            bp2 = [self.load_mp4(file) for file in files_bp2]

        return bp1, bp2

    @staticmethod
    def _resolve_paths(maybe_paths):
        """Accept either a glob string or a list of file paths and return a sorted list."""
        if maybe_paths is None:
            return []
        if isinstance(maybe_paths, (list, tuple)):
            return sorted(maybe_paths)
        return sorted(glob.glob(maybe_paths))


    # def load_mp4(self, file_path):
    #     dtype = None
    #     frame_count = 0
    #     first_frame = None

    #     # Read the first frame to infer dtype and get frame count
    #     with av.open(file_path) as container:
    #         frame_count = container.streams.video[0].frames
    #         for packet in container.demux():
    #             for frame in packet.decode():
    #                 first_frame = frame.to_ndarray(format='gray16le')
    #                 dtype = first_frame.dtype
    #                 break
    #             if dtype is not None:
    #                 break

    #     def frame_generator(file_path):
    #         with av.open(file_path) as container:
    #             video_stream = container.streams.video[0]
    #             for frame_index, frame in enumerate(container.decode(video_stream)):
    #                 yield frame_index, frame

    #     def read_frame(i):
    #         for index,frame in frame_generator(file_path):
    #             if index == i:
    #                 return frame.to_ndarray(format='gray16le')

    #     frames = [delayed(read_frame)(i) for i in range(frame_count)]

    #     # Create Dask arrays for each frame
    #     frame_arrays = [da.from_delayed(frame, shape=first_frame.shape, dtype=dtype) for frame in frames]

    #     # Concatenate frame arrays into a single Dask array
    #     video_array = da.stack(frame_arrays, axis=0)

    #     return video_array

    
    # def decode(self, path_bp1: str, path_bp2: str):
    #     print('Decoding images...')
    #     # files_bp1 = sorted(glob.glob(path_bp1))
    #     files_bp1 = self.path_encoded_bp1
    #     if self.path_encoded_bp2 is None:
    #         with concurrent.futures.ThreadPoolExecutor() as executor:
    #             bp1 = list(executor.map(lambda file: self.load_mp4(file), files_bp1))
    #         return bp1, None
    #     # files_bp2 = sorted(glob.glob(path_bp2))
    #     files_bp2 = self.path_encoded_bp2
    #     assert len(files_bp1) == len(files_bp2), 'Error: Both biplanes must have the same number of images.'
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         bp1 = list(executor.map(lambda file: self.load_mp4(file), files_bp1))
    #         bp2 = list(executor.map(lambda file: self.load_mp4(file), files_bp2))
    #     return bp1, bp2

    def _unshuffle_bytes(self, data: bytes, itemsize: int = 2) -> bytes:
        """Reverse the byte shuffle operation."""
        arr = np.frombuffer(data, dtype=np.uint8)
        n_elements = len(arr) // itemsize
        if n_elements == 0:
            return data
        reshaped = arr[:n_elements * itemsize].reshape(itemsize, n_elements)
        unshuffled = reshaped.T.flatten()
        return unshuffled.tobytes()

    def _decode_zst_file(self, file_path):
        """
        Decode a single zst file, handling new format (flags, delta, shuffle) and old format.
        """
        with open(file_path, 'rb') as f:
            # Read flags byte
            flags = np.frombuffer(f.read(1), dtype=np.uint8)[0]

            # Determine format and parse flags
            # Old format: flags was just 0 or 1 (zdict_present)
            # New format: flags is a bitfield (0-7 are valid)
            if flags <= 7:
                # Could be new format
                zdict_present = bool(flags & 0x01)
                delta_encoded = bool(flags & 0x02)
                shuffled = bool(flags & 0x04)
            else:
                # Old format - flags byte was actually zdict_present (0 or 1)
                # This shouldn't happen with old files since they only had 0 or 1
                # But handle gracefully
                f.seek(0)
                flags = np.frombuffer(f.read(1), dtype=np.uint8)[0]
                zdict_present = bool(flags)
                delta_encoded = False
                shuffled = False

            # Read dictionary if present
            if zdict_present:
                zdict_length = np.frombuffer(f.read(4), dtype=np.int32)[0]
                zdict = f.read(zdict_length)
                zstd_dict = zstd.ZstdCompressionDict(zdict)
                dctx = zstd.ZstdDecompressor(dict_data=zstd_dict)
            else:
                dctx = zstd.ZstdDecompressor()

            # Read shape and dtype
            data_shape = tuple(np.frombuffer(f.read(12), dtype=np.int32))
            dtype_str = np.frombuffer(f.read(20), dtype='S20').tobytes().decode('utf-8').rstrip('\x00')
            data_dtype = np.dtype(dtype_str)

            # Decompress data
            with dctx.stream_reader(f) as reader:
                decompressed_data = reader.read()

            # Calculate frame parameters
            n_frames = data_shape[0]
            frame_shape = data_shape[1:]
            frame_size = int(np.prod(frame_shape))

            if delta_encoded:
                # Delta decoding required
                # First frame is original dtype, subsequent frames are int16 deltas
                frames = np.zeros(data_shape, dtype=data_dtype)

                # Determine bytes per frame based on encoding
                first_frame_bytes = frame_size * data_dtype.itemsize
                delta_frame_bytes = frame_size * 2  # int16

                offset = 0
                for i in range(n_frames):
                    if i == 0:
                        # First frame: original dtype
                        frame_data = decompressed_data[offset:offset + first_frame_bytes]
                        if shuffled:
                            frame_data = self._unshuffle_bytes(frame_data, itemsize=data_dtype.itemsize)
                        frames[i] = np.frombuffer(frame_data, dtype=data_dtype).reshape(frame_shape)
                        offset += first_frame_bytes
                    else:
                        # Subsequent frames: int16 deltas
                        frame_data = decompressed_data[offset:offset + delta_frame_bytes]
                        if shuffled:
                            frame_data = self._unshuffle_bytes(frame_data, itemsize=2)
                        delta = np.frombuffer(frame_data, dtype=np.int16).reshape(frame_shape)
                        frames[i] = (frames[i-1].astype(np.int32) + delta).astype(data_dtype)
                        offset += delta_frame_bytes

                data_array = frames
            else:
                # No delta encoding - simple reshape
                if shuffled:
                    decompressed_data = self._unshuffle_bytes(decompressed_data, itemsize=data_dtype.itemsize)
                data_array = np.frombuffer(decompressed_data, dtype=data_dtype).reshape(data_shape)

            return da.from_array(data_array)

    def decode_zst(self, path_bp1, path_bp2=None):
        """
        Decode zst compressed files, supporting both old and new formats.
        New format supports delta encoding and byte shuffle for better compression.
        Accepts a glob string or an already-resolved list of paths.
        """
        print('Decoding zst images...')
        files_bp1 = self._resolve_paths(path_bp1)

        bp1 = []
        for file_path in files_bp1:
            bp1.append(self._decode_zst_file(file_path))

        if path_bp2:
            files_bp2 = self._resolve_paths(path_bp2)
            self.path_encoded_bp2 = files_bp2
            assert len(files_bp1) == len(files_bp2), 'Error: Both biplanes must have the same number of images.'
            bp2 = []
            for file_path in files_bp2:
                bp2.append(self._decode_zst_file(file_path))
            return bp1, bp2
        return bp1, None
        

    def process_frames(self):
        print('Lazily Processing images...')
        if self.encoded_bp2 is None:
            return [da.where(self.sparse_bp1[i]!=0,self.sparse_bp1[i],self.encoded_bp1[i]) for i in range(len(self.encoded_bp1))], None
        return [da.where(self.sparse_bp1[i]!=0,self.sparse_bp1[i],self.encoded_bp1[i]) for i in range(len(self.encoded_bp1))], [da.where(self.sparse_bp2[j]!=0,self.sparse_bp2[j],self.encoded_bp2[j])for j in range(len(self.encoded_bp2))]


    def save_as_dat(self, frames, filename, dtype='uint16'):
        """
        Save frames to a .dat file using numpy.memmap.

        Parameters:
            frames (dask.array): Dask array containing the frames to save.
            filename (str): Path to the output .dat file.
            dtype (str): Data type of the frames.
        """
        # Ensure the dtype is supported
        dtype = np.dtype(dtype)
        
        dtype = np.dtype(dtype)
        with open(filename, 'wb') as f:
            # Iterate over frames in chunks
            for i in range(0, frames.shape[0], self.chunk_size):
                chunk = frames[i:i + self.chunk_size].compute()
                f.write(chunk.tobytes())


    # ------------------------------------------------------------------
    # Streaming reconstruction helpers
    # ------------------------------------------------------------------

    def _iter_video_chunks_streaming(self, video_path, chunk_size):
        """Single-pass PyAV decode → bounded-memory uint16 chunks."""
        container = av.open(video_path)
        try:
            stream = container.streams.video[0]
            buf = []
            start = 0
            for frame in container.decode(stream):
                np_frame = frame.to_ndarray(format="gray16le").astype(np.uint16, copy=False)
                buf.append(np_frame)
                if len(buf) == chunk_size:
                    end = start + len(buf)
                    yield start, end, np.stack(buf, axis=0)
                    start = end
                    buf = []
            if buf:
                end = start + len(buf)
                yield start, end, np.stack(buf, axis=0)
        finally:
            container.close()

    def _load_roi_for_streaming(self, npz_path):
        """Load ROI as ``{coords, data, shape}`` without densifying anything."""
        if not npz_path or not os.path.exists(npz_path):
            return None
        coo = self.load_sparse_matrix_from_npz(npz_path)
        if coo is None:
            return None
        return {
            "coords": np.asarray(coo.coords),
            "data": np.asarray(coo.data),
            "shape": tuple(coo.shape),
        }

    def _apply_residual_and_roi_to_chunk(self, decoded_chunk, start, end,
                                         residual_chunk_iter, roi_info,
                                         target_dtype, H, W, context=None):
        """Apply matching residual + ROI slab to one decoded chunk.

        Memory: a single int64 buffer of ``decoded_chunk.size``, never the
        whole video. Raises on misalignment so silent corruption can't slip
        through.
        """
        n_frames = end - start
        if decoded_chunk.shape != (n_frames, H, W):
            raise ValueError(
                f"decoded chunk shape {decoded_chunk.shape} != ({n_frames},{H},{W})"
                + (f" (context: {context})" if context else "")
            )
        target_dtype = np.dtype(target_dtype)
        out = decoded_chunk

        if residual_chunk_iter is not None:
            try:
                res_start, res_end, res_chunk, _h = next(residual_chunk_iter)
            except StopIteration:
                raise ValueError(
                    f"residual stream exhausted before video chunk [{start},{end})"
                    + (f" (context: {context})" if context else "")
                )
            if (res_start, res_end) != (start, end):
                raise ValueError(
                    f"residual chunk [{res_start},{res_end}) misaligned with "
                    f"decoded chunk [{start},{end})"
                    + (f" (context: {context})" if context else "")
                )
            if res_chunk.shape != decoded_chunk.shape:
                raise ValueError(
                    f"residual chunk shape {res_chunk.shape} != decoded {decoded_chunk.shape}"
                    + (f" (context: {context})" if context else "")
                )
            summed = decoded_chunk.astype(np.int64) + res_chunk.astype(np.int64)
            if np.issubdtype(target_dtype, np.integer):
                info = np.iinfo(target_dtype)
                np.clip(summed, info.min, info.max, out=summed)
            out = summed.astype(target_dtype)
        elif decoded_chunk.dtype != target_dtype:
            out = decoded_chunk.astype(target_dtype)

        if roi_info is not None:
            coords = roi_info["coords"]
            if coords.size:
                sel = (coords[0] >= start) & (coords[0] < end)
                if sel.any():
                    cc = coords[:, sel]
                    vv = roi_info["data"][sel]
                    if not out.flags.writeable or out is decoded_chunk:
                        out = out.copy()
                    out[cc[0] - start, cc[1], cc[2]] = vv.astype(target_dtype)
        return out

    def _streaming_reconstruct_one(self, video_path, residual_path, roi_npz_path,
                                   T, H, W, original_dtype):
        """Generator yielding ``(start, end, chunk_uint_target)`` for one video.

        Aligns the chunk size to the residual sidecar's ``chunk_size`` when one
        is present so the chunk-by-chunk apply is exact and zero-misaligned.
        """
        residual_chunk_iter = None
        chunk_size_resid = None
        if residual_path and os.path.exists(residual_path):
            header = read_residual_header(residual_path)
            if header["shape"] != (T, H, W):
                raise ValueError(
                    f"residual {residual_path} shape {header['shape']} != "
                    f"video {(T, H, W)} for {video_path}"
                )
            chunk_size_resid = header.get("chunk_size") or T
            residual_chunk_iter = iter_residual_chunks_from_path(residual_path)

        chunk_size = max(1, int(chunk_size_resid or self.chunk_size or 50))

        roi_info = None
        if roi_npz_path:
            roi_info = self._load_roi_for_streaming(roi_npz_path)
            if roi_info is not None and roi_info["shape"] != (T, H, W):
                raise ValueError(
                    f"ROI {roi_npz_path} shape {roi_info['shape']} != video {(T, H, W)} "
                    f"for {video_path}"
                )

        last_end = 0
        for start, end, decoded_chunk in self._iter_video_chunks_streaming(video_path, chunk_size):
            if end > T:
                raise ValueError(
                    f"video {video_path} produced more than {T} frames (reached {end})"
                )
            if decoded_chunk.shape[1:] != (H, W):
                raise ValueError(
                    f"decoded frame size {decoded_chunk.shape[1:]} != ({H},{W}) "
                    f"for {video_path}"
                )
            yield start, end, self._apply_residual_and_roi_to_chunk(
                decoded_chunk, start, end, residual_chunk_iter, roi_info,
                np.dtype(original_dtype), H, W, context=video_path,
            )
            last_end = end

        if last_end != T:
            raise ValueError(
                f"video {video_path} ended at frame {last_end} but expected {T}"
            )
        if residual_chunk_iter is not None:
            leftover = next(residual_chunk_iter, None)
            if leftover is not None:
                raise ValueError(
                    f"residual {residual_path} has chunk(s) past the end of video {video_path}"
                )

    def _residual_path_for_index(self, residual_paths, k):
        if residual_paths is None:
            return None
        if isinstance(residual_paths, (list, tuple)):
            return residual_paths[k] if k < len(residual_paths) else None
        if isinstance(residual_paths, str):
            matched = sorted(glob.glob(residual_paths))
            return matched[k] if k < len(matched) else None
        return None

    def _roi_npz_path_for_index(self, sparse_paths, k):
        if sparse_paths is None:
            return None
        resolved = self._resolve_paths(sparse_paths)
        return resolved[k] if k < len(resolved) else None

    def _original_dtype_for(self, residual_path):
        if residual_path and os.path.exists(residual_path):
            try:
                return np.dtype(read_residual_header(residual_path)["original_dtype"])
            except Exception:
                pass
        return np.dtype(np.uint16)

    def _write_streaming_output(self, filename, chunks_gen, T, H, W, original_dtype,
                                file_metadata, debug_prefix=""):
        """Drive the chunk generator and write to ``self.output_format``.

        TIFF: allocate the final output buffer, fill via streaming chunks, then
        hand off to ``write_tiff_file`` which knows how to embed metadata.
        DAT: stream raw bytes incrementally — never holds more than one chunk.
        """
        if self.output_format == "tiff":
            target_dtype = np.dtype(original_dtype)
            out = np.empty((T, H, W), dtype=target_dtype)
            for start, end, chunk in chunks_gen:
                out[start:end] = chunk
            self.write_tiff_file(filename, out, file_metadata, debug_prefix=debug_prefix)
        elif self.output_format == "dat":
            target_dtype = np.dtype(original_dtype)
            with open(filename, "wb") as f:
                for _start, _end, chunk in chunks_gen:
                    f.write(np.ascontiguousarray(chunk, dtype=target_dtype).tobytes())
        else:
            raise ValueError(f"Unsupported output_format: {self.output_format}")

    def _run_streaming(self):
        """Streaming-reconstruction driver. Memory bound: one chunk + final output."""
        print("Inflating images (streaming)...")
        if self.use_roi:
            print("Patching in ROI per chunk...")
        else:
            print("Reconstructing background only (no ROI patch)...")

        n_videos = len(self.path_encoded_bp1)
        for k in range(n_videos):
            video_path = self.path_encoded_bp1[k]
            residual_path = self._residual_path_for_index(self.path_residual_bp1, k) if self.use_residuals else None
            roi_npz_path = self._roi_npz_path_for_index(self._path_sparse_bp1_resolved, k) if self.use_roi else None
            original_dtype = self._original_dtype_for(residual_path)
            T, H, W = self.encoded_bp1[k].shape

            input_file_name1 = os.path.splitext(os.path.split(os.path.normpath(video_path))[1])[0]
            output_filename1 = f"{self.output_path}{self.stem}_{input_file_name1}"
            ext = "tiff" if self.output_format == "tiff" else "dat"
            out_path1 = f"{output_filename1}.{ext}"
            file_metadata = self.metadata_bp1[k] if k < len(self.metadata_bp1) else {}

            chunks_gen = self._streaming_reconstruct_one(
                video_path, residual_path, roi_npz_path, T, H, W, original_dtype
            )
            self._write_streaming_output(
                out_path1, chunks_gen, T, H, W, original_dtype, file_metadata,
                debug_prefix="BP1 ",
            )

            if self.encoded_bp2 is not None:
                video_path2 = self.path_encoded_bp2[k]
                residual_path2 = self._residual_path_for_index(self.path_residual_bp2, k) if self.use_residuals else None
                roi_npz_path2 = self._roi_npz_path_for_index(self._path_sparse_bp2_resolved, k) if self.use_roi else None
                original_dtype2 = self._original_dtype_for(residual_path2)
                T2, H2, W2 = self.encoded_bp2[k].shape

                input_file_name2 = os.path.splitext(os.path.split(os.path.normpath(video_path2))[1])[0]
                output_filename2 = f"{self.output_path}{self.stem}_{input_file_name2}"
                out_path2 = f"{output_filename2}.{ext}"
                file_metadata2 = self.metadata_bp2[k] if self.metadata_bp2 and k < len(self.metadata_bp2) else {}

                chunks_gen2 = self._streaming_reconstruct_one(
                    video_path2, residual_path2, roi_npz_path2, T2, H2, W2, original_dtype2
                )
                self._write_streaming_output(
                    out_path2, chunks_gen2, T2, H2, W2, original_dtype2, file_metadata2,
                    debug_prefix="BP2 ",
                )

            gc.collect()

        print("Done.")
        self.cleanup_temp_directories()

    # ------------------------------------------------------------------
    # Legacy eager run (kept for streaming=False / zstd image stacks)
    # ------------------------------------------------------------------

    def run(self):
        if getattr(self, "streaming", False):
            return self._run_streaming()
        print('Inflating images...')
        show_progress_bar = False
        try:
            get_ipython()
            show_progress_bar = True
        except NameError:
            show_progress_bar = False
        if self.use_roi:
            print('Patching in ROI...')
            if show_progress_bar:
                    progress_bar1 = tqdm(total=len(self.processed_bp1), desc="Extracting frames from plane 1", position=0, leave=True)
                    if self.encoded_bp2 is not None:
                        progress_bar2 = tqdm(total=len(self.processed_bp2), desc="Extracting frames from plane 1", position=0, leave=True)
            for k in range(len(self.processed_bp1)):
                num_frames = self.processed_bp1[k].shape[0]

                input_file_name1 = os.path.splitext(os.path.split(os.path.normpath(self.path_encoded_bp1[k]))[1])[0]
                output_filename1 = f'{self.output_path}{self.stem}_{input_file_name1}'
                # filename1 = f'{self.output_path}{self.stem}_bp1_part_{k}.tiff'
                if self.output_format == 'tiff':
                    tiff_filename1 = f'{output_filename1}.tiff'
                    all_frames = self.processed_bp1[k].compute()
                    # Get metadata for this file
                    file_metadata = self.metadata_bp1[k] if k < len(self.metadata_bp1) else {}
                    
                    # print(f"Metadata_bp1 length: {len(self.metadata_bp1) if self.metadata_bp1 else 0}")
                    # print(f"k={k}, file_metadata keys: {list(file_metadata.keys()) if file_metadata else 'EMPTY'}")
                    # print(f"is_ome: {file_metadata.get('is_ome') if file_metadata else 'N/A'}")
                    # print(f"has ome_xml: {'ome_xml' in file_metadata if file_metadata else 'N/A'}")
                    
                    # Use imwrite instead of TiffWriter for OME-XML compatibility
                    ome_condition = file_metadata.get('is_ome') and 'ome_xml' in file_metadata
                    # print(f"OME condition: {ome_condition}")
                    
                    # Write TIFF with automatic IFD handling
                    self.write_tiff_file(tiff_filename1, all_frames, file_metadata, debug_prefix="BP1 ")
                    if show_progress_bar:
                        progress_bar1.update(num_frames)
                        progress_bar1.close()
                elif self.output_format == 'dat':
                    dat_filename1 = f'{output_filename1}.dat'
                    self.save_as_dat(self.processed_bp1[k], dat_filename1)
                    if show_progress_bar:
                        progress_bar1.update(num_frames)
                    if show_progress_bar:
                        progress_bar1.close()

                if self.encoded_bp2 is not None:
                    input_file_name2 = os.path.splitext(os.path.split(os.path.normpath(self.path_encoded_bp2[k]))[1])[0]
                    output_filename2 = f'{self.output_path}{self.stem}_{input_file_name2}.tiff'
                    # filename2 = f'{self.output_path}{self.stem}_bp2_part_{k}.tiff'
                    if self.output_format == 'tiff':
                        tiff_filename2 = f'{output_filename2}.tiff'
                        if show_progress_bar:
                            progress_bar2 = tqdm(total=len(self.processed_bp2), desc="Extracting frames from plane 2", position=0, leave=True)
                        all_frames = self.processed_bp2[k].compute()
                        # Get metadata for this file
                        file_metadata = self.metadata_bp2[k] if self.metadata_bp2 and k < len(self.metadata_bp2) else {}
                        
                        # DEBUG: Print metadata debugging info for BP2
                        # Write TIFF with automatic IFD handling
                        self.write_tiff_file(tiff_filename2, all_frames, file_metadata, debug_prefix="BP2 ")
                        if show_progress_bar:
                            progress_bar2.update(num_frames)
                            progress_bar2.close()
                    elif self.output_format == 'dat':
                        dat_filename2 = f'{output_filename2}.dat'
                        self.save_as_dat(self.processed_bp2[k], dat_filename2)
                        if show_progress_bar:
                            progress_bar2.update(num_frames)
                        if show_progress_bar:
                            progress_bar2.close()

                gc.collect()
                
            
        else:
            print('Extracting background only...')
            
            for k in range(len(self.encoded_bp1)):
                num_frames = self.encoded_bp1[k].shape[0]
                # if type(self.path_encoded_bp1) == list:
                input_file_name1 = os.path.splitext(os.path.split(os.path.normpath(self.path_encoded_bp1[k]))[1])[0]
                # else:
                    # input_file_name1 = os.path.splitext(os.path.split(os.path.normpath(self.path_encoded_bp1))[1])[0]
                
                filename1 = f'{self.output_path}{self.stem}_{input_file_name1}.tiff'
                # filename1 = f'{self.output_path}{self.stem}_bp1_part_{k}.tiff'

                if show_progress_bar:
                    progress_bar = tqdm(total=len(self.encoded_bp1), desc="Extracting frames from plane 1", position=0, leave=True)
                all_frames = self.encoded_bp1[k].compute()
                # Get metadata for this file
                file_metadata = self.metadata_bp1[k] if k < len(self.metadata_bp1) else {}
                
                # Write TIFF with automatic IFD handling
                self.write_tiff_file(filename1, all_frames, file_metadata, debug_prefix="ENCODED BP1 ")
                if show_progress_bar:
                    progress_bar.update(num_frames)
                    progress_bar.close()
                if self.encoded_bp2 is not None:
                    # if type(self.path_encoded_bp2) == list:
                    # print ('encoded_bp2:',self.path_encoded_bp2)
                    input_file_name2 = os.path.splitext(os.path.split(os.path.normpath(self.path_encoded_bp2[k]))[1])[0]
                    # else:
                        # input_file_name2 = os.path.splitext(os.path.split(os.path.normpath(self.path_encoded_bp2))[1])[0]
                    filename2 = f'{self.output_path}{self.stem}_{input_file_name2}.tiff'
                    # filename2 = f'{self.output_path}{self.stem}_bp2_part_{k}.tiff'
                    if show_progress_bar:
                        progress_bar = tqdm(total=len(self.encoded_bp1), desc="Extracting frames from plane 2", position=0, leave=True)
                    all_frames = self.encoded_bp2[k].compute()
                    # Get metadata for this file
                    file_metadata = self.metadata_bp2[k] if self.metadata_bp2 and k < len(self.metadata_bp2) else {}
                    
                    # Write TIFF with automatic IFD handling
                    self.write_tiff_file(filename2, all_frames, file_metadata, debug_prefix="ENCODED BP2 ")
                    if show_progress_bar:
                        progress_bar.update(num_frames)
                        progress_bar.close()
                gc.collect()
        print('Done.')
        
        # Clean up temporary directories used for MKV extraction
        self.cleanup_temp_directories()