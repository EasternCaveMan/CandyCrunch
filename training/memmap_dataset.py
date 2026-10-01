import ctypes
import fcntl
import gc
import json
import os
import pickle
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch


CACHE_VERSION = 1
METADATA_INDICES = (4, 5, 6, 7, 8, 9)


def _release_heap():
    gc.collect()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _source_signature(source_path):
    stat = source_path.stat()
    return {
        "path": str(source_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _cache_is_valid(cache_dir, source_signature, model_type):
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        with manifest_path.open() as file:
            manifest = json.load(file)
    except (OSError, ValueError):
        return False
    if manifest.get("cache_version") != CACHE_VERSION:
        return False
    if manifest.get("source") != source_signature:
        return False
    if manifest.get("model_type") != model_type:
        return False
    arrays = manifest.get("arrays")
    return isinstance(arrays, list) and all(
        (cache_dir / filename).exists() for filename in arrays
    )


def _write_array(features, feature_index, output_path):
    first = np.asarray(features[0][feature_index])
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(features), *first.shape),
    )
    for index, sample in enumerate(features):
        value = np.asarray(sample[feature_index], dtype=np.float32)
        if value.shape != first.shape:
            raise ValueError(
                f"Feature {feature_index} has inconsistent shape at sample {index}: "
                f"expected {first.shape}, got {value.shape}"
            )
        output[index] = value
    output.flush()
    del output


def _write_metadata(features, output_path):
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(features), len(METADATA_INDICES)),
    )
    for row_index, sample in enumerate(features):
        output[row_index] = [sample[index] for index in METADATA_INDICES]
    output.flush()
    del output


def ensure_memmap_cache(feature_path, cache_root, model_type, memory_reporter=None):
    feature_path = Path(feature_path)
    cache_root = Path(cache_root)
    source_signature = _source_signature(feature_path)
    signature = f"{source_signature['size']}_{source_signature['mtime_ns']}"
    cache_dir = cache_root / f"v{CACHE_VERSION}" / model_type.lower() / signature
    cache_dir.parent.mkdir(parents=True, exist_ok=True)

    lock_path = cache_root / f"v{CACHE_VERSION}" / f".{signature}.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if _cache_is_valid(cache_dir, source_signature, model_type):
            return cache_dir

        print(f"Building one-time {model_type} memory-map cache from {feature_path}")
        if memory_reporter is not None:
            memory_reporter("before pickle conversion")

        with feature_path.open("rb") as file:
            features = pickle.load(file)

        if not features:
            raise ValueError(f"Feature file is empty: {feature_path}")
        if memory_reporter is not None:
            memory_reporter("pickle loaded for conversion")

        temp_dir = Path(tempfile.mkdtemp(prefix=f".{signature}-", dir=cache_dir.parent))
        try:
            arrays = ["metadata.npy"]
            if model_type == "CNN":
                _write_array(features, 0, temp_dir / "binned_intensities.npy")
                _write_array(features, 2, temp_dir / "mz_remainder.npy")
                arrays.extend(["binned_intensities.npy", "mz_remainder.npy"])
            elif model_type == "Transformer":
                _write_array(features, 1, temp_dir / "peak_list.npy")
                arrays.append("peak_list.npy")
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            _write_metadata(features, temp_dir / "metadata.npy")
            manifest = {
                "cache_version": CACHE_VERSION,
                "model_type": model_type,
                "source": source_signature,
                "samples": len(features),
                "arrays": arrays,
            }
            with (temp_dir / "manifest.json").open("w") as file:
                json.dump(manifest, file, indent=2)
            os.replace(temp_dir, cache_dir)
        except BaseException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        finally:
            del features
            _release_heap()

        if memory_reporter is not None:
            memory_reporter("pickle released after conversion")
        return cache_dir