#!/usr/bin/env python3
"""
K-means based downsampling for representative, low-noise MS2 spectra.

Default behavior:
  1. Group spectra by glycan.
  2. Denoise and vectorize each spectrum from peak_d.
  3. Cluster spectra inside each glycan group into K clusters.
  4. For each cluster, select N real spectra nearest to the cluster center.

N is dynamic per cluster:
  sqrt      N = ceil(keep_scale * sqrt(cluster_size))
  log2      N = ceil(keep_scale * log2(cluster_size + 1))
  fraction  N = ceil(keep_fraction * cluster_size)

Recommended use inside build_training_pickles_2.py:

    from downsample_representative_spectra_V2 import downsample_representative_spectra_V2

    combined = downsample_representative_spectra_V2(
        combined,
        group_cols=("glycan",),
        n_clusters=5,
        keep_policy="sqrt",
        keep_scale=2.0,
    )
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sqlite3
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def thread_limit_context(inner_threads: int):
    if threadpool_limits is None or inner_threads is None or inner_threads <= 0:
        return nullcontext()
    return threadpool_limits(limits=inner_threads)


def _safe_peak_parse(value):
    if isinstance(value, dict):
        return {float(k): float(v) for k, v in value.items()}

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return None
        if isinstance(parsed, dict):
            return {float(k): float(v) for k, v in parsed.items()}

    return None


def _normalise_spectrum(spec):
    total = float(sum(spec.values()))
    if total <= 0:
        return None
    return {mz: intensity / total for mz, intensity in spec.items()}


@dataclass
class Entry:
    row_id: object
    group_key: str
    glycan: str
    quality: float
    n_peaks: int
    retained_fraction: float
    idx: np.ndarray
    val: np.ndarray


@dataclass
class ClusterSelection:
    entry: Entry
    cluster_id: int
    cluster_size: int
    rank_in_cluster: int
    distance_to_center: float
    similarity_to_center: float


def keep_count_for_cluster(cluster_size: int, args) -> int:
    """Return dynamic N for one cluster, with no max_keep cap."""
    if cluster_size <= 0:
        return 0

    if args.keep_policy == "sqrt":
        n = math.ceil(args.keep_scale * math.sqrt(cluster_size))
    elif args.keep_policy == "log2":
        n = math.ceil(args.keep_scale * math.log2(cluster_size + 1))
    elif args.keep_policy == "fraction":
        n = math.ceil(args.keep_fraction * cluster_size)
    else:
        raise ValueError(f"Unknown keep policy: {args.keep_policy}")

    n = max(args.min_keep, n)
    n = min(n, cluster_size)
    return int(n)


def make_group_key(row: dict, group_cols: list[str]) -> str:
    payload = {col: str(row.get(col, "")).strip() for col in group_cols}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def make_group_key_from_values(group_cols, group_value) -> str:
    if not isinstance(group_value, tuple):
        group_value = (group_value,)
    payload = dict(zip(group_cols, [str(v) for v in group_value]))
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def denoise_and_vectorize(
    peak_d: dict,
    min_mz: float,
    max_mz: float,
    bin_width: float,
    rel_intensity_min: float,
    min_clean_peaks: int,
):
    spec = _normalise_spectrum(peak_d)
    if spec is None:
        return None

    mzs = np.asarray(list(spec.keys()), dtype=np.float32)
    intensities = np.asarray(list(spec.values()), dtype=np.float32)

    valid = (
        np.isfinite(mzs)
        & np.isfinite(intensities)
        & (mzs >= min_mz)
        & (mzs <= max_mz)
        & (intensities > 0)
    )
    mzs = mzs[valid]
    intensities = intensities[valid]

    if intensities.size == 0:
        return None

    total_before = float(intensities.sum())
    base_peak = float(intensities.max())
    if total_before <= 0 or base_peak <= 0:
        return None

    keep = (intensities / base_peak) >= rel_intensity_min
    if int(keep.sum()) < min_clean_peaks:
        return None

    mzs = mzs[keep]
    intensities = intensities[keep]
    retained_fraction = float(intensities.sum() / total_before)

    n_bins = int(math.floor((max_mz - min_mz) / bin_width)) + 1
    bins = np.floor((mzs - min_mz) / bin_width).astype(np.int32)
    valid_bins = (bins >= 0) & (bins < n_bins)
    bins = bins[valid_bins]
    intensities = intensities[valid_bins]

    if bins.size == 0:
        return None

    order = np.argsort(bins)
    bins = bins[order]
    intensities = intensities[order]

    unique_bins, starts = np.unique(bins, return_index=True)
    summed = np.add.reduceat(intensities, starts).astype(np.float32)

    total = float(summed.sum())
    if total <= 0:
        return None

    summed /= total
    values = np.sqrt(summed)
    norm = float(np.linalg.norm(values))
    if norm <= 0:
        return None

    values = (values / norm).astype(np.float32)
    unique_bins = unique_bins.astype(np.int32)
    quality = retained_fraction * math.log1p(len(unique_bins))

    return unique_bins, values, quality, len(unique_bins), retained_fraction, n_bins


def entries_to_csr(entries: list[Entry], n_bins: int):
    indptr = [0]
    indices = []
    data = []

    for entry in entries:
        indices.extend(entry.idx.tolist())
        data.extend(entry.val.tolist())
        indptr.append(len(indices))

    return sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(len(entries), n_bins),
        dtype=np.float32,
    )


def distances_to_center(X_cluster, center: np.ndarray) -> np.ndarray:
    center = np.asarray(center, dtype=np.float32)
    center_norm_sq = float(np.dot(center, center))
    similarities = np.asarray(X_cluster.dot(center)).reshape(-1)

    # Rows are L2-normalized in denoise_and_vectorize, so row_norm_sq is ~1.
    distances_sq = 1.0 + center_norm_sq - (2.0 * similarities)
    distances_sq = np.maximum(distances_sq, 0.0)
    return np.sqrt(distances_sq).astype(np.float32), similarities.astype(np.float32)


def select_representatives_kmeans(entries: list[Entry], n_bins: int, args):
    if not entries:
        return [], {
            "group_size": 0,
            "n_clusters_requested": args.n_clusters,
            "n_clusters_used": 0,
            "selected_count": 0,
        }

    X = entries_to_csr(entries, n_bins)
    n_clusters_used = min(args.n_clusters, len(entries))

    if n_clusters_used <= 1:
        labels = np.zeros(len(entries), dtype=np.int32)
        centers = np.asarray(X.mean(axis=0), dtype=np.float32).reshape(1, -1)
    elif n_clusters_used == len(entries):
        labels = np.arange(len(entries), dtype=np.int32)
        centers = X.toarray().astype(np.float32)
    else:
        with thread_limit_context(args.inner_threads):
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters_used,
                random_state=args.random_state,
                batch_size=args.kmeans_batch_size,
                n_init=args.kmeans_n_init,
                max_iter=args.kmeans_max_iter,
                reassignment_ratio=args.reassignment_ratio,
            )
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_.astype(np.float32)

    selected = []

    for cluster_id in sorted(np.unique(labels).tolist()):
        cluster_positions = np.flatnonzero(labels == cluster_id)
        cluster_size = int(cluster_positions.size)

        if cluster_size < args.min_cluster_size:
            continue

        keep_n = keep_count_for_cluster(cluster_size, args)
        if keep_n <= 0:
            continue

        X_cluster = X[cluster_positions]
        distances, similarities = distances_to_center(X_cluster, centers[cluster_id])

        ordered_local = sorted(
            range(cluster_size),
            key=lambda i: (
                float(distances[i]),
                -float(entries[int(cluster_positions[i])].quality),
            ),
        )

        for rank, local_pos in enumerate(ordered_local[:keep_n], start=1):
            global_pos = int(cluster_positions[local_pos])
            selected.append(
                ClusterSelection(
                    entry=entries[global_pos],
                    cluster_id=int(cluster_id),
                    cluster_size=cluster_size,
                    rank_in_cluster=rank,
                    distance_to_center=float(distances[local_pos]),
                    similarity_to_center=float(similarities[local_pos]),
                )
            )

    selected.sort(
        key=lambda sel: (
            sel.entry.glycan,
            sel.cluster_id,
            sel.rank_in_cluster,
            sel.entry.row_id,
        )
    )

    return selected, {
        "group_size": len(entries),
        "n_clusters_requested": args.n_clusters,
        "n_clusters_used": n_clusters_used,
        "selected_count": len(selected),
    }


def build_selection_result(
    group_order: int,
    group_key: str,
    group_size: int,
    indexed_count: int,
    skipped_count: int,
    selected: list[ClusterSelection],
    stats: dict,
):
    selected_row_ids = []
    selected_report = []
    cluster_counts = {}

    for sel in selected:
        selected_row_ids.append(sel.entry.row_id)
        cluster_counts.setdefault(sel.cluster_id, 0)
        cluster_counts[sel.cluster_id] += 1
        selected_report.append(
            {
                "row_id": sel.entry.row_id,
                "group_key": group_key,
                "glycan": sel.entry.glycan,
                "cluster_id": sel.cluster_id,
                "cluster_size": sel.cluster_size,
                "rank_in_cluster": sel.rank_in_cluster,
                "distance_to_center": f"{sel.distance_to_center:.6f}",
                "similarity_to_center": f"{sel.similarity_to_center:.6f}",
                "n_peaks": sel.entry.n_peaks,
                "retained_fraction": f"{sel.entry.retained_fraction:.6f}",
                "quality": f"{sel.entry.quality:.6f}",
            }
        )

    summary_report = {
        "group_key": group_key,
        "group_size": group_size,
        "indexed_count": indexed_count,
        "skipped_count": skipped_count,
        "n_clusters_requested": stats["n_clusters_requested"],
        "n_clusters_used": stats["n_clusters_used"],
        "selected_count": stats["selected_count"],
        "selected_per_cluster": json.dumps(cluster_counts, sort_keys=True),
    }

    return {
        "group_order": group_order,
        "selected_row_ids": selected_row_ids,
        "selected_report": selected_report,
        "summary_report": summary_report,
        "indexed_count": indexed_count,
        "skipped_count": skipped_count,
    }


def process_dataframe_group(group_order, group_cols, group_value, group_df, n_bins: int, args):
    entries = []
    skipped = 0
    group_key = make_group_key_from_values(group_cols, group_value)

    with thread_limit_context(args.inner_threads):
        for row_index, peak_value, glycan in zip(
            group_df.index,
            group_df["peak_d"].to_numpy(),
            group_df["glycan"].to_numpy(),
        ):
            peak_d = _safe_peak_parse(peak_value)
            if peak_d is None:
                skipped += 1
                continue

            packed = denoise_and_vectorize(
                peak_d,
                min_mz=args.min_mz,
                max_mz=args.max_mz,
                bin_width=args.bin_width,
                rel_intensity_min=args.rel_intensity_min,
                min_clean_peaks=args.min_clean_peaks,
            )
            if packed is None:
                skipped += 1
                continue

            idx, val, quality, n_peaks, retained_fraction, packed_n_bins = packed
            if packed_n_bins != n_bins:
                raise RuntimeError("Unexpected n_bins change.")

            entries.append(
                Entry(
                    row_id=row_index,
                    group_key=group_key,
                    glycan=str(glycan),
                    quality=float(quality),
                    n_peaks=int(n_peaks),
                    retained_fraction=float(retained_fraction),
                    idx=idx,
                    val=val,
                )
            )

        selected, stats = select_representatives_kmeans(entries, n_bins, args)

    return build_selection_result(
        group_order=group_order,
        group_key=group_key,
        group_size=len(group_df),
        indexed_count=len(entries),
        skipped_count=skipped,
        selected=selected,
        stats=stats,
    )


def process_sqlite_group(db_path: str, group_order: int, group_key: str, indexed_count: int, n_bins: int, args):
    with thread_limit_context(args.inner_threads):
        conn = sqlite3.connect(db_path)
        try:
            entries = load_group_entries(conn, group_key)
        finally:
            conn.close()

        selected, stats = select_representatives_kmeans(entries, n_bins, args)

    return build_selection_result(
        group_order=group_order,
        group_key=group_key,
        group_size=indexed_count,
        indexed_count=len(entries),
        skipped_count=0,
        selected=selected,
        stats=stats,
    )


def merge_group_results(results):
    selected_row_ids = set()
    selected_report = []
    summary_report = []
    indexed = 0
    skipped = 0

    for result in sorted(results, key=lambda item: item["group_order"]):
        selected_row_ids.update(result["selected_row_ids"])
        selected_report.extend(result["selected_report"])
        summary_report.append(result["summary_report"])
        indexed += result["indexed_count"]
        skipped += result["skipped_count"]

    return selected_row_ids, selected_report, summary_report, indexed, skipped


def create_index_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE spectra (
            row_id INTEGER PRIMARY KEY,
            group_key TEXT NOT NULL,
            glycan TEXT NOT NULL,
            quality REAL NOT NULL,
            n_peaks INTEGER NOT NULL,
            retained_fraction REAL NOT NULL,
            idx BLOB NOT NULL,
            val BLOB NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX spectra_group_key_idx ON spectra(group_key)")
    return conn


def index_spectra(input_csv: Path, db_path: Path, args):
    if db_path.exists():
        db_path.unlink()

    conn = create_index_db(db_path)
    raise_csv_field_limit()

    skipped = 0
    inserted = 0
    n_bins_seen = None
    batch = []

    with input_csv.open("r", newline="") as fh:
        reader = csv.DictReader(fh)

        for row_id, row in tqdm(enumerate(reader), desc="Indexing spectra", unit="spectra"):
            glycan = str(row.get("glycan", "")).strip()
            if not glycan:
                skipped += 1
                continue

            peak_d = _safe_peak_parse(row.get("peak_d"))
            if peak_d is None:
                skipped += 1
                continue

            packed = denoise_and_vectorize(
                peak_d,
                min_mz=args.min_mz,
                max_mz=args.max_mz,
                bin_width=args.bin_width,
                rel_intensity_min=args.rel_intensity_min,
                min_clean_peaks=args.min_clean_peaks,
            )
            if packed is None:
                skipped += 1
                continue

            idx, val, quality, n_peaks, retained_fraction, n_bins = packed
            if n_bins_seen is None:
                n_bins_seen = n_bins
            elif n_bins_seen != n_bins:
                raise RuntimeError("Unexpected n_bins change.")

            group_key = make_group_key(row, args.group_cols)
            batch.append(
                (
                    row_id,
                    group_key,
                    glycan,
                    float(quality),
                    int(n_peaks),
                    float(retained_fraction),
                    sqlite3.Binary(idx.astype(np.int32).tobytes()),
                    sqlite3.Binary(val.astype(np.float32).tobytes()),
                )
            )
            inserted += 1

            if len(batch) >= args.sqlite_commit_every:
                conn.executemany(
                    """
                    INSERT INTO spectra
                    (row_id, group_key, glycan, quality, n_peaks, retained_fraction, idx, val)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()
                batch.clear()

    if batch:
        conn.executemany(
            """
            INSERT INTO spectra
            (row_id, group_key, glycan, quality, n_peaks, retained_fraction, idx, val)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
        conn.commit()

    return conn, inserted, skipped, n_bins_seen


def load_group_entries(conn, group_key: str) -> list[Entry]:
    rows = conn.execute(
        """
        SELECT row_id, group_key, glycan, quality, n_peaks, retained_fraction, idx, val
        FROM spectra
        WHERE group_key = ?
        ORDER BY row_id
        """,
        (group_key,),
    ).fetchall()

    entries = []
    for row_id, group_key, glycan, quality, n_peaks, retained_fraction, idx_blob, val_blob in rows:
        entries.append(
            Entry(
                row_id=int(row_id),
                group_key=group_key,
                glycan=glycan,
                quality=float(quality),
                n_peaks=int(n_peaks),
                retained_fraction=float(retained_fraction),
                idx=np.frombuffer(idx_blob, dtype=np.int32).copy(),
                val=np.frombuffer(val_blob, dtype=np.float32).copy(),
            )
        )

    return entries


def process_groups(conn, n_bins: int, args):
    group_rows = conn.execute(
        "SELECT group_key, COUNT(*) FROM spectra GROUP BY group_key ORDER BY group_key"
    ).fetchall()

    db_path = conn.execute("PRAGMA database_list").fetchone()[2]

    if args.n_jobs == 1 or not db_path:
        results = []
        iterator = tqdm(
            enumerate(group_rows),
            total=len(group_rows),
            desc="Clustering groups",
            unit="groups",
        )
        for group_order, (group_key, indexed_count) in iterator:
            entries = load_group_entries(conn, group_key)
            selected, stats = select_representatives_kmeans(entries, n_bins, args)
            results.append(
                build_selection_result(
                    group_order=group_order,
                    group_key=group_key,
                    group_size=indexed_count,
                    indexed_count=len(entries),
                    skipped_count=0,
                    selected=selected,
                    stats=stats,
                )
            )
    else:
        iterator = tqdm(
            enumerate(group_rows),
            total=len(group_rows),
            desc="Submitting groups",
            unit="groups",
        )
        results = Parallel(
            n_jobs=args.n_jobs,
            backend=args.parallel_backend,
            pre_dispatch=args.pre_dispatch,
        )(
            delayed(process_sqlite_group)(
                db_path,
                group_order,
                group_key,
                indexed_count,
                n_bins,
                args,
            )
            for group_order, (group_key, indexed_count) in iterator
        )

    selected_row_ids, selected_report, summary_report, _, _ = merge_group_results(results)
    return selected_row_ids, selected_report, summary_report


def write_selected_csv(input_csv: Path, output_csv: Path, selected_row_ids: set[int]) -> int:
    raise_csv_field_limit()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with input_csv.open("r", newline="") as fin, output_csv.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row_id, row in tqdm(enumerate(reader), desc="Writing CSV", unit="spectra"):
            if row_id in selected_row_ids:
                writer.writerow(row)
                written += 1

    return written


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_args_namespace(
    group_cols,
    n_clusters,
    keep_policy,
    min_keep,
    keep_scale,
    keep_fraction,
    min_mz,
    max_mz,
    bin_width,
    rel_intensity_min,
    min_clean_peaks,
    min_cluster_size,
    random_state,
    kmeans_batch_size,
    kmeans_n_init,
    kmeans_max_iter,
    reassignment_ratio,
    n_jobs,
    parallel_backend,
    pre_dispatch,
    inner_threads,
):
    if keep_policy not in {"sqrt", "log2", "fraction"}:
        raise ValueError("keep_policy must be one of: sqrt, log2, fraction")
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    if min_keep < 1:
        raise ValueError("min_keep must be >= 1")
    if keep_scale <= 0:
        raise ValueError("keep_scale must be > 0")
    if keep_fraction <= 0:
        raise ValueError("keep_fraction must be > 0")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be >= 1")
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    if parallel_backend not in {"loky", "threading"}:
        raise ValueError("parallel_backend must be 'loky' or 'threading'")
    if inner_threads < 1:
        raise ValueError("inner_threads must be >= 1")

    return argparse.Namespace(
        group_cols=list(group_cols),
        n_clusters=int(n_clusters),
        keep_policy=keep_policy,
        min_keep=int(min_keep),
        keep_scale=float(keep_scale),
        keep_fraction=float(keep_fraction),
        min_mz=float(min_mz),
        max_mz=float(max_mz),
        bin_width=float(bin_width),
        rel_intensity_min=float(rel_intensity_min),
        min_clean_peaks=int(min_clean_peaks),
        min_cluster_size=int(min_cluster_size),
        random_state=int(random_state),
        kmeans_batch_size=int(kmeans_batch_size),
        kmeans_n_init=int(kmeans_n_init),
        kmeans_max_iter=int(kmeans_max_iter),
        reassignment_ratio=float(reassignment_ratio),
        n_jobs=int(n_jobs),
        parallel_backend=parallel_backend,
        pre_dispatch=pre_dispatch,
        inner_threads=int(inner_threads),
    )


def downsample_representative_spectra_V2(
    combined=None,
    input_path: Union[str, Path] = "combined.csv",
    output_path: Optional[Union[str, Path]] = None,
    group_cols: Union[list[str], tuple[str, ...]] = ("glycan",),
    n_clusters: int = 5,
    keep_policy: str = "sqrt",
    min_keep: int = 1,
    keep_scale: float = 2.0,
    keep_fraction: float = 0.1,
    min_mz: float = 39.714,
    max_mz: float = 3000.0,
    bin_width: float = 1.0,
    rel_intensity_min: float = 0.005,
    min_clean_peaks: int = 5,
    min_cluster_size: int = 1,
    random_state: int = 42,
    kmeans_batch_size: int = 4096,
    kmeans_n_init: int = 10,
    kmeans_max_iter: int = 100,
    reassignment_ratio: float = 0.01,
    n_jobs: int = 1,
    parallel_backend: str = "loky",
    pre_dispatch: str = "2*n_jobs",
    inner_threads: int = 1,
    report_prefix: Optional[Union[str, Path]] = None,
    return_reports: bool = False,
    verbose: bool = True,
):
    """
    Return a dataframe containing K-means representative spectra.

    If combined is provided, it is used directly and array columns remain intact.
    If combined is None, input_path is read with pandas.
    """
    if combined is None:
        import pandas as pd

        combined = pd.read_csv(input_path)

    args = build_args_namespace(
        group_cols=group_cols,
        n_clusters=n_clusters,
        keep_policy=keep_policy,
        min_keep=min_keep,
        keep_scale=keep_scale,
        keep_fraction=keep_fraction,
        min_mz=min_mz,
        max_mz=max_mz,
        bin_width=bin_width,
        rel_intensity_min=rel_intensity_min,
        min_clean_peaks=min_clean_peaks,
        min_cluster_size=min_cluster_size,
        random_state=random_state,
        kmeans_batch_size=kmeans_batch_size,
        kmeans_n_init=kmeans_n_init,
        kmeans_max_iter=kmeans_max_iter,
        reassignment_ratio=reassignment_ratio,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        pre_dispatch=pre_dispatch,
        inner_threads=inner_threads,
    )

    missing_cols = [col for col in ["peak_d", "glycan", *args.group_cols] if col not in combined.columns]
    if missing_cols:
        raise ValueError(f"combined is missing required columns: {missing_cols}")

    n_bins = int(math.floor((max_mz - min_mz) / bin_width)) + 1
    grouped = combined.groupby(args.group_cols, sort=False, dropna=False)

    if args.n_jobs == 1:
        iterator = (
            tqdm(
                enumerate(grouped),
                total=grouped.ngroups,
                desc="Clustering glycan groups",
                unit="groups",
            )
            if verbose
            else enumerate(grouped)
        )
        results = [
            process_dataframe_group(
                group_order,
                args.group_cols,
                group_value,
                group_df,
                n_bins,
                args,
            )
            for group_order, (group_value, group_df) in iterator
        ]
    else:
        iterator = (
            tqdm(
                enumerate(grouped),
                total=grouped.ngroups,
                desc="Submitting glycan groups",
                unit="groups",
            )
            if verbose
            else enumerate(grouped)
        )
        results = Parallel(
            n_jobs=args.n_jobs,
            backend=args.parallel_backend,
            pre_dispatch=args.pre_dispatch,
        )(
            delayed(process_dataframe_group)(
                group_order,
                args.group_cols,
                group_value,
                group_df,
                n_bins,
                args,
            )
            for group_order, (group_value, group_df) in iterator
        )

    selected_indices, selected_report, summary_report, indexed, skipped = merge_group_results(results)

    downsampled = combined.loc[combined.index.isin(selected_indices)].copy()

    if output_path is not None:
        downsampled.to_csv(output_path, index=False)

    if report_prefix is not None:
        report_prefix = Path(report_prefix)
        write_tsv(
            report_prefix.with_suffix(".selected_rows.tsv"),
            selected_report,
            [
                "row_id",
                "group_key",
                "glycan",
                "cluster_id",
                "cluster_size",
                "rank_in_cluster",
                "distance_to_center",
                "similarity_to_center",
                "n_peaks",
                "retained_fraction",
                "quality",
            ],
        )
        write_tsv(
            report_prefix.with_suffix(".group_summary.tsv"),
            summary_report,
            [
                "group_key",
                "group_size",
                "indexed_count",
                "skipped_count",
                "n_clusters_requested",
                "n_clusters_used",
                "selected_count",
                "selected_per_cluster",
            ],
        )

    if verbose:
        print(f"Indexed spectra: {indexed}")
        print(f"Skipped spectra during denoising/indexing: {skipped}")
        print(f"Selected spectra: {len(downsampled)}")

    if return_reports:
        return downsampled, selected_report, summary_report

    return downsampled


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("combined.csv"))
    parser.add_argument("--output", type=Path, default=Path("combined_representative_kmeans.csv"))
    parser.add_argument("--group-cols", nargs="+", default=["glycan"])
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--keep-policy", choices=["sqrt", "log2", "fraction"], default="sqrt")
    parser.add_argument("--min-keep", type=int, default=1)
    parser.add_argument("--keep-scale", type=float, default=2.0)
    parser.add_argument("--keep-fraction", type=float, default=0.1)
    parser.add_argument("--min-mz", type=float, default=39.714)
    parser.add_argument("--max-mz", type=float, default=3000.0)
    parser.add_argument("--bin-width", type=float, default=1.0)
    parser.add_argument("--rel-intensity-min", type=float, default=0.005)
    parser.add_argument("--min-clean-peaks", type=int, default=5)
    parser.add_argument("--min-cluster-size", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    parser.add_argument("--kmeans-max-iter", type=int, default=100)
    parser.add_argument("--reassignment-ratio", type=float, default=0.01)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["loky", "threading"], default="loky")
    parser.add_argument("--pre-dispatch", default="2*n_jobs")
    parser.add_argument("--inner-threads", type=int, default=1)
    parser.add_argument("--tmp-db", type=Path, default=None)
    parser.add_argument("--keep-db", action="store_true")
    parser.add_argument("--sqlite-commit-every", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validated = build_args_namespace(
        group_cols=args.group_cols,
        n_clusters=args.n_clusters,
        keep_policy=args.keep_policy,
        min_keep=args.min_keep,
        keep_scale=args.keep_scale,
        keep_fraction=args.keep_fraction,
        min_mz=args.min_mz,
        max_mz=args.max_mz,
        bin_width=args.bin_width,
        rel_intensity_min=args.rel_intensity_min,
        min_clean_peaks=args.min_clean_peaks,
        min_cluster_size=args.min_cluster_size,
        random_state=args.random_state,
        kmeans_batch_size=args.kmeans_batch_size,
        kmeans_n_init=args.kmeans_n_init,
        kmeans_max_iter=args.kmeans_max_iter,
        reassignment_ratio=args.reassignment_ratio,
        n_jobs=args.n_jobs,
        parallel_backend=args.parallel_backend,
        pre_dispatch=args.pre_dispatch,
        inner_threads=args.inner_threads,
    )
    validated.sqlite_commit_every = args.sqlite_commit_every

    db_path = args.tmp_db
    if db_path is None:
        db_path = Path(tempfile.gettempdir()) / f"{args.output.stem}.spectrum_index.sqlite"

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Grouping columns: {validated.group_cols}")
    print(f"K clusters per group: {validated.n_clusters}")
    print(f"Keep policy per cluster: {validated.keep_policy}")
    print(f"Group parallelism: n_jobs={validated.n_jobs}, backend={validated.parallel_backend}")
    print(f"Worker math threads: inner_threads={validated.inner_threads}")
    print(f"Temporary SQLite index: {db_path}")

    conn, inserted, skipped, n_bins = index_spectra(args.input, db_path, validated)
    if inserted == 0 or n_bins is None:
        raise RuntimeError("No valid spectra were indexed. Check peak_d and thresholds.")

    print(f"Indexed spectra: {inserted}")
    print(f"Skipped spectra during denoising/indexing: {skipped}")
    print(f"m/z similarity bins: {n_bins}")

    selected_row_ids, selected_report, summary_report = process_groups(conn, n_bins, validated)
    written = write_selected_csv(args.input, args.output, selected_row_ids)

    selected_tsv = args.output.with_suffix(".selected_rows.tsv")
    summary_tsv = args.output.with_suffix(".group_summary.tsv")
    write_tsv(
        selected_tsv,
        selected_report,
        [
            "row_id",
            "group_key",
            "glycan",
            "cluster_id",
            "cluster_size",
            "rank_in_cluster",
            "distance_to_center",
            "similarity_to_center",
            "n_peaks",
            "retained_fraction",
            "quality",
        ],
    )
    write_tsv(
        summary_tsv,
        summary_report,
        [
            "group_key",
            "group_size",
            "indexed_count",
            "skipped_count",
            "n_clusters_requested",
            "n_clusters_used",
            "selected_count",
            "selected_per_cluster",
        ],
    )

    conn.close()
    if not args.keep_db and db_path.exists():
        db_path.unlink()

    print(f"Selected spectra written: {written}")
    print(f"Selected-row report: {selected_tsv}")
    print(f"Group summary report: {summary_tsv}")


if __name__ == "__main__":
    main()
