#!/usr/bin/env python3
"""
Silhouette-based K-means downsampling for representative, low-noise MS2 spectra.

V3 keeps the same denoising/vectorization logic as V2, but chooses the number
of clusters K separately for each glycan group using the silhouette coefficient.

Recommended use inside build_training_pickles_2.py:

    from downsample_representative_spectra_V3 import downsample_representative_spectra_V3

    combined = downsample_representative_spectra_V3(
        combined,
        group_cols=("glycan",),
        k_min=2,
        k_max=10,
        silhouette_sample_size=1000,
        keep_policy="sqrt",
        keep_scale=2.0,
        n_jobs=24,
        parallel_backend="loky",
        inner_threads=1,
        report_prefix=output_dir / "representative_kmeans_silhouette",
        verbose=True,
    )
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from downsample_representative_spectra_V2 import (
    ClusterSelection,
    Entry,
    denoise_and_vectorize,
    distances_to_center,
    entries_to_csr,
    keep_count_for_cluster,
    load_group_entries,
    make_group_key,
    make_group_key_from_values,
    raise_csv_field_limit,
    thread_limit_context,
    write_selected_csv,
    write_tsv,
    _safe_peak_parse,
)

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


def build_args_namespace_v3(
    group_cols,
    k_min,
    k_max,
    silhouette_sample_size,
    silhouette_metric,
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
    if k_min < 2:
        raise ValueError("k_min must be >= 2 for silhouette scoring")
    if k_max < k_min:
        raise ValueError("k_max must be >= k_min")
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
    if silhouette_sample_size is not None and silhouette_sample_size < 0:
        raise ValueError("silhouette_sample_size must be >= 0, or None")

    return argparse.Namespace(
        group_cols=list(group_cols),
        k_min=int(k_min),
        k_max=int(k_max),
        silhouette_sample_size=None
        if silhouette_sample_size in (None, 0)
        else int(silhouette_sample_size),
        silhouette_metric=silhouette_metric,
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


def fit_kmeans(X, n_clusters: int, args):
    with thread_limit_context(args.inner_threads):
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=args.random_state,
            batch_size=args.kmeans_batch_size,
            n_init=args.kmeans_n_init,
            max_iter=args.kmeans_max_iter,
            reassignment_ratio=args.reassignment_ratio,
        )
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_.astype(np.float32)
    return labels, centers


def silhouette_sample_size_for_group(group_size: int, args):
    if args.silhouette_sample_size is None:
        return None
    return min(args.silhouette_sample_size, group_size)


def silhouette_indices_for_labels(labels, sample_size, random_state):
    labels = np.asarray(labels)
    n_samples = labels.shape[0]
    unique_labels = np.unique(labels)

    if unique_labels.size < 2 or unique_labels.size >= n_samples:
        return None

    if sample_size is None or sample_size >= n_samples:
        return np.arange(n_samples, dtype=np.int64)

    sample_size = max(int(sample_size), int(unique_labels.size) + 1)
    sample_size = min(sample_size, n_samples)
    rng = np.random.default_rng(random_state)

    selected = []
    selected_set = set()

    # Guarantee that every non-empty cluster is represented in the silhouette sample.
    for label in unique_labels:
        positions = np.flatnonzero(labels == label)
        chosen = int(rng.choice(positions))
        selected.append(chosen)
        selected_set.add(chosen)

    remaining_slots = sample_size - len(selected)
    if remaining_slots > 0:
        remaining = np.asarray(
            [idx for idx in range(n_samples) if idx not in selected_set],
            dtype=np.int64,
        )
        if remaining_slots >= remaining.size:
            selected.extend(remaining.tolist())
        elif remaining.size:
            selected.extend(
                rng.choice(remaining, size=remaining_slots, replace=False).tolist()
            )

    selected = np.asarray(sorted(set(selected)), dtype=np.int64)
    sampled_labels = np.unique(labels[selected])

    if sampled_labels.size < 2 or sampled_labels.size >= selected.size:
        return None

    return selected


def choose_k_by_silhouette(X, args):
    group_size = X.shape[0]

    # silhouette_score is undefined when every sample is its own cluster.
    candidate_max = min(args.k_max, group_size - 1)
    candidate_min = min(args.k_min, candidate_max)

    if group_size < 3 or candidate_max < 2:
        center = np.asarray(X.mean(axis=0), dtype=np.float32).reshape(1, -1)
        labels = np.zeros(group_size, dtype=np.int32)
        return labels, center, {
            "optimal_k": 1,
            "best_silhouette": np.nan,
            "candidate_scores": "{}",
            "selection_reason": "group_too_small",
        }

    best_score = -np.inf
    best_k = None
    best_labels = None
    best_centers = None
    candidate_scores = {}

    for k in range(candidate_min, candidate_max + 1):
        labels, centers = fit_kmeans(X, k, args)
        unique_labels = np.unique(labels)

        if unique_labels.size < 2 or unique_labels.size >= group_size:
            candidate_scores[str(k)] = None
            continue

        sample_size = silhouette_sample_size_for_group(group_size, args)
        sample_indices = silhouette_indices_for_labels(
            labels,
            sample_size=sample_size,
            random_state=args.random_state + k,
        )

        if sample_indices is None:
            candidate_scores[str(k)] = None
            continue

        with thread_limit_context(args.inner_threads):
            try:
                score = silhouette_score(
                    X[sample_indices],
                    labels[sample_indices],
                    metric=args.silhouette_metric,
                    sample_size=None,
                )
            except ValueError:
                candidate_scores[str(k)] = None
                continue

        score = float(score)
        candidate_scores[str(k)] = score

        # Keep the smaller K on exact ties for simpler, deterministic clusters.
        if score > best_score + 1e-12:
            best_score = score
            best_k = k
            best_labels = labels
            best_centers = centers

    if best_k is None:
        center = np.asarray(X.mean(axis=0), dtype=np.float32).reshape(1, -1)
        labels = np.zeros(group_size, dtype=np.int32)
        return labels, center, {
            "optimal_k": 1,
            "best_silhouette": np.nan,
            "candidate_scores": json.dumps(candidate_scores, sort_keys=True),
            "selection_reason": "no_valid_silhouette_candidate",
        }

    return best_labels, best_centers, {
        "optimal_k": int(best_k),
        "best_silhouette": float(best_score),
        "candidate_scores": json.dumps(candidate_scores, sort_keys=True),
        "selection_reason": "max_silhouette",
    }


def select_representatives_silhouette(entries: list[Entry], n_bins: int, args):
    if not entries:
        return [], {
            "group_size": 0,
            "optimal_k": 0,
            "best_silhouette": np.nan,
            "candidate_scores": "{}",
            "selection_reason": "empty_group",
            "selected_count": 0,
        }

    X = entries_to_csr(entries, n_bins)
    labels, centers, k_stats = choose_k_by_silhouette(X, args)
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
        "optimal_k": k_stats["optimal_k"],
        "best_silhouette": k_stats["best_silhouette"],
        "candidate_scores": k_stats["candidate_scores"],
        "selection_reason": k_stats["selection_reason"],
        "selected_count": len(selected),
    }


def build_selection_result_v3(
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
        "k_min": stats.get("k_min"),
        "k_max": stats.get("k_max"),
        "optimal_k": stats["optimal_k"],
        "best_silhouette": stats["best_silhouette"],
        "selection_reason": stats["selection_reason"],
        "selected_count": stats["selected_count"],
        "selected_per_cluster": json.dumps(cluster_counts, sort_keys=True),
        "candidate_scores": stats["candidate_scores"],
    }

    return {
        "group_order": group_order,
        "selected_row_ids": selected_row_ids,
        "selected_report": selected_report,
        "summary_report": summary_report,
        "indexed_count": indexed_count,
        "skipped_count": skipped_count,
    }


def process_dataframe_group_v3(group_order, group_cols, group_value, group_df, n_bins: int, args):
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

        selected, stats = select_representatives_silhouette(entries, n_bins, args)

    stats["k_min"] = args.k_min
    stats["k_max"] = min(args.k_max, max(len(entries) - 1, 1))

    return build_selection_result_v3(
        group_order=group_order,
        group_key=group_key,
        group_size=len(group_df),
        indexed_count=len(entries),
        skipped_count=skipped,
        selected=selected,
        stats=stats,
    )


def process_sqlite_group_v3(db_path: str, group_order: int, group_key: str, indexed_count: int, n_bins: int, args):
    with thread_limit_context(args.inner_threads):
        conn = sqlite3.connect(db_path)
        try:
            entries = load_group_entries(conn, group_key)
        finally:
            conn.close()

        selected, stats = select_representatives_silhouette(entries, n_bins, args)

    stats["k_min"] = args.k_min
    stats["k_max"] = min(args.k_max, max(len(entries) - 1, 1))

    return build_selection_result_v3(
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
            desc="Clustering groups with silhouette K",
            unit="groups",
        )
        for group_order, (group_key, indexed_count) in iterator:
            entries = load_group_entries(conn, group_key)
            selected, stats = select_representatives_silhouette(entries, n_bins, args)
            stats["k_min"] = args.k_min
            stats["k_max"] = min(args.k_max, max(len(entries) - 1, 1))
            results.append(
                build_selection_result_v3(
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
            delayed(process_sqlite_group_v3)(
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


def downsample_representative_spectra_V3(
    combined=None,
    input_path: Union[str, Path] = "combined.csv",
    output_path: Optional[Union[str, Path]] = None,
    group_cols: Union[list[str], tuple[str, ...]] = ("glycan",),
    k_min: int = 2,
    k_max: int = 10,
    silhouette_sample_size: Optional[int] = 1000,
    silhouette_metric: str = "euclidean",
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
    Return a dataframe containing silhouette-KMeans representative spectra.

    K is chosen separately per group by trying k_min..k_max and selecting the
    K with the highest silhouette coefficient. For large groups, silhouette is
    computed on a deterministic sample of size silhouette_sample_size.
    """
    if combined is None:
        import pandas as pd

        combined = pd.read_csv(input_path)

    args = build_args_namespace_v3(
        group_cols=group_cols,
        k_min=k_min,
        k_max=k_max,
        silhouette_sample_size=silhouette_sample_size,
        silhouette_metric=silhouette_metric,
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
                desc="Clustering glycan groups with silhouette K",
                unit="groups",
            )
            if verbose
            else enumerate(grouped)
        )
        results = [
            process_dataframe_group_v3(
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
            delayed(process_dataframe_group_v3)(
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
                "k_min",
                "k_max",
                "optimal_k",
                "best_silhouette",
                "selection_reason",
                "selected_count",
                "selected_per_cluster",
                "candidate_scores",
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
    parser.add_argument("--output", type=Path, default=Path("combined_representative_silhouette.csv"))
    parser.add_argument("--group-cols", nargs="+", default=["glycan"])
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--silhouette-sample-size", type=int, default=1000)
    parser.add_argument("--silhouette-metric", default="euclidean")
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
    validated = build_args_namespace_v3(
        group_cols=args.group_cols,
        k_min=args.k_min,
        k_max=args.k_max,
        silhouette_sample_size=args.silhouette_sample_size,
        silhouette_metric=args.silhouette_metric,
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
    print(f"Silhouette K range: {validated.k_min}..{validated.k_max}")
    print(f"Silhouette sample size: {validated.silhouette_sample_size}")
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
            "k_min",
            "k_max",
            "optimal_k",
            "best_silhouette",
            "selection_reason",
            "selected_count",
            "selected_per_cluster",
            "candidate_scores",
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
