#!/usr/bin/env python3
"""
Downsample combined.csv to representative, low-noise MS2 spectra.

The code intentionally reuses the parsing/normalization style from
training/build_training_pickles_2.py, but does not import that module because
it has top-level code that rebuilds the full training dataset on import.

Default selection:
  keep N spectra per group, where N = ceil(sqrt(group_size)).

Example:
  python downsample_representative_spectra.py \
    --input /Users/xatava/CandyCrunch/training/combined.csv \
    --output /Users/xatava/CandyCrunch/training/combined_representative.csv \
    --group-cols glycan mode lc modification trap precursor_charge \
    --keep-policy sqrt
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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

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
class Selection:
    entry: Entry
    rank: int
    consensus_score: float
    clean_consensus_score: float


def keep_count_for_group(group_size: int, args) -> int:
    """Return dynamic N based only on group size, with no max_keep cap."""
    if group_size <= 0:
        return 0

    if args.keep_policy == "sqrt":
        n = math.ceil(args.keep_scale * math.sqrt(group_size))
    elif args.keep_policy == "log2":
        n = math.ceil(args.keep_scale * math.log2(group_size + 1))
    elif args.keep_policy == "fraction":
        n = math.ceil(args.keep_fraction * group_size)
    else:
        raise ValueError(f"Unknown keep policy: {args.keep_policy}")

    n = max(args.min_keep, n)
    n = min(n, group_size)
    return int(n)


def make_group_key(row: dict, group_cols: list[str]) -> str:
    payload = {col: str(row.get(col, "")).strip() for col in group_cols}
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


def sparse_dot(a_idx: np.ndarray, a_val: np.ndarray, b_idx: np.ndarray, b_val: np.ndarray) -> float:
    i = 0
    j = 0
    total = 0.0

    while i < len(a_idx) and j < len(b_idx):
        if a_idx[i] == b_idx[j]:
            total += float(a_val[i]) * float(b_val[j])
            i += 1
            j += 1
        elif a_idx[i] < b_idx[j]:
            i += 1
        else:
            j += 1

    return total


def consensus_vector(entries: list[Entry], n_bins: int):
    dense = np.zeros(n_bins, dtype=np.float32)
    for entry in entries:
        dense[entry.idx] += entry.val

    norm = float(np.linalg.norm(dense))
    if norm <= 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    dense /= norm
    idx = np.flatnonzero(dense).astype(np.int32)
    val = dense[idx].astype(np.float32)
    return idx, val


def robust_low_threshold(scores: np.ndarray, floor: float, mad_multiplier: float):
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    robust_sigma = 1.4826 * mad

    if robust_sigma <= 1e-8:
        return floor, median, mad

    return max(floor, median - mad_multiplier * robust_sigma), median, mad


def select_representatives(entries: list[Entry], n_bins: int, args):
    group_size = len(entries)
    keep_n = keep_count_for_group(group_size, args)
    if group_size == 0 or keep_n == 0:
        return [], {
            "group_size": group_size,
            "keep_n": keep_n,
            "clean_count": 0,
            "threshold": np.nan,
            "median_consensus": np.nan,
            "mad_consensus": np.nan,
        }

    consensus_idx, consensus_val = consensus_vector(entries, n_bins)
    consensus_scores = np.asarray(
        [sparse_dot(e.idx, e.val, consensus_idx, consensus_val) for e in entries],
        dtype=np.float32,
    )

    threshold, median_score, mad_score = robust_low_threshold(
        consensus_scores,
        floor=args.min_consensus_similarity,
        mad_multiplier=args.mad_multiplier,
    )

    clean_positions = np.flatnonzero(consensus_scores >= threshold)
    if clean_positions.size == 0:
        clean_positions = np.array([int(np.argmax(consensus_scores))], dtype=np.int64)

    clean_entries = [entries[i] for i in clean_positions]
    clean_consensus_idx, clean_consensus_val = consensus_vector(clean_entries, n_bins)
    clean_scores = np.asarray(
        [
            sparse_dot(entry.idx, entry.val, clean_consensus_idx, clean_consensus_val)
            for entry in clean_entries
        ],
        dtype=np.float32,
    )

    original_score_by_row = {
        entries[i].row_id: float(consensus_scores[i]) for i in range(len(entries))
    }
    order = sorted(
        range(len(clean_entries)),
        key=lambda i: (float(clean_scores[i]), clean_entries[i].quality),
        reverse=True,
    )

    selected_positions = []
    selected_set = set()

    for pos in order:
        candidate = clean_entries[pos]
        redundant = any(
            sparse_dot(candidate.idx, candidate.val, clean_entries[old].idx, clean_entries[old].val)
            >= args.diversity_similarity
            for old in selected_positions
        )
        if redundant:
            continue

        selected_positions.append(pos)
        selected_set.add(pos)
        if len(selected_positions) >= keep_n:
            break

    # Fill to top N from clean spectra if diversity filtering skipped too many.
    for pos in order:
        if len(selected_positions) >= keep_n:
            break
        if pos not in selected_set:
            selected_positions.append(pos)
            selected_set.add(pos)

    selections = []
    for rank, pos in enumerate(selected_positions, start=1):
        entry = clean_entries[pos]
        selections.append(
            Selection(
                entry=entry,
                rank=rank,
                consensus_score=original_score_by_row[entry.row_id],
                clean_consensus_score=float(clean_scores[pos]),
            )
        )

    return selections, {
        "group_size": group_size,
        "keep_n": keep_n,
        "clean_count": len(clean_entries),
        "threshold": float(threshold),
        "median_consensus": float(median_score),
        "mad_consensus": float(mad_score),
    }


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

    selected_row_ids = set()
    selected_report = []
    summary_report = []

    for group_key, group_size in tqdm(group_rows, desc="Selecting representatives", unit="groups"):
        entries = load_group_entries(conn, group_key)
        selected, stats = select_representatives(entries, n_bins, args)

        for sel in selected:
            selected_row_ids.add(sel.entry.row_id)
            selected_report.append(
                {
                    "row_id": sel.entry.row_id,
                    "rank": sel.rank,
                    "group_key": group_key,
                    "glycan": sel.entry.glycan,
                    "consensus_score": f"{sel.consensus_score:.6f}",
                    "clean_consensus_score": f"{sel.clean_consensus_score:.6f}",
                    "n_peaks": sel.entry.n_peaks,
                    "retained_fraction": f"{sel.entry.retained_fraction:.6f}",
                    "quality": f"{sel.entry.quality:.6f}",
                }
            )

        summary_report.append(
            {
                "group_key": group_key,
                "group_size": group_size,
                "keep_n": stats["keep_n"],
                "clean_count": stats["clean_count"],
                "selected_count": len(selected),
                "threshold": stats["threshold"],
                "median_consensus": stats["median_consensus"],
                "mad_consensus": stats["mad_consensus"],
            }
        )

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


def downsample_representative_spectra(
    combined=None,
    input_path: Union[str, Path] = "combined.csv",
    output_path: Optional[Union[str, Path]] = None,
    group_cols: Union[list[str], tuple[str, ...]] = ("glycan",),
    keep_policy: str = "sqrt",
    min_keep: int = 1,
    keep_scale: float = 2.0,
    keep_fraction: float = 0.1,
    min_mz: float = 39.714,
    max_mz: float = 3000.0,
    bin_width: float = 1.0,
    rel_intensity_min: float = 0.005,
    min_clean_peaks: int = 5,
    min_consensus_similarity: float = 0.25,
    mad_multiplier: float = 2.5,
    diversity_similarity: float = 0.90,
    report_prefix: Optional[Union[str, Path]] = None,
    return_reports: bool = False,
    verbose: bool = True,
):
    """
    Return a downsampled combined dataframe with representative spectra.

    Parameters mirror the CLI arguments. If combined is None, input_path is read
    with pandas. If output_path is provided, the downsampled dataframe is also
    written to CSV.

    Recommended in build_training_pickles_2.py:
        combined = downsample_representative_spectra(
            combined,
            group_cols=["glycan", "mode", "lc", "modification", "trap", "precursor_charge"],
        )
    """
    if keep_policy not in {"sqrt", "log2", "fraction"}:
        raise ValueError("keep_policy must be one of: sqrt, log2, fraction")
    if min_keep < 1:
        raise ValueError("min_keep must be >= 1")
    if keep_scale <= 0:
        raise ValueError("keep_scale must be > 0")
    if keep_fraction <= 0:
        raise ValueError("keep_fraction must be > 0")

    if combined is None:
        import pandas as pd

        combined = pd.read_csv(input_path)

    group_cols = list(group_cols)
    missing_cols = [col for col in ["peak_d", "glycan", *group_cols] if col not in combined.columns]
    if missing_cols:
        raise ValueError(f"combined is missing required columns: {missing_cols}")

    args = argparse.Namespace(
        keep_policy=keep_policy,
        min_keep=min_keep,
        keep_scale=keep_scale,
        keep_fraction=keep_fraction,
        min_mz=min_mz,
        max_mz=max_mz,
        bin_width=bin_width,
        rel_intensity_min=rel_intensity_min,
        min_clean_peaks=min_clean_peaks,
        min_consensus_similarity=min_consensus_similarity,
        mad_multiplier=mad_multiplier,
        diversity_similarity=diversity_similarity,
        group_cols=group_cols,
    )

    n_bins = int(math.floor((max_mz - min_mz) / bin_width)) + 1
    selected_indices = set()
    selected_report = []
    summary_report = []
    indexed = 0
    skipped = 0

    grouped = combined.groupby(group_cols, sort=False, dropna=False)
    iterator = tqdm(grouped, desc="Selecting representatives", unit="groups") if verbose else grouped

    for group_value, group_df in iterator:
        entries = []
        if not isinstance(group_value, tuple):
            group_value = (group_value,)
        group_key = json.dumps(
            dict(zip(group_cols, [str(v) for v in group_value])),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

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
                min_mz=min_mz,
                max_mz=max_mz,
                bin_width=bin_width,
                rel_intensity_min=rel_intensity_min,
                min_clean_peaks=min_clean_peaks,
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
            indexed += 1

        selected, stats = select_representatives(entries, n_bins, args)
        for sel in selected:
            selected_indices.add(sel.entry.row_id)
            selected_report.append(
                {
                    "row_id": sel.entry.row_id,
                    "rank": sel.rank,
                    "group_key": group_key,
                    "glycan": sel.entry.glycan,
                    "consensus_score": f"{sel.consensus_score:.6f}",
                    "clean_consensus_score": f"{sel.clean_consensus_score:.6f}",
                    "n_peaks": sel.entry.n_peaks,
                    "retained_fraction": f"{sel.entry.retained_fraction:.6f}",
                    "quality": f"{sel.entry.quality:.6f}",
                }
            )

        summary_report.append(
            {
                "group_key": group_key,
                "group_size": len(group_df),
                "indexed_count": len(entries),
                "keep_n": stats["keep_n"],
                "clean_count": stats["clean_count"],
                "selected_count": len(selected),
                "threshold": stats["threshold"],
                "median_consensus": stats["median_consensus"],
                "mad_consensus": stats["mad_consensus"],
            }
        )

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
                "rank",
                "group_key",
                "glycan",
                "consensus_score",
                "clean_consensus_score",
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
                "keep_n",
                "clean_count",
                "selected_count",
                "threshold",
                "median_consensus",
                "mad_consensus",
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
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/Users/xatava/CandyCrunch/training/combined.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/xatava/CandyCrunch/training/combined_representative.csv"),
    )

    parser.add_argument("--group-cols", nargs="+", default=["glycan"])
    parser.add_argument("--keep-policy", choices=["sqrt", "log2", "fraction"], default="sqrt")
    parser.add_argument("--min-keep", type=int, default=1)
    parser.add_argument("--keep-scale", type=float, default=2.0)
    parser.add_argument("--keep-fraction", type=float, default=0.1)

    parser.add_argument("--min-mz", type=float, default=39.714)
    parser.add_argument("--max-mz", type=float, default=3000.0)
    parser.add_argument("--bin-width", type=float, default=1.0)
    parser.add_argument("--rel-intensity-min", type=float, default=0.005)
    parser.add_argument("--min-clean-peaks", type=int, default=5)

    parser.add_argument("--min-consensus-similarity", type=float, default=0.25)
    parser.add_argument("--mad-multiplier", type=float, default=2.5)
    parser.add_argument("--diversity-similarity", type=float, default=0.90)

    parser.add_argument("--tmp-db", type=Path, default=None)
    parser.add_argument("--keep-db", action="store_true")
    parser.add_argument("--sqlite-commit-every", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_keep < 1:
        raise ValueError("--min-keep must be >= 1")
    if args.keep_fraction <= 0:
        raise ValueError("--keep-fraction must be > 0")
    if args.keep_scale <= 0:
        raise ValueError("--keep-scale must be > 0")

    db_path = args.tmp_db
    if db_path is None:
        db_path = Path(tempfile.gettempdir()) / f"{args.output.stem}.spectrum_index.sqlite"

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Grouping columns: {args.group_cols}")
    print(f"Keep policy: {args.keep_policy}")
    print("No max_keep cap is used.")
    print(f"Temporary SQLite index: {db_path}")

    conn, inserted, skipped, n_bins = index_spectra(args.input, db_path, args)
    if inserted == 0 or n_bins is None:
        raise RuntimeError("No valid spectra were indexed. Check peak_d and thresholds.")

    print(f"Indexed spectra: {inserted}")
    print(f"Skipped spectra during denoising/indexing: {skipped}")
    print(f"m/z similarity bins: {n_bins}")

    selected_row_ids, selected_report, summary_report = process_groups(conn, n_bins, args)
    written = write_selected_csv(args.input, args.output, selected_row_ids)

    selected_tsv = args.output.with_suffix(".selected_rows.tsv")
    summary_tsv = args.output.with_suffix(".group_summary.tsv")

    write_tsv(
        selected_tsv,
        selected_report,
        [
            "row_id",
            "rank",
            "group_key",
            "glycan",
            "consensus_score",
            "clean_consensus_score",
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
            "keep_n",
            "clean_count",
            "selected_count",
            "threshold",
            "median_consensus",
            "mad_consensus",
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
