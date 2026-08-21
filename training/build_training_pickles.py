"""
Helper script to convert a condensed spectra table (e.g. full_dataset.xlsx) and
its metadata into CandyCrunch-ready training pickles.  Adjust the paths in the
configuration block before running.
"""

import ast
import pickle
from pathlib import Path
import torch
import os
from os.path import join
import tqdm
from sklearn.model_selection import train_test_split
import numpy_indexed as npi
import numpy as np
import sys
import numpy.core.numeric
# Create the alias that the pickle expects
np._core = np.core
np._core.numeric = np.core.numeric
# Register the module so pickle can find it
sys.modules['numpy._core.numeric'] = np.core.numeric
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from datasail.sail import datasail
from glycowork.motif.tokenization import get_stem_lib, glycan_to_composition
import hashlib
import ast
import gc
import argparse
from downsample_representative_spectra import downsample_representative_spectra
from downsample_representative_spectra_V2 import downsample_representative_spectra_V2
from downsample_representative_spectra_V3 import downsample_representative_spectra_V3
# from candycrunch.prediction import bin_intensities
full_dataset_path = Path("full_dataset_20260717.pkl")
metadata_path = Path("file_checklist_template.csv")
test_size = 0.20
random_state = 42

MODE_MAP = {"negative": 0, "positive": 1}
LC_MAP = {"PGC": 0, "C18": 1}
MOD_MAP = {"reduced": 0, "permethylated": 1}
TRAP_MAP = {"linear": 0, "orbitrap": 1, "amazon": 2}

FEATURE_COLUMNS = [
    "binned_intensities",
    "peak_list",
    "mz_remainder",
    "m/z",
    "glycan_type",
    "RT",
    "mode",
    "lc",
    "modification",
    "trap",
]


def bin_intensities(peak_d, frames):
    """sums up intensities for each bin across a spectrum\n
   | Arguments:
   | :-
   | peak_d (dict): dictionary of form (fragment) m/z : intensity
   | frames (list): m/z boundaries separating each bin\n
   | Returns:
   | :-
   | (1) a list of binned intensities
   | (2) a list of the difference (bin edge - m/z of highest peak in bin) for each bin
   """
    num_frames = len(frames)
    binned_intensities = np.zeros(num_frames)
    mz_diff = np.zeros(num_frames)
    mzs = np.array(list(peak_d.keys()), dtype='float32')
    intensities = np.array(list(peak_d.values()))
    bin_indices = np.digitize(mzs, frames, right=True)
    mz_remainder = mzs - frames[bin_indices - 1]
    max_intensities = npi.group_by(bin_indices - 1).max(intensities)
    mz_remainder = mz_remainder * np.isin(intensities, max_intensities)
    unique_bins, summed_intensities = npi.group_by(bin_indices).sum(intensities)
    _, max_mz_remainder = npi.group_by(bin_indices).max(mz_remainder)
    binned_intensities[unique_bins - 1] = summed_intensities
    mz_diff[unique_bins - 1] = max_mz_remainder
    return binned_intensities, mz_diff


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

def _spectrum_to_peak_list(spec, max_peaks=None, min_mz=None, max_mz=None):
    # peaks = [(float(mz), float(intensity)) for mz, intensity in spec.items() if min_mz <= float(mz) <= max_mz and float(intensity) > 0]
    peaks = []
    for mz, intensity in spec.items():
        if min_mz <= float(mz) <= max_mz and float(intensity) > 0:
            peaks.append((float(mz), float(intensity)))

    if not peaks:
        return None

    peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:max_peaks]

    total_intensity = sum(intensity for _, intensity in peaks)
    if total_intensity <= 0:
        return None
    peaks = sorted(peaks, key=lambda x: x[0])
    while len(peaks) < max_peaks:
        peaks.append((0.0, 0.0))

    return np.asarray(peaks, dtype=np.float32)

def _process_peaks(df, min_mz=39.714, max_mz=3000.0, bin_num=2048, max_peaks=2048):
    frames = np.linspace(min_mz, max_mz, bin_num)
    parsed = df["peak_d"].map(_safe_peak_parse)
    parsed = parsed.map(lambda spec: None if spec is None else _normalise_spectrum(spec))
    keep_mask = parsed.notnull()
    df = df.loc[keep_mask].copy()
    parsed = parsed.loc[keep_mask]
    binned, remainders = zip(*(bin_intensities(spec, frames) for spec in parsed))

    peak_lists = parsed.map(lambda spec: _spectrum_to_peak_list( spec, max_peaks=max_peaks, min_mz=min_mz, max_mz=max_mz,))
    keep_mask = peak_lists.notnull()
    df = df.loc[keep_mask].copy()
    binned = [vec for vec, keep in zip(binned, keep_mask) if keep]
    remainders = [vec for vec, keep in zip(remainders, keep_mask) if keep]
    peak_lists = peak_lists.loc[keep_mask]

    df["binned_intensities"] = [np.asarray(vec, dtype=np.float32) for vec in binned]
    df["mz_remainder"] = [np.asarray(vec, dtype=np.float32) for vec in remainders]
    df["peak_list"] = [np.asarray(vec, dtype=np.float32) for vec in peak_lists]

    return df

def _process_retention_times(df):
    df = df.copy()
    df["RT"] = df["RT"].fillna(15)
    df = df[df["RT"] > 2]
    df["RT"] = df.groupby("filename")["RT"].transform(lambda rt: rt / max(rt.max(), 30.0))
    return df

def _infer_glycan_types(df):
    def classify(g):
        if g.endswith(("GalNAc", "GalNAc6S", "GalNAcOS", "Fuc", "Man", "Gal")):
            return 0
        if "GlcNAc(b1-4)GlcNAc" in g:
            return 1
        if g.endswith(("Glc", "GlcOS", "GlcNAc", "Ins")):
            return 2
        return 3
    df = df.copy()
    df["glycan_type"] = df["glycan"].map(classify)
    return df

def _attach_metadata(df, checklist):
    meta = checklist.copy()
    meta.columns = [col.strip() for col in meta.columns]
    meta = meta.set_index("GlycoPOST_ID")
    meta.index = meta.index.astype(str).str.strip().str.lower()

    def build_dict(column):
        if column not in meta.columns:
            return {}
        return meta[column].fillna("").astype(str).str.lower().str.strip().to_dict()

    mode_dict = build_dict("mode")
    lc_dict = build_dict("LC_type")
    mod_dict = build_dict("modification")
    trap_dict = build_dict("trap")

    def lookup(mapping, value, fallback):
        if pd.isna(value) or str(value).strip() == "":
            return fallback
        return mapping.get(str(value).lower().strip(), fallback)

    def fill_from_checklist(ids, current_values, source_dict):
        ids = ids.fillna("").astype(str).str.strip().str.lower()
        current_values = current_values.copy()
        missing = current_values.isna() | (current_values.astype(str).str.strip() == "")
        checklist_values = ids.map(source_dict)
        current_values.loc[missing] = checklist_values.loc[missing]
        return current_values

    def map_metadata(column, source_dict, mapping, fallback):
        if column in df.columns:
            values = df[column]
        else:
            values = pd.Series(np.nan, index=df.index)

        values = fill_from_checklist(df["GlycoPost_ID"], values, source_dict)
        return values.map(lambda x: lookup(mapping, x, fallback)).astype(int)

    df = df.copy()

    df["mode"] = map_metadata("mode", mode_dict, MODE_MAP, 2)
    df["lc"] = map_metadata("lc_type", lc_dict, LC_MAP, 2)
    df["modification"] = map_metadata("modification", mod_dict, MOD_MAP, 2)
    df["trap"] = map_metadata("trap", trap_dict, TRAP_MAP, 3)
    return df

def process_full_dataset(full_df, checklist):
    df = _process_retention_times(full_df)
    df = _infer_glycan_types(df)
    df = _process_peaks(df)
    df = _attach_metadata(df, checklist)
    return df

def downcast_numeric(df):
    result = df.copy()
    int_cols = result.select_dtypes(include=["int", "uint", "int64", "uint64"]).columns
    float_cols = result.select_dtypes(include=["float", "float64"]).columns
    for col in int_cols:
        result[col] = pd.to_numeric(result[col], downcast="unsigned")
    for col in float_cols:
        result[col] = pd.to_numeric(result[col], downcast="float")
    return result

def tupleify(df, columns):
    return list(df[list(columns)].itertuples(index=False, name=None))


def gpu_cosine_similarity_unique(matrix, ids, output_file, batch_size = 3482, desc = "Computing similarity"):
    """
    Compute cosine similarity matrix for unique items and save to file
    """
    n_samples = matrix.shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Normalize the matrix for cosine similarity
    matrix_tensor = torch.from_numpy(matrix).float().to(device)
    norms = torch.norm(matrix_tensor, dim = 1, keepdim = True)
    matrix_tensor = matrix_tensor / (norms + 1e-8)

    # Initialize result matrix
    similarity_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)

    # Calculate total number of batches
    n_batches = (n_samples + batch_size - 1) // batch_size

    # Progress bar
    with tqdm.tqdm(total = n_batches, desc = desc, unit = 'batch') as pbar:
        for i in range(0, n_samples, batch_size):
            end_i = min(i + batch_size, n_samples)
            batch = matrix_tensor[i:end_i]
            similarity = torch.mm(batch, matrix_tensor.t())
            similarity_matrix[i:end_i, :] = similarity.cpu().numpy()
            pbar.update(1)

    # Create DataFrame with proper indices
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index = ids,
        columns = ids
    )

    # Save to TSV
    similarity_df.to_csv(output_file, sep = '\t')
    print(f"Saved similarity matrix to {output_file}")

    return similarity_df


def prepare_unique_embeddings(df, id_col, embedding_col):
    """
    Extract unique embeddings for each ID
    """
    unique_ids = df[id_col].unique()
    embeddings = []

    for uid in unique_ids:
        # Get the first occurrence of this ID
        embedding = df[df[id_col] == uid][embedding_col].iloc[0]
        # Ensure it's a numpy array and properly shaped
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.numpy()
        elif isinstance(embedding, (list, np.ndarray)):
            embedding = np.asarray(embedding).squeeze()
        else:
            embedding = np.asarray(embedding).squeeze()
        embeddings.append(embedding)

    embeddings = np.stack(embeddings)
    return unique_ids, embeddings


def add_embeddings_to_df(df, glycan_path, glycan_pattern,
                         lectin_col = None, glycan_col = None, max_files = 20):
    def load_embeddings(path, pattern, ids, max_files):
        embedding_dict = {pid: None for pid in ids}
        for i in range(max_files):
            file_path = join(path, pattern.format(i) if '{}' in pattern else pattern)
            if not os.path.exists(file_path):
                continue
            rep_dict = torch.load(file_path, weights_only = False)
            for pid in embedding_dict:
                if embedding_dict[pid] is None and pid in rep_dict:
                    embedding_dict[pid] = rep_dict[pid]
            if all(v is not None for v in embedding_dict.values()):
                break
        return embedding_dict

    glycan_dict = load_embeddings(glycan_path, glycan_pattern, df[glycan_col].unique(), max_files)
    df['GlycanGT_mean'] = df[glycan_col].map(glycan_dict)

    return df

def downcast_numeric(df):
    result = df.copy()
    int_cols = result.select_dtypes(include=["int", "uint", "int64", "uint64"]).columns
    float_cols = result.select_dtypes(include=["float", "float64"]).columns
    for col in int_cols:
        result[col] = pd.to_numeric(result[col], downcast="unsigned")
    for col in float_cols:
        result[col] = pd.to_numeric(result[col], downcast="float")
    return result


def two_split_report(train_set, test_set):
    nan_check_train_set = train_set.isnull().sum()
    empty_check_train_set = train_set.map(lambda x: isinstance(x, str) and x == '').sum()
    nan_check_test_set = test_set.isnull().sum()
    empty_check_test_set = test_set.map(lambda x: isinstance(x, str) and x == '').sum()

    result = pd.concat(
        [nan_check_train_set, empty_check_train_set, nan_check_test_set, empty_check_test_set], axis = 1)
    result.columns = ['NaNTrain', 'NullTrain', 'NaNTest', 'NullTest']

    test_to_data = round(len(test_set) / (len(test_set) + len(train_set)), 2)
    number_data = len(train_set) + len(test_set)

    return result, number_data, test_to_data


# Create a mapping from glycan to its frequency category
def get_freq_category(glycan, glycan_counts):
    count = glycan_counts[glycan]
    if count <= 10:
        return 'rare'
    elif count <= 100:
        return 'medium'
    else:
        return 'frequent'


def random_fallback_split(group_df, test_size=0.20, random_state=42):
    if len(group_df) < 2:
        return group_df.copy(), group_df.iloc[0:0].copy()

    train_idx, test_idx = train_test_split(
        group_df.index,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    return (
        group_df.loc[train_idx].copy(),
        group_df.loc[test_idx].copy(),
    )

def split_one_glycan_group_with_datasail(
    group_df,
    glycan_name,
    test_size=0.20,
    random_state=42,
    output_dir=None,
    batch_size=3482,
):
    group_df = group_df.reset_index(drop=False).rename(columns={"index": "_combined_index"})

    if len(group_df) <= 2:
        group_df["split"] = "train"
        return group_df[group_df["split"] == "train"], group_df.iloc[0:0]

    item_ids = np.array(
        [f"row_{idx}" for idx in group_df["_combined_index"].tolist()],
        dtype=str,
    )

    embeddings = np.stack(
        group_df["binned_intensities"]
        .map(lambda x: np.asarray(x, dtype=np.float32).squeeze())
        .to_numpy()
    )

    safe_glycan_id = hashlib.md5(str(glycan_name).encode("utf-8")).hexdigest()[:12]
    sim_file = output_dir / f"datasail_sim_glycan_{safe_glycan_id}.tsv"

    gpu_cosine_similarity_unique(
        embeddings,
        item_ids,
        sim_file,
        batch_size=batch_size,
        desc=f"Computing similarity for glycan {glycan_name}",
    )
    try:
        e_splits, f_splits, inter_sp = datasail(
            e_type="M",
            e_data=dict(zip(item_ids, embeddings)),
            e_sim=str(sim_file),
            names=["train", "test"],
            splits=[1.0 - test_size, test_size],
            techniques=["C1e"],
            solver="GUROBI",      # use "CBC" if GUROBI is not available
            epsilon=0.05,
            max_sec=600,
        )

        split_dict = e_splits["C1e"][0] if isinstance(e_splits["C1e"], list) else e_splits["C1e"]

        group_df["_datasail_id"] = item_ids
        group_df["split"] = group_df["_datasail_id"].map(split_dict)

        missing = group_df["split"].isna().sum()
        if missing:
            raise ValueError(f"{missing} rows did not receive a DataSAIL split for glycan {glycan_name}")

        train_part = group_df[group_df["split"] == "train"]
        test_part = group_df[group_df["split"] == "test"]

        return train_part, test_part
    except Exception as exc:
        last_error = exc
        print(
            f"DataSAIL failed for glycan {glycan_name}: {type(exc).__name__}: {exc}"
        )
        print(
            f"Falling back to random within-glycan split for {glycan_name}. "
            f"Last DataSAIL error: {type(last_error).__name__}: {last_error}"
        )
        return random_fallback_split(
            group_df,
            test_size = test_size,
            random_state = random_state,
        )
    finally:
        if sim_file.exists():
            sim_file.unlink()

def filter_data_exceptions_v1(data):
    from glycowork.motif.graph import glycan_to_nxGraph
    excluded_composition_keys = {"-H2O"}
    excluded_name_tokens = ("lactone", "4/6S")
    data = data.copy()
    bad_indices = []
    checked_glycans = {}
    for idx, glycan in data["glycan"].items():
        if glycan in checked_glycans:
            if not checked_glycans[glycan]:
                bad_indices.append(idx)
            continue
        try:
            glycan_str = str(glycan)
            glycan_lower = glycan_str.lower()
            if any(token.lower() in glycan_lower for token in excluded_name_tokens):
                checked_glycans[glycan] = False
                bad_indices.append(idx)
                print(f"Removing row {idx} with glycan {glycan!r}: excluded artifact/unsupported token")
                continue
            composition = glycan_to_composition(glycan)
            if set(composition) & excluded_composition_keys:
                checked_glycans[glycan] = False
                bad_indices.append(idx)
                print(f"Removing row {idx} with glycan {glycan!r}: contains -H2O artifact")
                continue
            glycan_to_nxGraph(glycan)
            checked_glycans[glycan] = True
        except Exception as e:
            checked_glycans[glycan] = False
            bad_indices.append(idx)
            print(f"Removing row {idx} with glycan {glycan!r}: {type(e).__name__}: {e}")
    if bad_indices:
        print(f"Removed {len(bad_indices)} rows with invalid/artifact glycans.")
        data = data.drop(index=bad_indices)
    return data


def peak_list_to_similarity_vector(
    peak_list,
    min_mz=39.714,
    max_mz=3000.0,
    intensity_weight=1.0,
):
    peaks = np.asarray(peak_list, dtype=np.float32).squeeze()

    if peaks.ndim == 1:
        if peaks.size % 2 != 0:
            raise ValueError(f"Cannot reshape peak_list with size {peaks.size} into pairs")
        peaks = peaks.reshape(-1, 2)

    if peaks.ndim != 2 or peaks.shape[1] != 2:
        raise ValueError(f"Expected peak_list shape (max_peaks, 2), got {peaks.shape}")

    mz = peaks[:, 0].copy()
    intensity = peaks[:, 1].copy()

    valid = mz > 0
    mz_scaled = np.zeros_like(mz, dtype=np.float32)
    mz_scaled[valid] = (mz[valid] - min_mz) / (max_mz - min_mz)
    mz_scaled = np.clip(mz_scaled, 0.0, 1.0)

    intensity = intensity * intensity_weight

    return np.column_stack([mz_scaled, intensity]).reshape(-1).astype(np.float32)


def split_one_glycan_group_with_datasail_max_peaks(
    group_df,
    glycan_name,
    test_size=0.20,
    random_state=42,
    output_dir=None,
    batch_size=3482,
    min_mz=39.714,
    max_mz=3000.0,
    intensity_weight=1.0,
):
    group_df = group_df.reset_index(drop=False).rename(columns={"index": "_combined_index"})

    if len(group_df) <= 2:
        group_df["split"] = "train"
        return group_df[group_df["split"] == "train"].copy(), group_df.iloc[0:0].copy()

    item_ids = np.array(
        [f"row_{idx}" for idx in group_df["_combined_index"].tolist()],
        dtype=str,
    )

    embeddings = np.stack(group_df["peak_list"] .map(  lambda x: peak_list_to_similarity_vector(x,min_mz=min_mz,
                max_mz=max_mz,intensity_weight=intensity_weight,)).to_numpy())

    safe_glycan_id = hashlib.md5(str(glycan_name).encode("utf-8")).hexdigest()[:12]
    sim_file = output_dir / f"datasail_peak_sim_glycan_{safe_glycan_id}.tsv"

    gpu_cosine_similarity_unique(embeddings,item_ids,sim_file,batch_size=batch_size,
                                 desc=f"Computing max-peak similarity for glycan {glycan_name}")

    try:
        e_splits, f_splits, inter_sp = datasail(
            e_type="M",
            e_data=dict(zip(item_ids, embeddings)),
            e_sim=str(sim_file),
            names=["train", "test"],
            splits=[1.0 - test_size, test_size],
            techniques=["C1e"],
            solver="GUROBI",
            epsilon=0.05,
            max_sec=600,
        )

        split_dict = e_splits["C1e"][0] if isinstance(e_splits["C1e"], list) else e_splits["C1e"]

        group_df["_datasail_id"] = item_ids
        group_df["split"] = group_df["_datasail_id"].map(split_dict)

        missing = group_df["split"].isna().sum()
        if missing:
            raise ValueError(f"{missing} rows did not receive a DataSAIL split for glycan {glycan_name}")

        train_part = group_df[group_df["split"] == "train"].copy()
        test_part = group_df[group_df["split"] == "test"].copy()

        return train_part, test_part

    except Exception as exc:
        print(
            f"DataSAIL failed for glycan {glycan_name}: {type(exc).__name__}: {exc}"
        )
        print(f"Falling back to random within-glycan split for {glycan_name}.")

        return random_fallback_split(
            group_df,
            test_size=test_size,
            random_state=random_state,
        )

    finally:
        if sim_file.exists():
            sim_file.unlink()

def main(args):
    print("Loading condensed spectra")
    full_df = pd.read_pickle(full_dataset_path)
    counts = []
    for x in full_df["peak_d"]:
        if pd.isna(x):
            counts.append(0)
        else:
            try:
                d = ast.literal_eval(x)
                counts.append(len(d) if isinstance(d, dict) else None)
            except Exception:
                counts.append(None)
    full_df["n_keys"] = counts
    print(full_df["n_keys"].describe())
    print(full_df["n_keys"].max())
    full_df.drop(columns = ["n_keys"], inplace = True)


    meta_df = pd.read_csv(metadata_path)
    processed = process_full_dataset(full_df, meta_df)
    frames = [processed]
    combined = pd.concat(frames, ignore_index=True)
    del full_df
    del meta_df
    del processed
    del frames
    del counts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    nan_count = combined["filename"].isna().sum()
    total_count = len(combined)
    print(f"Total rows: {total_count}")
    print(f"Rows with NaN in filename: {nan_count}")
    combined = combined.dropna(subset=["filename"])
    combined = filter_data_exceptions_v1(combined)
    combined.reset_index(drop=True, inplace=True)
    glycans = sorted(set(combined["glycan"]))
    glycan_dict = dict(zip(glycans, range(len(glycans))))

    combined["glycan_ids"] = "GID" + combined["glycan"].map(glycan_dict).astype(str)
    # combined.to_csv("combined.csv", index = False)
    with open("./glycans.pkl", "wb") as fh:
        pickle.dump(glycans, fh)

    if args.data_processing == "DS":
        policy_tag = {"sqrt": "S", "log2": "L", "fraction": "F"}.get(args.keep_policy)
        #output_dir = Path(f"prepared_datasets_DS{args.nclusters}{policy_tag}{args.keep_scale if policy_tag in ('S', 'L') else args.keep_fraction}")
        output_dir = Path(
            f"prepared_datasets_DS{policy_tag}{args.keep_scale if policy_tag in ('S', 'L') else args.keep_fraction}")
        output_dir.mkdir(parents = True, exist_ok = True)

        # combined = downsample_representative_spectra(
        #     combined,
        #     group_cols=["glycan", "mode", "lc", "modification", "trap", "precursor_charge"],
        #     keep_policy="sqrt",
        #     min_keep=1,
        #     keep_scale=7.0,
        #     rel_intensity_min=0.005,
        #     min_consensus_similarity=0.20,
        #     mad_multiplier=2,
        #     diversity_similarity=0.80,
        #     report_prefix=output_dir / "representative_downsample",
        #     verbose=True,
        # )

        # combined = downsample_representative_spectra_V2(
        #     combined,
        #     group_cols=("glycan",),
        #     n_clusters=args.nclusters,
        #     keep_policy=args.keep_policy,
        #     keep_scale=args.keep_scale,
        #     keep_fraction=args.keep_fraction,
        #     rel_intensity_min=args.rel_intensity_min,
        #     min_clean_peaks=args.min_clean_peaks,
        #     random_state=42,
        #     n_jobs=24,
        #     parallel_backend="loky",
        #     inner_threads=1,
        #     report_prefix=output_dir / "representative_kmeans",
        #     verbose=True,
        # )

        combined = downsample_representative_spectra_V3(
            combined,
            group_cols = ("glycan",),
            k_min = 2,
            k_max = 10,
            silhouette_sample_size = 1000,
            keep_policy = args.keep_policy,
            keep_scale = args.keep_scale,
            keep_fraction = args.keep_fraction,
            rel_intensity_min = args.rel_intensity_min,
            min_clean_peaks = args.min_clean_peaks,
            random_state = 42,
            n_jobs = 24,
            parallel_backend = "loky",
            inner_threads = 1,
            report_prefix = output_dir / "representative_kmeans_silhouette",
            verbose = True,
        )
        combined.reset_index(drop = True, inplace = True)
    else:
        output_dir = Path(f"prepared_datasets_{args.data_processing}")
        output_dir.mkdir(parents=True, exist_ok=True)

########################################################################################################################
    # Random Group-shuffle-split
########################################################################################################################
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
    train_idx, test_idx = next(splitter.split(combined, groups=combined["filename"]))
    train_df_GShS = combined.iloc[train_idx].reset_index(drop=True)
    test_df_GShS = combined.iloc[test_idx].reset_index(drop=True)

    print("Random Group-shuffle-split")
    train_df_GShS = downcast_numeric(train_df_GShS)
    test_df_GShS = downcast_numeric(test_df_GShS)
    X_train_GShS = tupleify(train_df_GShS, FEATURE_COLUMNS)
    X_test_GShS = tupleify(test_df_GShS, FEATURE_COLUMNS)
    y_train_GShS = train_df_GShS["glycan"].tolist()
    y_test_GShS = test_df_GShS["glycan"].tolist()

    with open(output_dir / "X_train_GShS.pkl", "wb") as fh:
        pickle.dump(X_train_GShS, fh)
    with open(output_dir / "X_test_GShS.pkl", "wb") as fh:
        pickle.dump(X_test_GShS, fh)
    with open(output_dir / "y_train_GShS.pkl", "wb") as fh:
        pickle.dump(y_train_GShS, fh)
    with open(output_dir / "y_test_GShS.pkl", "wb") as fh:
        pickle.dump(y_test_GShS, fh)

########################################################################################################################
    # Random Group-similarity-split
########################################################################################################################

    # print("Random Group-similarity-split with DataSAIL")
    # train_parts_CNN = []
    # test_parts_CNN = []
    # train_parts_TRN = []
    # test_parts_TRN = []
    # for old_file in output_dir.glob("datasail_sim_glycan_*.tsv"):
    #     old_file.unlink()
    #
    # for glycan_name, group_df in combined.groupby("glycan", sort=False):
    #     print(f"Splitting glycan {glycan_name}: {len(group_df)} rows")
    #
    #     train_part_CNN, test_part_CNN = split_one_glycan_group_with_datasail(
    #         group_df,
    #         glycan_name=glycan_name,
    #         test_size=test_size,
    #         random_state=random_state,
    #         output_dir=output_dir,
    #         batch_size=3482,
    #     )
    #
    #     train_part_TRN, test_part_TRN = split_one_glycan_group_with_datasail_max_peaks(
    #         group_df,
    #         glycan_name = glycan_name,
    #         test_size = test_size,
    #         random_state = random_state,
    #         output_dir = output_dir,
    #     )
    #
    #     train_parts_CNN.append(train_part_CNN)
    #     test_parts_CNN.append(test_part_CNN)
    #     train_parts_TRN.append(train_part_TRN)
    #     test_parts_TRN.append(test_part_TRN)
    #
    #
    # train_df_CGSiS = pd.concat(train_parts_CNN, ignore_index=True)
    # test_df_CGSiS = pd.concat(test_parts_CNN, ignore_index=True)
    #
    # train_df_TGSiS = pd.concat(train_parts_TRN, ignore_index=True)
    # test_df_TGSiS = pd.concat(test_parts_TRN, ignore_index=True)
    #
    # drop_temp_cols = ["_combined_index", "_datasail_id", "split"]
    # train_df_CGSiS = train_df_CGSiS.drop(columns=[c for c in drop_temp_cols if c in train_df_CGSiS.columns])
    # test_df_CGSiS = test_df_CGSiS.drop(columns=[c for c in drop_temp_cols if c in test_df_CGSiS.columns])
    #
    # train_df_TGSiS = train_df_TGSiS.drop(columns=[c for c in drop_temp_cols if c in train_df_TGSiS.columns])
    # test_df_TGSiS = test_df_TGSiS.drop(columns=[c for c in drop_temp_cols if c in test_df_TGSiS.columns])
    #
    # print(
    #     "Final DataSAIL glycan-wise split for CNN:",
    #     f"train rows={len(train_df_CGSiS)},",
    #     f"test rows={len(test_df_CGSiS)},",
    #     f"test fraction={len(test_df_CGSiS) / (len(train_df_CGSiS) + len(test_df_CGSiS)):.3f}",
    # )
    #
    # print(
    #     "Final DataSAIL glycan-wise split for Transformer:",
    #     f"train rows={len(train_df_TGSiS)},",
    #     f"test rows={len(test_df_TGSiS)},",
    #     f"test fraction={len(test_df_TGSiS) / (len(train_df_TGSiS) + len(test_df_TGSiS)):.3f}",
    # )
    #
    # train_df_CGSiS = downcast_numeric(train_df_CGSiS)
    # test_df_CGSiS = downcast_numeric(test_df_CGSiS)
    # X_train_CGSiS = tupleify(train_df_CGSiS, FEATURE_COLUMNS)
    # X_test_CGSiS = tupleify(test_df_CGSiS, FEATURE_COLUMNS)
    # y_train_CGSiS = train_df_CGSiS["glycan"].tolist()
    # y_test_CGSiS = test_df_CGSiS["glycan"].tolist()
    #
    # train_df_TGSiS = downcast_numeric(train_df_TGSiS)
    # test_df_TGSiS = downcast_numeric(test_df_TGSiS)
    # X_train_TGSiS = tupleify(train_df_TGSiS, FEATURE_COLUMNS)
    # X_test_TGSiS = tupleify(test_df_TGSiS, FEATURE_COLUMNS)
    # y_train_TGSiS = train_df_TGSiS["glycan"].tolist()
    # y_test_TGSiS = test_df_TGSiS["glycan"].tolist()
    #
    # with open(output_dir / "X_train_CGSiS.pkl", "wb") as fh:
    #     pickle.dump(X_train_CGSiS, fh)
    # with open(output_dir / "X_test_CGSiS.pkl", "wb") as fh:
    #     pickle.dump(X_test_CGSiS, fh)
    # with open(output_dir / "y_train_CGSiS.pkl", "wb") as fh:
    #     pickle.dump(y_train_CGSiS, fh)
    # with open(output_dir / "y_test_CGSiS.pkl", "wb") as fh:
    #     pickle.dump(y_test_CGSiS, fh)
    #
    #
    # with open(output_dir / "X_train_TGSiS.pkl", "wb") as fh:
    #     pickle.dump(X_train_TGSiS, fh)
    # with open(output_dir / "X_test_TGSiS.pkl", "wb") as fh:
    #     pickle.dump(X_test_TGSiS, fh)
    # with open(output_dir / "y_train_TGSiS.pkl", "wb") as fh:
    #     pickle.dump(y_train_TGSiS, fh)
    # with open(output_dir / "y_test_TGSiS.pkl", "wb") as fh:
    #     pickle.dump(y_test_TGSiS, fh)
########################################################################################################################
    # Random split
########################################################################################################################
    # print("Random split")
    # glycan_counts = combined["glycan"].value_counts()
    # combined["glycan_freq"] = combined["glycan"].map(lambda glycan: get_freq_category(glycan, glycan_counts))
    # # Use the frequency category for stratification
    # unique_glycans = combined[['glycan', 'glycan_freq']].drop_duplicates()
    # train_glycans, test_glycans = train_test_split(
    #     unique_glycans['glycan'],
    #     test_size = test_size,
    #     random_state = 42,
    #     stratify = unique_glycans['glycan_freq']
    # )
    #
    # train_df_R = combined[combined['glycan'].isin(train_glycans)]
    # test_df_R = combined[combined['glycan'].isin(test_glycans)]
    # train_df_R = downcast_numeric(train_df_R)
    # test_df_R = downcast_numeric(test_df_R)
    # X_train_R = tupleify(train_df_R, FEATURE_COLUMNS)
    # X_test_R = tupleify(test_df_R, FEATURE_COLUMNS)
    # y_train_R = train_df_R["glycan"].tolist()
    # y_test_R = test_df_R["glycan"].tolist()

    # with open(output_dir /"X_train_R.pkl", "wb") as fh:
    #     pickle.dump(X_train_R, fh)
    # with open(output_dir /"X_test_R.pkl", "wb") as fh:
    #     pickle.dump(X_test_R, fh)
    # with open(output_dir /"y_train_R.pkl", "wb") as fh:
    #     pickle.dump(y_train_R, fh)
    # with open(output_dir /"y_test_R.pkl", "wb") as fh:
    #     pickle.dump(y_test_R, fh)
########################################################################################################################
    # Similarity split
########################################################################################################################
    # print("Similarity split")
    # combined = add_embeddings_to_df(
    #     combined,
    #     glycan_path = "./",
    #     glycan_pattern = "graph_tokens.pt",
    #     glycan_col = 'glycan_ids'
    # )
    # combined.dropna(subset = ['GlycanGT_mean'], inplace = True)
    #
    # # Extract unique lectin embeddings for similarity matrix
    # print("\n" + "=" * 50)
    # print("Preparing unique binned_intensities for similarity matrix...")
    # print("=" * 50)
    # unique_glycans, glycan_embeddings = prepare_unique_embeddings(combined, 'glycan_ids', 'GlycanGT_mean')
    # print(f"Number of unique glycan: {len(unique_glycans)}")
    # print(f"binned_intensities shape: {glycan_embeddings.shape}")
    #
    # # Compute lectin similarity matrix
    # lectin_sim_file = output_dir / "bin_sim_matrix.tsv"
    # lectin_sim_matrix = gpu_cosine_similarity_unique(
    #     glycan_embeddings,
    #     unique_glycans,
    #     lectin_sim_file,
    #     batch_size = 3482,
    #     desc = "Computing lectin similarity"
    # )
    #
    # # Now use datasail for splitting
    # print("\n" + "=" * 50)
    # print("Running datasail for data splitting...")
    # print("=" * 50)
    #
    # e_splits, f_splits, inter_sp = datasail(
    #     e_type = "M",  # Multi-entity split
    #     e_data = dict(combined[["glycan_ids", "GlycanGT_mean"]].values.tolist()),  # Map index to lectin ID
    #     e_sim = lectin_sim_file,  # Use lectin similarity matrix
    #     names = ["train", "test"],
    #     splits = [0.8, 0.2],
    #     techniques = ["C1e"],  # Constrained 1-entity split
    #     solver = "GUROBI",
    #     epsilon = 0.03,
    #     max_sec = 600
    # )
    #
    # # Get the split dictionary
    # print(f"this is e_splits {e_splits}")
    # print(f"this is f_splits {f_splits}")
    # print(f"this is inter_sp {inter_sp}")
    # split_dict = e_splits['C1e'][0]
    # combined['split'] = combined['glycan_ids'].map(split_dict)
    #
    # # Create train and test sets
    # train_df_C1e = combined[combined["split"] == "train"]
    # test_df_C1e = combined[combined["split"] == "test"]
    # train_df_C1e = downcast_numeric(train_df_C1e)
    # test_df_C1e = downcast_numeric(test_df_C1e)
    # X_train_C1e = tupleify(train_df_C1e, FEATURE_COLUMNS)
    # X_test_C1e = tupleify(test_df_C1e, FEATURE_COLUMNS)
    # y_train_C1e = train_df_C1e["glycan"].tolist()
    # y_test_C1e = test_df_C1e["glycan"].tolist()
    # with open(output_dir /"X_train_C1e.pkl", "wb") as fh:
    #     pickle.dump(X_train_C1e, fh)
    # with open(output_dir /"X_test_C1e.pkl", "wb") as fh:
    #     pickle.dump(X_test_C1e, fh)
    # with open(output_dir /"y_train_C1e.pkl", "wb") as fh:
    #     pickle.dump(y_train_C1e, fh)
    # with open(output_dir /"y_test_C1e.pkl", "wb") as fh:
    #     pickle.dump(y_test_C1e, fh)

########################################################################################################################
########################################################################################################################
    # # print("Plot data distribution")
    # glycan_counts = combined["glycan"].value_counts()
    # plt.figure(figsize=(8, 5))
    # sns.kdeplot(glycan_counts, fill=True)
    # plt.xscale("log")
    # plt.xlabel("Number of occurrences per glycan (log scale)")
    # plt.ylabel("Density")
    # plt.title("KDE of glycan frequencies")
    # plt.tight_layout()
    # plt.savefig("glycan_kde_distribution_log.png", dpi=300)
    # plt.show()
    #
    # plt.figure(figsize=(8, 5))
    # sns.histplot(glycan_counts, bins=50)
    #
    # plt.xscale("log")
    # plt.xlabel("Number of occurrences per glycan (log scale)")
    # plt.ylabel("Number of glycans")
    # plt.title("Distribution of glycan frequencies")
    # plt.tight_layout()
    # plt.savefig("glycan_count_histogram_log.png", dpi=300)
    # plt.show()
    #
    #
    # plt.figure(figsize=(8, 5))
    # sns.ecdfplot(glycan_counts)
    #
    # plt.xscale("log")
    # plt.xlabel("Number of occurrences per glycan (log scale)")
    # plt.ylabel("Proportion of glycans")
    # plt.title("ECDF of glycan frequencies")
    # plt.tight_layout()
    # plt.savefig("glycan_count_ecdf.png", dpi=300)
    # plt.show()
########################################################################################################################
########################################################################################################################
    print(f"Saved processed datasets to {output_dir.resolve()}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Data Processing')
    parser.add_argument('--data_processing', type = str, required = True, choices = ["org", "DS"])
    # parser.add_argument('--nclusters', type = int, required = False, default = 5)
    parser.add_argument('--keep_scale', type = float, required = False, default = 9.0)
    parser.add_argument('--keep_fraction', type = float, required = False, default = 0.1)
    parser.add_argument('--keep_policy', type = str, required = False, choices = ["sqrt", "log2","fraction"], default = "sqrt")
    parser.add_argument('--min_clean_peaks', type = int, required = False, default = 5)
    parser.add_argument('--rel_intensity_min', type = float, required = False, default = 0.005)
    args = parser.parse_args()
    main(args)
