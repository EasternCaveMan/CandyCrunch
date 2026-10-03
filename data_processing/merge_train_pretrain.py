from pathlib import Path
from datetime import datetime
import argparse
import pandas as pd


def normalize_columns(df):
    """
    Make sure m/z and reducing_mass are merged into one column called m/z.
    """
    if "m/z" in df.columns and "reducing_mass" in df.columns:
        df["m/z"] = df["m/z"].combine_first(df["reducing_mass"])
        df = df.drop(columns=["reducing_mass"])
    elif "reducing_mass" in df.columns and "m/z" not in df.columns:
        df = df.rename(columns={"reducing_mass": "m/z"})
    return df


def read_dataframe(file_path):
    """Read pkl/xlsx/csv data and normalize legacy mass column names."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(file_path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    elif suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported input file type: {file_path}")
    return normalize_columns(df)


def write_dataframe(df, output_file):
    """Write merged data using the output file extension."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_file.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        df.to_pickle(output_file)
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(output_file, index=False)
    elif suffix == ".csv":
        df.to_csv(output_file, index=False)
    else:
        raise ValueError(f"Unsupported output file type: {output_file}")


def format_file_size(file_path):
    """Return a readable file size string."""
    file_path = Path(file_path)
    if not file_path.exists():
        return "0 bytes"

    size = file_path.stat().st_size
    for unit in ("bytes", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            if unit == "bytes":
                return f"{size} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024

def get_glycan_label(glycan):
    if pd.isna(glycan):
        return 3
    from glycowork.motif.processing import get_class

    cls = get_class(glycan)
    if cls == "N":
        return 1
    elif cls == "O":
        return 0
    elif cls in {"free", "lipid", "lipid/free"}:
        return 2
    else:  # repeat or ''
        return 3

def count_unique_glycans(df):
    """Count unique glycans if the glycan column is present."""
    if "glycan" not in df.columns:
        return None
    return df["glycan"].dropna().nunique()


def write_final_dataset_report(df, output_file, report_file=None):
    """Write a small text report for the final merged dataset."""
    output_file = Path(output_file)
    if report_file is None:
        report_file = output_file.with_name(f"{output_file.stem}_report.txt")
    else:
        report_file = Path(report_file)

    unique_glycans = count_unique_glycans(df)
    unique_glycans_text = "N/A - no glycan column" if unique_glycans is None else str(unique_glycans)
    lines = [
        "Final dataset report",
        "=" * 80,
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset file: {output_file}",
        f"File size: {format_file_size(output_file)}",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Unique glycans: {unique_glycans_text}",
        "",
    ]

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Final dataset report: {report_file}")


def find_matching_files(base_dir, dataset_type):
    """Find train/pretrain files from the new pkl pipeline and older Excel outputs."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        print(f"Results directory does not exist: {base_dir}")
        return []

    if dataset_type == "train":
        patterns = ("train_*.pkl", "train_*.pickle", "*_train.xlsx", "*_train.csv")
    elif dataset_type == "pretrain":
        patterns = ("pretrain_*.pkl", "pretrain_*.pickle", "*_pretrain.xlsx", "*_pretrain.csv")
    else:
        raise ValueError("dataset_type must be 'train' or 'pretrain'")

    matched_files = []
    for pattern in patterns:
        matched_files.extend(base_dir.rglob(pattern))

    return sorted({file_path for file_path in matched_files if file_path.is_file()})


def find_latest_full_dataset(training_dir):
    """Find the latest existing full_dataset file by filename."""
    training_dir = Path(training_dir)
    if not training_dir.exists():
        return None

    candidates = []
    for pattern in ("full_dataset_*.pkl", "full_dataset_*.pickle",
                    "full_dataset_*.xlsx", "full_dataset_*.csv"):
        candidates.extend(training_dir.glob(pattern))

    candidates = sorted(file_path for file_path in candidates if file_path.is_file())
    return candidates[-1] if candidates else None


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Merge CandyCrunch train/pretrain outputs from the new pkl pipeline."
    )
    parser.add_argument("--base-dir", type=Path, default=Path("Final_Results"),
                        help="Directory containing *_results folders")
    parser.add_argument("--training-dir", type=Path, default=Path("../training"),
                        help="Directory where full_dataset_* files are read/written")
    parser.add_argument("--current-train", type=Path, default=None,
                        help="Optional existing full training dataset; otherwise latest full_dataset_* is used if present")
    parser.add_argument("--skip-current-merge", action="store_true",
                        help="Only merge new train/pretrain outputs; do not merge with current full dataset")
    parser.add_argument("--output-format", choices=("pkl", "csv", "xlsx"), default="pkl",
                        help="Format for merged outputs")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="Do not drop duplicate rows after merging")
    parser.add_argument('--datasets', nargs='+', type=str, default=["OPS", "HC", "OP", "XPF", "GMS", "full"]) #
    return parser


def main(args=None):
    args = build_arg_parser().parse_args(args)
    date_str = datetime.today().strftime("%Y%m%d")
    drop_duplicates = not args.keep_duplicates

    files = find_matching_files(args.base_dir, "train")
    # files = find_matching_files(args.base_dir, "pretrain")
    dataframes = []
    for file_path in files:
        print(f"Reading: {file_path}")
        dataframes.append(read_dataframe(file_path))
    merged_df = pd.concat(dataframes, ignore_index=True)
    del dataframes
    if drop_duplicates:
        try:
            merged_df = merged_df.drop_duplicates().reset_index(drop=True)
        except TypeError as e:
            print(f"Could not drop duplicates because some columns contain unhashable values: {e}")
            print("Keeping all rows.")
    merged_df["glycan_type"] = merged_df["glycan"].apply(get_glycan_label)
    cglycans1 = pd.read_csv("corrected_glycans_1.csv")
    cglycans2 = pd.read_csv("corrected_glycans_2.csv")
    mapping1 = (cglycans1.loc[cglycans1["correct_glycan"].notna()].set_index("glycan")["correct_glycan"])
    merged_df["glycan"] = merged_df["glycan"].map(mapping1).fillna(merged_df["glycan"])
    mapping2 = (cglycans2.loc[cglycans2["new_glycan"].notna()].set_index("glycan")["new_glycan"])
    merged_df["glycan"] = merged_df["glycan"].map(mapping2).fillna(merged_df["glycan"])
    columns = ['m/z', 'peak_d', 'RT', 'precursor_charge', 'peak_RT', 'on_peak', 'xic_peak_found', 'glycan',
               'glycan_score',
               'glycan_match_status', 'filename', 'GlycoPost_ID', 'mode', 'trap', 'instrument', 'fragmentation',
               'ion_source',
               'lc_type', 'modification']
    merged_df = merged_df[columns]


    for dataset in args.datasets:
        print(f"\n=== Creating  {dataset} dataset ===")
        threshold = 0.7
        output = (args.base_dir/f"{f'{dataset}{threshold}' if dataset == 'OPS' else dataset}_{date_str}.{args.output_format}")
        # output = args.base_dir / f"all_pretrain_{dataset}{date_str}.{args.output_format}"
        output_file = Path(output)
        ## HC
        if dataset == "HC":
            final_df = merged_df[(merged_df["on_peak"] == True) & (merged_df["xic_peak_found"] == True) & (
                        merged_df["glycan_match_status"] == "unique")]
        ## XPF
        if dataset == "XPF":
            final_df = merged_df[merged_df["xic_peak_found"] == True]
        ## OP
        if dataset == "OP":
            final_df = merged_df[merged_df["on_peak"] == True]
        ## GMS
        if dataset == "GMS":
            final_df = merged_df[merged_df["glycan_match_status"] == "unique"]
        # # ## OPS
        if dataset == "OPS":
            final_df = merged_df[merged_df["on_peak"] == True].copy()
            passes = final_df["glycan_score"] <= threshold
            # Glycans having at least one row that passes
            has_passing = passes.groupby(final_df["glycan"]).transform("any")
            # Keep passing rows from those glycans
            passing_rows = final_df[passes & has_passing]

            # For glycans where nothing passes, keep the 7 smallest scores
            fallback_rows = (
                final_df[~has_passing].sort_values("glycan_score").groupby("glycan", group_keys=False).head(7))
            final_df = pd.concat([passing_rows, fallback_rows], ignore_index=True)

        if dataset == "full":
            final_df = merged_df

        write_dataframe(final_df, output_file)
        print("\nDone")
        print(f"Output file: {output_file}")
        print(f"{dataset} Dataset: {len(final_df)}")

        write_final_dataset_report(final_df, output_file)
        print("\nDone")
        print(f"Output file: {output_file}")
        print(f"{dataset} dataset: {len(final_df)}")
        unique_glycans = count_unique_glycans(final_df)
        if unique_glycans is not None:
            print(f"Unique glycans: {unique_glycans}")
        del final_df


if __name__ == "__main__":
    main()
