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


def merge_files(files, output_file_fd,output_file_hcd, drop_duplicates=True):
    """Merge files into one dataframe and save it."""
    output_file = Path(output_file_fd)
    if not files:
        print(f"No files found for {output_file.name}")
        return pd.DataFrame()

    dataframes = []
    for file_path in files:
        print(f"Reading: {file_path}")
        dataframes.append(read_dataframe(file_path))

    merged_df = pd.concat(dataframes, ignore_index=True)
    if drop_duplicates:
        try:
            merged_df = merged_df.drop_duplicates().reset_index(drop=True)
        except TypeError as e:
            print(f"Could not drop duplicates because some columns contain unhashable values: {e}")
            print("Keeping all rows.")
    merged_df["glycan_type"] = merged_df["glycan"].apply(get_glycan_label)

    ## HC
    # best_data= merged_df[(merged_df["on_peak"]==True) & (merged_df["xic_peak_found"]==True) &  (merged_df["glycan_match_status"] == "unique")]
    ## XPF
    # best_data= merged_df[merged_df["xic_peak_found"]==True]
    ## OP
    # best_data= merged_df[merged_df["on_peak"]==True]
    ## GMS
    best_data= merged_df[merged_df["glycan_match_status"] == "unique"]

    # # ## OPS
    # best_data = merged_df[merged_df["on_peak"] == True].copy()
    # passes = best_data["glycan_score"] <= 0.9
    # # Glycans having at least one row that passes
    # has_passing = passes.groupby(best_data["glycan"]).transform("any")
    # # Keep passing rows from those glycans
    # passing_rows = best_data[passes & has_passing]
    #
    # # For glycans where nothing passes, keep the 7 smallest scores
    # fallback_rows = (best_data[~has_passing].sort_values("glycan_score").groupby("glycan", group_keys=False).head(7))
    # best_data = pd.concat([passing_rows, fallback_rows],ignore_index=True)

    write_dataframe(merged_df, output_file_fd)
    print("\nDone")
    print(f"Output file: {output_file_fd}")
    print(f"Files merged: {len(files)}")
    print(f"Full dataset: {len(merged_df)}")

    write_dataframe(best_data, output_file_hcd)
    print("\nDone")
    print(f"Output file: {output_file_hcd}")
    print(f"High confident Dataset: {len(best_data)}")

    write_final_dataset_report(merged_df, output_file_fd)
    print("\nDone")
    print(f"Output file: {output_file_fd}")
    print(f"Full dataset: {len(merged_df)}")
    unique_glycans = count_unique_glycans(merged_df)
    if unique_glycans is not None:
        print(f"Unique glycans: {unique_glycans}")


    write_final_dataset_report(best_data, output_file_hcd)
    print("\nDone")
    print(f"Output file: {output_file_hcd}")
    print(f"High confident Dataset: {len(best_data)}")
    unique_glycans = count_unique_glycans(best_data)
    if unique_glycans is not None:
        print(f"Unique glycans for High confident Dataset: {unique_glycans}")

    return merged_df, best_data


# def merge_with_current_data(current_data_path=None, new_data_path=None, output_file=None,
#                             drop_duplicates=True):
#     """Merge with an existing full dataset, or save new data as the full dataset."""
#     new_data_path = Path(new_data_path)
#     if not new_data_path.exists():
#         print(f"New merged training data does not exist: {new_data_path}")
#         print("Skipping full-dataset merge.")
#         return pd.DataFrame()
#
#     new_df = read_dataframe(new_data_path)
#     if current_data_path is None:
#         print("No existing full_dataset_* file found.")
#         print("Saving newly merged training data as the full dataset.")
#         merged_df = new_df
#     else:
#         current_data_path = Path(current_data_path)
#         if current_data_path.exists():
#             dataframes = [
#                 read_dataframe(current_data_path),
#                 new_df,
#             ]
#             merged_df = pd.concat(dataframes, ignore_index=True)
#             print(f"Merging with existing full dataset: {current_data_path}")
#         else:
#             print(f"Current full dataset does not exist: {current_data_path}")
#             print("Saving newly merged training data as the full dataset.")
#             merged_df = new_df
#
#     if drop_duplicates:
#         try:
#             merged_df = merged_df.drop_duplicates().reset_index(drop=True)
#         except TypeError as e:
#             print(f"Could not drop duplicates because some columns contain unhashable values: {e}")
#             print("Keeping all rows.")
#
#     write_dataframe(merged_df, output_file)
#     write_final_dataset_report(merged_df, output_file)
#     print("\nDone")
#     print(f"Output file: {output_file}")
#     print(f"Rows merged: {len(merged_df)}")
#     unique_glycans = count_unique_glycans(merged_df)
#     if unique_glycans is not None:
#         print(f"Unique glycans: {unique_glycans}")
#     return merged_df


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
    return parser


def main(args=None):
    args = build_arg_parser().parse_args(args)
    date_str = datetime.today().strftime("%Y%m%d")
    drop_duplicates = not args.keep_duplicates

    train_files = find_matching_files(args.base_dir, "train")
    pretrain_files = find_matching_files(args.base_dir, "pretrain")

    FD_output = args.base_dir / f"full{date_str}.{args.output_format}"
    GMS_output = args.base_dir / f"GMS{date_str}.{args.output_format}"
    pretrain_output = args.base_dir / f"all_pretrain_{date_str}.{args.output_format}"

    merge_files(
        files=train_files,
        output_file_fd=FD_output,
        output_file_hcd=GMS_output,
        drop_duplicates=drop_duplicates,
    )

    # merge_files(
    #     files=pretrain_files,
    #     output_file=pretrain_output,
    #     drop_duplicates=drop_duplicates,
    # )

    # if not args.skip_current_merge:
    #     current_train = args.current_train
    #     if current_train is None:
    #         current_train = find_latest_full_dataset(args.training_dir)
    #     full_output = args.training_dir / f"full_dataset_{date_str}.{args.output_format}"
    #     merge_with_current_data(
    #         current_data_path=current_train,
    #         new_data_path=train_output,
    #         output_file=full_output,
    #         drop_duplicates=drop_duplicates,
    #     )


if __name__ == "__main__":
    main()
