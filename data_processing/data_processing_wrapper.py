import pandas as pd
import os
from pathlib import Path
import sys
import re
import time
import traceback
from datetime import datetime
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import candycrunch_processingV5_db6 as candycrunch_processing

"""
O:0
N:1
free/lipid:2
E:3
"""

def get_all_files_name_from_dir(path):
    """Get all files from a directory with allowed extensions"""
    allowed_extensions = {'.csv'}
    files = []

    if not os.path.exists(path):
        print(f"✗ Path does not exist: {path}")
        return files

    for file in os.listdir(path):
        file_lower = file.lower()
        if any(file_lower.endswith(ext) for ext in allowed_extensions):
            files.append(file)
            print(f"    Found file: {file}")

    return files


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


def write_group_report(report_path, status, file_name, glycopost_id, glycan_class,
                       retention, files_downloaded, files_path, output_dir,
                       row_count, start_time, summary=None, error=None):
    """Append one text report block for a processed mapping-file/glycan-class group."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    end_time = time.strftime("%Y-%m-%d %H:%M:%S")
    duration_seconds = time.time() - start_time

    lines = [
        "=" * 80,
        f"Status: {status}",
        f"Mapping file: {file_name}",
        f"GlycoPost_ID: {glycopost_id}",
        f"Glycan class: {glycan_class}",
        f"Retention: {retention}",
        f"Rows in mapping group: {row_count}",
        f"Files downloaded before group: {files_downloaded}",
        f"Files path: {files_path}",
        f"Output dir: {output_dir}",

    ]
    if summary is not None:
        lines.extend([
            f"Training datapoints: {summary.get('training_datapoints', 0)}",
            f"Unique glycans extracted: {summary.get('unique_glycans', 0)}",
            f"Pre-training datapoints: {summary.get('pretraining_datapoints', 0)}",
            f"Finished at: {end_time}",
            f"Duration seconds: {duration_seconds:.2f}",
        ])
    if error is not None:
        lines.extend([
            f"Error type: {type(error).__name__}",
            f"Error message: {error}",
            "Traceback:",
            traceback.format_exc().rstrip(),
        ])
    lines.append("")

    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main(args):
    mz_maps_dir = Path(args.mz_maps_dir)
    data_processing_dir = Path(__file__).resolve().parent
    folder_name = mz_maps_dir.name
    date_str = datetime.today().strftime("%Y%m%d")
    report_path = Path(__file__).resolve().with_name(f"report_{folder_name}_{date_str}.txt")
    report_history = Path(__file__).resolve().with_name(f"history.txt")
    report_history.parent.mkdir(parents=True, exist_ok=True)
    if report_history.exists():
        completed_ids = {
            line.strip()
            for line in report_history.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    else:
        completed_ids = set()

    files = get_all_files_name_from_dir(mz_maps_dir)
    print(f"Found {len(files)} files")
    for file in files:
        file_name = file.removeprefix("df_mz_").rsplit(".", 1)[0]
        # _N/_O is a class suffix (it is stripped with removesuffix), so names like "_Neutrophil" must not force class 1
        run_base = file_name.removesuffix("_N").removesuffix("_O")
        file_df = pd.read_csv(mz_maps_dir / file)
        if file_name.endswith("_O"):
            file_df["glycan_class"] = 0
        elif file_name.endswith("_N"):
            file_df["glycan_class"] = 1
        else:
            file_df["glycan_class"] = file_df["glycan"].apply(get_glycan_label)
        file_df.to_csv(mz_maps_dir / file, index = False)
        if "GPST" in file:
            glycopost_match = re.search(r"(GPST\d+)", file)
            if glycopost_match is None:
                continue
            glycopost_id = glycopost_match.group(1)
            files_downloaded = False
            files_path = None
            output_dir = data_processing_dir / "raw_files" / glycopost_id

            if glycopost_id in completed_ids and output_dir.exists():
                files_downloaded = True
                files_path = output_dir
            if "RT" in file_df.columns:
                file_df["rt_group"] = file_df["RT"].notna().map({
                    True: "rtT",
                    False: "rtF",
                })
            else:
                file_df["rt_group"] = "rtF"

            grouped = file_df.groupby(["glycan_class", "rt_group"])
            total_groups = len(grouped)

            for group_num, ((glycan_class, rt_group), group_df) in enumerate(grouped, start = 1):
                rt = rt_group == "rtT"
                # file_name used to be rewritten here, so the 2nd group of "X_N" became "X_1_1_rtF" instead of "X_1_rtF"
                group_run_id = f"{run_base}_{glycan_class}_{rt_group}"
                group_mapping_df = group_df.drop(columns=["rt_group"])
                files_downloaded_before = files_downloaded
                files_path_before = files_path
                start_time = time.time()
                try:
                    print(f"{file}: GlycoPost_ID={glycopost_id}, glycan_class={glycan_class}, retention={rt}. Group {group_num} out of {total_groups} group(s)")
                    summary = candycrunch_processing.run_processing( glcopost_id = glycopost_id,
                                                                     output_dir = output_dir,
                                                                     mapping_file = group_mapping_df,
                                                                     glycan_class = glycan_class,
                                                                     retention = rt,
                                                                     files_path = files_path_before,
                                                                     files_downloaded = files_downloaded_before,
                                                                     xic_mode = "score",
                                                                     mass_tolerance = 0.5,
                                                                     rt_tolerance = 1.0,
                                                                     xic_mz_tolerance = 0.2,
                                                                     xic_score_margin = 0.25,
                                                                     group_run_id=group_run_id,
                                                                     peak_workers=args.peak_workers )


                    files_downloaded = True
                    files_path = output_dir
                    if glycopost_id not in completed_ids:
                        with report_history.open("a", encoding="utf-8") as handle:
                            handle.write(f"{glycopost_id}\n")
                        completed_ids.add(glycopost_id)
                    write_group_report(
                        report_path=report_path,
                        status="SUCCESS",
                        file_name=file,
                        glycopost_id=group_run_id,
                        glycan_class=glycan_class,
                        retention=rt,
                        files_downloaded=files_downloaded_before,
                        files_path=files_path_before,
                        output_dir=output_dir,
                        row_count=len(group_mapping_df),
                        start_time=start_time,
                        summary=summary,
                    )

                except Exception as e:
                    print(f"extraction of {file}: {e}")
                    write_group_report(
                        report_path=report_path,
                        status="FAILED",
                        file_name=file,
                        glycopost_id=group_run_id,
                        glycan_class=glycan_class,
                        retention=rt,
                        files_downloaded=files_downloaded_before,
                        files_path=files_path_before,
                        output_dir=output_dir,
                        row_count=len(group_mapping_df),
                        start_time=start_time,
                        error=e,
                    )
        else:
            files_path = data_processing_dir / "raw_files" / run_base
            # no GPST ID means nothing to download; the old files_downloaded=None crashed in _str_to_bool and was logged as a cryptic FAILED
            if not files_path.exists():
                print(f"✗ {file}: no GlycoPOST ID and no raw files in {files_path}, skipping")
                continue
            files_downloaded = True
            if "RT" in file_df.columns:
                file_df["rt_group"] = file_df["RT"].notna().map({
                    True: "rtT",
                    False: "rtF",
                })
            else:
                file_df["rt_group"] = "rtF"

            grouped = file_df.groupby(["glycan_class", "rt_group"])
            total_groups = len(grouped)

            for group_num, ((glycan_class, rt_group), group_df) in enumerate(grouped, start = 1):
                rt = rt_group == "rtT"
                group_run_id = f"{run_base}_{glycan_class}_{rt_group}"
                group_mapping_df = group_df.drop(columns=["rt_group"])
                start_time = time.time()
                try:
                    print(f"{file}: File_name={run_base}, glycan_class={glycan_class}, retention={rt}. Group {group_num} out of {total_groups} group(s)")
                    summary = candycrunch_processing.run_processing( glcopost_id = run_base,
                                                                     output_dir = files_path,
                                                                     mapping_file = group_mapping_df,
                                                                     glycan_class = glycan_class,
                                                                     retention = rt,
                                                                     files_path = files_path,
                                                                     files_downloaded = files_downloaded,
                                                                     xic_mode = "score",
                                                                     mass_tolerance = 0.5,
                                                                     rt_tolerance = 1.0,
                                                                     xic_mz_tolerance = 0.2,
                                                                     xic_score_margin = 0.25,
                                                                     group_run_id=group_run_id,
                                                                     peak_workers=args.peak_workers )
                    if run_base not in completed_ids:
                        with report_history.open("a", encoding="utf-8") as handle:
                            handle.write(f"{run_base}\n")
                        completed_ids.add(run_base)
                    write_group_report(
                        report_path=report_path,
                        status="SUCCESS",
                        file_name=file,
                        glycopost_id=group_run_id,
                        glycan_class=glycan_class,
                        retention=rt,
                        files_downloaded=files_downloaded,
                        files_path=files_path,
                        output_dir=files_path,
                        row_count=len(group_mapping_df),
                        start_time=start_time,
                        summary=summary,
                    )

                except Exception as e:
                    print(f"extraction of {file}: {e}")
                    write_group_report(
                        report_path=report_path,
                        status="FAILED",
                        file_name=file,
                        glycopost_id=group_run_id,
                        glycan_class=glycan_class,
                        retention=rt,
                        files_downloaded=files_downloaded,
                        files_path=files_path,
                        output_dir=files_path,
                        row_count=len(group_mapping_df),
                        start_time=start_time,
                        error=e,
                    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="data_processing_wrapper",)
    parser.add_argument("--mz-maps-dir", type=str, required=True,
                        help="Directory where the mapping file are saved")
    parser.add_argument("--peak-workers", type=int, default=None,
                        help="Parallel mzML peak extraction workers (default: CPU count)")
    args = parser.parse_args()
    main(args)
