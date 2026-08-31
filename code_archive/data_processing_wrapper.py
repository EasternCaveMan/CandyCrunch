import pandas as pd
import os
from pathlib import Path
import re
import time
import traceback
from datetime import datetime
import argparse

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
                       row_count, start_time, summary=None, error=None, message=None):
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
    if message is not None:
        lines.extend([
            f"Message: {message}",
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
        file_df = pd.read_csv(mz_maps_dir / file).copy()
        if "_O" in file_name:
            file_df["glycan_class"] = 0
        elif "_N" in file_name:
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
            output_dir = Path(f"./raw_files/{glycopost_id}")

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
                if any(marker in file_name for marker in ("_N", "_O")):
                    file_name = file_name.replace("_N", "_1").replace("_O", "_0")
                    group_run_id = f"{file_name}_{rt_group}"
                else:
                    group_run_id = f"{file_name}_{glycan_class}_{rt_group}"
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
                                                                     group_run_id=group_run_id )


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

        # Non-glycoPost
        else:
            files_downloaded = False
            file_name = file.removeprefix("df_mz_").rsplit(".", 1)[0]
            folder_name = file_name.removesuffix("_N").removesuffix("_O")
            files_path = Path(f"./raw_files/{folder_name}")
            if files_path.exists():
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

            if not files_downloaded:
                missing_message = (
                    f"Missing local files folder: {files_path}. "
                    "Non-GPST datasets cannot be downloaded automatically, so this group was skipped."
                )
                for group_num, ((glycan_class, rt_group), group_df) in enumerate(grouped, start=1):
                    rt = rt_group == "rtT"
                    if any(marker in file_name for marker in ("_N", "_O")):
                        report_group_id = f"{file_name.replace('_N', '_1').replace('_O', '_0')}_{rt_group}"
                    else:
                        report_group_id = f"{file_name}_{glycan_class}_{rt_group}"
                    print(f"{file}: skipped missing local files folder {files_path}. Group {group_num} out of {total_groups} group(s)")
                    write_group_report(
                        report_path=report_path,
                        status="SKIPPED",
                        file_name=file,
                        glycopost_id=report_group_id,
                        glycan_class=glycan_class,
                        retention=rt,
                        files_downloaded=files_downloaded,
                        files_path=files_path,
                        output_dir=files_path,
                        row_count=len(group_df),
                        start_time=time.time(),
                        message=missing_message,
                    )
                continue

            for group_num, ((glycan_class, rt_group), group_df) in enumerate(grouped, start = 1):
                rt = rt_group == "rtT"
                if any(marker in file_name for marker in ("_N", "_O")):
                    file_name = file_name.replace("_N", "_1").replace("_O", "_0")
                    group_run_id = f"{file_name}_{rt_group}"
                else:
                    group_run_id = f"{file_name}_{glycan_class}_{rt_group}"
                group_mapping_df = group_df.drop(columns=["rt_group"])
                start_time = time.time()
                try:
                    print(f"{file}: File_name={file_name}, glycan_class={glycan_class}, retention={rt}. Group {group_num} out of {total_groups} group(s)")
                    summary = candycrunch_processing.run_processing( glcopost_id = file_name,
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
                                                                     group_run_id=group_run_id )
                    if file_name not in completed_ids:
                        with report_history.open("a", encoding="utf-8") as handle:
                            handle.write(f"{file_name}\n")
                        completed_ids.add(file_name)
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
    args = parser.parse_args()
    main(args)
