#!/usr/bin/env python3

import argparse
import os
from ftplib import FTP_TLS
from posixpath import join as posix_join
from urllib.parse import urlparse

import requests

BASE = "https://massive.ucsd.edu/ProteoSAFe"
METADATA_URL = f"{BASE}/MassiveServlet"
RAW_FILE_EXTENSIONS = (".raw", ".mzml", ".mzxml")

def get_dataset_metadata(dataset):
    print(f"Fetching dataset metadata for {dataset}")

    r = requests.get(
        METADATA_URL,
        params={"function": "massiveinformation", "massiveid": dataset},
        timeout=60,
    )
    r.raise_for_status()

    metadata = r.json()
    if metadata.get("private") == "true" and metadata.get("has_access") != "true":
        raise RuntimeError(
            f"{dataset} is private or your session does not have access to it."
        )
    if not metadata.get("ftp"):
        raise RuntimeError(f"No FTP root found for {dataset}.")

    return metadata


def connect_ftp(host):
    ftp = FTP_TLS(host, timeout=60)
    ftp.login()
    ftp.prot_p()
    return ftp


def walk_ftp(ftp, remote_dir):
    for name, facts in ftp.mlsd(remote_dir):
        if name in {".", ".."}:
            continue

        path = posix_join(remote_dir, name)
        entry_type = facts.get("type")
        if entry_type == "dir":
            yield from walk_ftp(ftp, path)
        elif entry_type == "file":
            yield path


def is_raw_data_file(path):
    return path.lower().endswith(RAW_FILE_EXTENSIONS)


def get_file_list(dataset, raw_only=False):
    metadata = get_dataset_metadata(dataset)
    ftp_root = metadata["ftp"]
    parsed = urlparse(ftp_root)
    remote_root = parsed.path.rstrip("/")
    search_root = posix_join(remote_root, f"raw") if raw_only else remote_root

    print(f"Fetching file list from {ftp_root}")

    ftp = connect_ftp(parsed.hostname)
    try:
        files = sorted(
            path.removeprefix(remote_root + "/") for path in walk_ftp(ftp, search_root)
        )
    finally:
        ftp.quit()

    if raw_only:
        files = [path for path in files if is_raw_data_file(path)]

    if not files:
        raise RuntimeError("No files found. The dataset may be private or the page format has changed.")

    return files, ftp_root


def get_ftp_size(ftp, remote_path):
    try:
        ftp.voidcmd("TYPE I")
        return ftp.size(remote_path)
    except Exception:
        return None


def local_output_path(output_dir, remote_path, strip_raw_prefix=False):
    parts = remote_path.split("/")
    if strip_raw_prefix and parts and parts[0] == "raw":
        parts = parts[1:]
    return os.path.join(output_dir, *parts)


def download(files, ftp_root, output_dir, strip_raw_prefix=False):
    parsed = urlparse(ftp_root)
    remote_root = parsed.path.rstrip("/")

    ftp = connect_ftp(parsed.hostname)
    try:
        for f in files:
            out = local_output_path(output_dir, f, strip_raw_prefix=strip_raw_prefix)
            out_dir = os.path.dirname(out)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            remote_path = posix_join(remote_root, f)
            remote_size = get_ftp_size(ftp, remote_path)
            offset = os.path.getsize(out) if os.path.exists(out) else 0

            if remote_size is not None and offset == remote_size:
                print(f"Skipping complete file {f}")
                continue
            if remote_size is not None and offset > remote_size:
                print(f"Restarting {f}; local file is larger than remote file")
                offset = 0

            mode = "ab" if offset else "wb"
            action = "Resuming" if offset else "Downloading"
            print(f"{action} {f}")

            with open(out, mode) as handle:
                ftp.retrbinary(
                    f"RETR {remote_path}",
                    handle.write,
                    rest=offset if offset else None,
                )
    finally:
        ftp.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="MassIVE accession, e.g. MSV000081579")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--raw-only", action="store_true",
                        help="Download only .raw, .mzML, and .mzXML files")

    args = parser.parse_args()

    files, ftp_root = get_file_list(args.dataset, raw_only=args.raw_only)
    output_dir = args.output
    if output_dir is None:
        output_dir = os.path.join("raw_files", args.dataset) if args.raw_only else "."

    print(f"Found {len(files)} files")
    print(f"Saving files to {output_dir}")

    download(files, ftp_root, output_dir, strip_raw_prefix=args.raw_only)


if __name__ == "__main__":
    main()
