from pandas.io.common import file_exists
import os
import time
import argparse
import subprocess
import pymzml
import numpy as np
import pandas as pd
import zipfile
import shutil
import json
import urllib.parse
import urllib.request
import urllib.error
from scipy.signal import find_peaks, peak_widths
MASS_TOLERANCE = 0.5
RT_TOLERANCE = 1.0
XIC_MZ_TOLERANCE = 0.2
XIC_PEAK_PROMINENCE_FRAC = 0.1
XIC_RT_PAD = 0.25
XIC_SCORE_MARGIN = 0.25
XIC_NOT_ON_PEAK_PENALTY = 0.25
XIC_NO_PEAK_PENALTY = 0.5
NUMBER_PEAKS = 1000
XIC_MODES = {"off", "annotate", "filter", "score"}
MZML_METADATA_FIELDS = (
    "mode",
    "trap",
    "instrument",
    "fragmentation",
    "ion_source",
    "lc_system",
    "modification")
MZML_METADATA_OUTPUT_COLUMNS = {
    "mode": "mode",
    "trap": "trap",
    "instrument": "instrument",
    "fragmentation": "fragmentation",
    "ion_source": "ion_source",
    "lc_system": "lc_type",
    "modification": "modification"}


def _str_to_bool(value):
    """Parse a command-line boolean and reject misspellings."""
    if isinstance(value, bool):
        return value

    value_normalized = value.strip().lower()
    if value_normalized in {"true", "t", "yes", "y", "1"}:
        return True
    if value_normalized in {"false", "f", "no", "n", "0"}:
        return False

    raise argparse.ArgumentTypeError(
        f"expected a boolean value such as True or False, got {value!r}"
    )


def extract_mzml_metadata(run=None, attrs=None, df=None, metadata=None, filepath=None):
    """Extract mzML metadata, read it from attrs, or add it as output columns."""
    metadata_out = {field: None for field in MZML_METADATA_FIELDS}
    if metadata:
        for field in MZML_METADATA_FIELDS:
            metadata_out[field] = metadata.get(field)
    if attrs:
        for field in MZML_METADATA_FIELDS:
            metadata_out[field] = metadata_out[field] or attrs.get(field)
        metadata_out["mode"] = metadata_out["mode"] or attrs.get("detected_mode")
        metadata_out["trap"] = metadata_out["trap"] or attrs.get("detected_trap")

    def update_modification(text):
        if not text or metadata_out["modification"] is not None:
            return
        text_lower = str(text).lower()
        if "permethyl" in text_lower:
            metadata_out["modification"] = "permethylated"
        elif "reduced" in text_lower or "reduction" in text_lower:
            metadata_out["modification"] = "reduced"

    if filepath:
        update_modification(os.path.basename(filepath))

    if run is not None:
        instrument_list = run.info.get("instrument_configuration_list_element")
        if instrument_list is not None:
            for cv in instrument_list.iter():
                if not cv.tag.endswith("cvParam"):
                    continue

                accession = cv.attrib.get("accession", "")
                name = cv.attrib.get("name", "")
                name_lower = name.lower()
                update_modification(name)

                if accession == "MS:1000031":
                    metadata_out["instrument"] = name
                elif "orbitrap" in name_lower:
                    metadata_out["trap"] = "orbitrap"
                elif "linear ion trap" in name_lower:
                    metadata_out["trap"] = "linear"
                elif "quadrupole" in name_lower:
                    metadata_out["trap"] = "quadrupole"
                elif "time-of-flight" in name_lower:
                    metadata_out["trap"] = "tof"
                elif "electrospray ionization" in name_lower:
                    metadata_out["ion_source"] = "ESI"
                elif "nanoelectrospray" in name_lower:
                    metadata_out["ion_source"] = "nanoESI"
                elif "apci" in name_lower:
                    metadata_out["ion_source"] = "APCI"
                elif "maldi" in name_lower:
                    metadata_out["ion_source"] = "MALDI"

                if any(lc_name in name_lower for lc_name in (
                        "easy-nlc",
                        "ultimate",
                        "vanquish",
                        "uplc",
                        "hplc",
                        "nano lc",
                        "nano-lc",
                )):
                    metadata_out["lc_system"] = name

    if df is not None:
        for field, column in MZML_METADATA_OUTPUT_COLUMNS.items():
            df[column] = metadata_out.get(field)
        return df
    return metadata_out


def _read_glycopost_json(url):
    """Read JSON from a public GlycoPOST API URL."""
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "CandyCrunch/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _resolve_glycopost_location(entry_id=None):
    """Resolve GPST000000 to its revisioned location, e.g. GPST000000.0."""
    if entry_id and "." in entry_id:
        return entry_id

    if entry_id:
        quoted_entry_id = urllib.parse.quote(entry_id, safe="")
        metadata_url = f"https://glycopost.glycosmos.org/api/projects/{quoted_entry_id}"
        metadata = _read_glycopost_json(metadata_url)
        if metadata.get("location"):
            return metadata["location"]
        if metadata.get("gpostId") is not None and metadata.get("revision") is not None:
            return f"{metadata['gpostId']}.{metadata['revision']}"

    raise ValueError("A GlycoPOST entry ID is required.")


def get_all_files_from_glycopost_api(entry_id=None, page_size=100):
    """Get downloadable files and the revisioned location from the public GlycoPOST JSON API."""
    allowed_extensions = {'.raw', '.mzml', '.mzxml', '.zip'}
    location = _resolve_glycopost_location(entry_id=entry_id)
    quoted_location = urllib.parse.quote(location, safe="")
    all_files = []
    offset = 0
    total = None

    while True:
        url = (
            f"https://glycopost.glycosmos.org/api/projects/{quoted_location}/files"
            f"?limit={page_size}&offset={offset}"
        )
        payload = _read_glycopost_json(url)
        file_entries = payload.get("list", [])
        if total is None:
            total = payload.get("meta", {}).get("total")
            print(f"  Reading GlycoPOST file list from API location {location}")

        if not file_entries:
            break

        for entry in file_entries:
            file_name = entry.get("name")
            if not file_name:
                continue
            if any(file_name.lower().endswith(ext) for ext in allowed_extensions):
                all_files.append(file_name)
                print(f"    Found via API: {file_name}")

        offset += len(file_entries)
        if total is not None and offset >= total:
            break

    return sorted(set(all_files)), location


def get_all_files_name_from_dir(path):
    """Get all files from a directory with allowed extensions"""
    allowed_extensions = {'.raw', '.mzml', '.mzxml', '.zip'}
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


def _convert_mzxml_to_mzml_docker(mzxml_file_path, output_dir):
    """
    Convert an mzXML file to mzML using Docker msconvert.
    """
    mzxml_abs_path = os.path.abspath(mzxml_file_path)
    mzxml_dir = os.path.dirname(mzxml_abs_path)
    mzxml_filename = os.path.basename(mzxml_abs_path)

    # FIX: Handle both .mzxml and .mzXML extensions
    if mzxml_filename.lower().endswith('.mzxml'):
        base_name = mzxml_filename[:-6]  # Remove last 6 chars
    else:
        base_name = os.path.splitext(mzxml_filename)[0]

    mzml_filename = f"{base_name}.mzML"
    mzml_path = os.path.join(mzxml_dir, mzml_filename)

    print(f"  mzXML file directory: {mzxml_dir}")
    print(f"  mzXML filename: {mzxml_filename}")
    print(f"  Target mzML filename: {mzml_filename}")

    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{mzxml_dir}:/data",
        "chambm/pwiz-skyline-i-agree-to-the-vendor-licenses",
        "wine", "msconvert",
        f"/data/{mzxml_filename}",
        "--mzML",
        "--outfile", f"/data/{mzml_filename}"
    ]

    print("  Converting mzXML to mzML via Docker...")

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            check=True
        )

        if os.path.exists(mzml_path):
            print(f"  ✓ Successfully converted {mzxml_filename} to {mzml_filename}")
            return mzml_path
        else:
            print(f"  ✗ Conversion completed but file not found: {mzml_path}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to convert {mzxml_filename}")
        if e.stderr:
            print(f"    {e.stderr}")
        return None


def _convert_raw_to_mzml_docker(raw_file_path, output_dir):
    """
    Convert a RAW file to mzML using the Docker container of msconvert

    Args:
        raw_file_path: Path to the RAW file (can be relative or absolute)
        output_dir: Directory where mzML file will be saved

    Returns:
        Path to the generated mzML file or None if conversion failed
    """

    raw_abs_path = os.path.abspath(raw_file_path)
    raw_dir = os.path.dirname(raw_abs_path)
    raw_filename = os.path.basename(raw_abs_path)

    print(f"  RAW file directory: {raw_dir}")
    print(f"  RAW filename: {raw_filename}")

    # Docker command to run msconvert
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{raw_dir}:/data",
        "chambm/pwiz-skyline-i-agree-to-the-vendor-licenses",
        "wine", "msconvert", f"/data/{raw_filename}", "--mzML",
        "--outdir", "/data"
    ]

    print(f"  Running msconvert via Docker...")

    try:
        # Run the conversion
        result = subprocess.run(docker_cmd, capture_output=True, text=True, check=True)

        # FIX: Handle both .raw and .RAW extensions
        if raw_filename.lower().endswith('.raw'):
            mzml_filename = raw_filename[:-4] + '.mzML'  # Remove last 4 chars (.raw or .RAW)
        else:
            mzml_filename = raw_filename + '.mzML'

        mzml_path = os.path.join(raw_dir, mzml_filename)

        if os.path.exists(mzml_path):
            print(f"  ✓ Successfully converted {raw_filename} to mzML")
            return mzml_path
        else:
            print(f"  ✗ Conversion completed but {mzml_path} not found")
            return None

    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to convert {raw_filename} with Docker:")
        print(f"    Error: {e}")
        if e.stderr:
            print(f"    stderr: {e.stderr}")
        return None


def _extract_zip_file(zip_path, extract_to):
    """Extract a zip file and return list of extracted files"""
    extracted_files = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            extracted_files = [os.path.join(extract_to, f) for f in zip_ref.namelist()]
        print(f"  ✓ Extracted zip file to: {extract_to}")
        return extracted_files
    except Exception as e:
        print(f"  ✗ Failed to extract zip file: {e}")
        return []


def _process_downloaded_file(file_path, output_dir):
    """
    Process a downloaded file based on its extension:
    - .raw: convert to mzML
    - .mzXML: convert to mzML
    - .mzML: keep as is
    - .zip: extract and recursively process contents
    """
    mzml_files = []
    file_lower = file_path.lower()
    print(f"Processing: {os.path.basename(file_path)}")

    if file_lower.endswith('.raw'):
        print(f"  Converting RAW to mzML...")
        mzml_path = _convert_raw_to_mzml_docker(file_path, output_dir)
        if mzml_path and os.path.exists(mzml_path):
            mzml_files.append(mzml_path)
            print(f"  ✓ Converted to: {os.path.basename(mzml_path)}")
            # Remove original RAW file
            try:
                # os.remove(file_path)
                # print(f"  Removed original RAW file")
                print(f"  Keep the original RAW file")
            except:
                pass
        else:
            print(f"  ✗ Conversion failed")

    elif file_lower.endswith('.mzxml'):
        print(f"  Converting mzXML to mzML...")
        mzml_path = _convert_mzxml_to_mzml_docker(file_path, output_dir)
        if mzml_path and os.path.exists(mzml_path):
            mzml_files.append(mzml_path)
            print(f"  ✓ Converted to: {os.path.basename(mzml_path)}")
            try:
                os.remove(file_path)
                print(f"  Removed original mzXML file")
            except:
                pass
        else:
            print(f"  ✗ Conversion failed")

    elif file_lower.endswith('.mzml'):
        print(f"  File is already mzML, no conversion needed")
        if os.path.exists(file_path):
            mzml_files.append(file_path)
        else:
            print(f"  ✗ File not found: {file_path}")

    elif file_lower.endswith('.zip'):
        print(f"  Extracting zip file...")
        extract_dir = os.path.join(output_dir, f"temp_extracted_{int(time.time())}")
        os.makedirs(extract_dir, exist_ok=True)

        extracted_files = _extract_zip_file(file_path, extract_dir)
        print(f"  Extracted {len(extracted_files)} files from zip")

        if extracted_files:
            print(f"  Processing extracted files...")
            for extracted_file in extracted_files:
                sub_mzml_files = _process_downloaded_file(extracted_file, output_dir)
                mzml_files.extend(sub_mzml_files)
            try:
                shutil.rmtree(extract_dir)
                print(f"  Cleaned up temporary extraction directory")
            except Exception as e:
                print(f"  Warning: Could not clean up {extract_dir}: {e}")
        try:
            os.remove(file_path)
            print(f"  Removed original zip file")
        except:
            pass

    return mzml_files


def download_and_process(download_base_url=None, output_dir=None, file_names=None):
    """Download files and process them based on their extensions"""
    all_mzml_files = []
    os.makedirs(output_dir, exist_ok=True)
    original_cwd = os.getcwd()
    os.chdir(output_dir)
    download_base_url = download_base_url.rstrip('/')
    for file_name in file_names:
        url = f"{download_base_url}/{urllib.parse.quote(file_name)}"
        print(f"\n{'=' * 60}")
        print(f"Processing {file_name}")
        print(f"{'=' * 60}")
        print(f"1. Downloading {file_name}...")
        try:
            subprocess.run(["wget", "-c", url, "-O", file_name], check=True)
            print(f"✓ Successfully downloaded {file_name}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to download {file_name}: {e}")
            continue
        downloaded_file_path = os.path.abspath(file_name)
        print(f"2. Processing {file_name}...")
        mzml_files = _process_downloaded_file(downloaded_file_path, output_dir)

        if mzml_files:
            all_mzml_files.extend(mzml_files)
            for mzml_file in mzml_files:
                print(f"  ✓ Final mzML file: {mzml_file}")
        else:
            print(f"  ✗ No mzML files generated from {file_name}")

        time.sleep(1)
    os.chdir(original_cwd)
    return all_mzml_files


def process_local_files(output_dir = None, file_names = None):
    """Process local files based on their extensions"""
    all_mzml_files = []
    for file_name in file_names:
        # Construct the full file path
        file_path = os.path.join(output_dir, file_name)

        if not os.path.exists(file_path):
            print(f"✗ File not found: {file_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing local file: {file_name}")
        print(f"{'=' * 60}")

        mzml_files = _process_downloaded_file(file_path, output_dir)

        if mzml_files:
            all_mzml_files.extend(mzml_files)
            for mzml_file in mzml_files:
                print(f"  ✓ Final mzML file: {mzml_file}")
        else:
            print(f"  ✗ No mzML files generated from {file_name}")

        time.sleep(1)
    return all_mzml_files


def extract_xic_peak(
        precursor_mz,
        scan_rt,
        ms1_rts,
        ms1_scans,
        mz_tolerance=None,
        min_prominence_frac=None,
        rt_pad=None):
    mz_tolerance = XIC_MZ_TOLERANCE if mz_tolerance is None else mz_tolerance
    min_prominence_frac = XIC_PEAK_PROMINENCE_FRAC if min_prominence_frac is None else min_prominence_frac
    rt_pad = XIC_RT_PAD if rt_pad is None else rt_pad
    empty_result = {
        'peak_RT': scan_rt,
        'on_peak': False,
        'xic_peak_found': False,
        'xic_rt_delta': np.nan,
        'xic_peak_prominence': np.nan,
        'xic_peak_width': np.nan,
        'xic_left_RT': np.nan,
        'xic_right_RT': np.nan,
        'xic_apex_intensity': np.nan,
    }
    if len(ms1_rts) == 0:
        return empty_result
    xic = np.zeros(len(ms1_rts))
    for s in range(len(ms1_scans)):
        mzs, ints = ms1_scans[s]
        lo = np.searchsorted(mzs, precursor_mz - mz_tolerance, side='left')
        hi = np.searchsorted(mzs, precursor_mz + mz_tolerance, side='right')
        if hi > lo:
            xic[s] = ints[lo:hi].sum()
    if xic.max() <= 0:
        return empty_result
    peaks, peak_props = find_peaks(xic, prominence=min_prominence_frac * xic.max())
    if len(peaks) == 0:
        return empty_result
    idx = np.arange(len(ms1_rts))
    _, _, lefts, rights = peak_widths(xic, peaks, rel_height=0.5)
    peak_options = []
    for peak_number, (p, l, r) in enumerate(zip(peaks, lefts, rights)):
        left_rt = float(np.interp(l, idx, ms1_rts))
        right_rt = float(np.interp(r, idx, ms1_rts))
        peak_rt = float(ms1_rts[p])
        on_peak = left_rt - rt_pad <= scan_rt <= right_rt + rt_pad
        peak_options.append({
            'peak_RT': peak_rt,
            'on_peak': on_peak,
            'xic_peak_found': True,
            'xic_rt_delta': abs(float(scan_rt) - peak_rt),
            'xic_peak_prominence': float(peak_props['prominences'][peak_number]),
            'xic_peak_width': right_rt - left_rt,
            'xic_left_RT': left_rt,
            'xic_right_RT': right_rt,
            'xic_apex_intensity': float(xic[p]),
        })

    on_peak_options = [option for option in peak_options if option['on_peak']]
    if on_peak_options:
        return min(on_peak_options, key=lambda option: option['xic_rt_delta'])
    return min(peak_options, key=lambda option: option['xic_rt_delta'])


def _process_mzML_stack(filepath, num_peaks= None,
                        ms_level=2, intensity=False, extract_ms1=False,
                        xic_mode="score"):
    """function extracting all MS/MS spectra from .mzML file"""
    run = pymzml.run.Reader(filepath)
    metadata = extract_mzml_metadata(run=run, filepath=filepath)
    highest_i_dict = {}
    rts, intensities, mzs, charges = [], [], [], []
    detected_mode = metadata.get("mode")
    detected_trap = metadata.get("trap")
    ms1_rts, ms1_scans = [], []
    for spectrum in run:
        if extract_ms1 and spectrum.ms_level == 1:
            peaks_raw = spectrum.peaks("raw")
            if len(peaks_raw) > 0:
                ms1_rts.append(spectrum.scan_time_in_minutes())
                ms1_scans.append((peaks_raw[:, 0].copy(), peaks_raw[:, 1].copy()))
        if spectrum.ms_level == ms_level:
            try:
                temp = spectrum.highest_peaks(2)
            except:
                continue
            # Fallback metadata detector if extract_mzml_metadata failed
            if detected_mode is None or detected_trap is None or metadata.get("fragmentation") is None:
                ns_uri = '{http://psi.hupo.org/ms/mzml}'
                for cv in spectrum.element.iter(f'{ns_uri}cvParam'):
                    acc = cv.get('accession', '')
                    name = cv.get('name', '').lower()
                    if detected_mode is None and acc == 'MS:1000129':
                        detected_mode = 'negative'
                    elif detected_mode is None and acc == 'MS:1000130':
                        detected_mode = 'positive'
                    elif detected_trap is None and acc == 'MS:1000512':
                        filt = cv.get('value', '')
                        if filt.startswith('ITMS'):
                            detected_trap = 'linear'
                        elif filt.startswith('FTMS'):
                            detected_trap = 'orbitrap'
                    if metadata.get("modification") is None:
                        if "permethyl" in name:
                            metadata["modification"] = "permethylated"
                        elif "reduced" in name or "reduction" in name:
                            metadata["modification"] = "reduced"
                    if metadata.get("fragmentation") is None:
                        if "hcd" in name:
                            metadata["fragmentation"] = "HCD"
                        elif "cid" in name:
                            metadata["fragmentation"] = "CID"
                        elif "etd" in name:
                            metadata["fragmentation"] = "ETD"
            mz_i_dict = {}
            num_actual_peaks = min(num_peaks, len(spectrum.peaks("raw")))
            for mz, i in spectrum.highest_peaks(num_actual_peaks):
                mz_i_dict[mz] = i
            if mz_i_dict:
                if not spectrum.selected_precursors:
                    continue
                key = f"{spectrum.ID}_{spectrum.selected_precursors[0]['mz']}"
                highest_i_dict[key] = mz_i_dict
                mzs.append(float(key.split('_')[-1]))
                rts.append(spectrum.scan_time_in_minutes())
                raw_charge = spectrum.selected_precursors[0].get('charge', None)
                if raw_charge is not None and abs(int(raw_charge)) == 1:
                    raw_charge = None
                charges.append(int(raw_charge) if raw_charge is not None else None)
                if intensity:
                    inty = spectrum.selected_precursors[0].get('i', np.nan)
                    intensities.append(inty)
    for key in highest_i_dict.keys():
        highest_i_dict[key] = dict(sorted(highest_i_dict[key].items(), key=lambda x: x[1], reverse=True))
    df_out = pd.DataFrame({
        'm/z': mzs,
        'peak_d': list(highest_i_dict.values()),
        'RT': rts,
        'precursor_charge': charges,
    })
    if intensity:
        df_out['intensity'] = intensities
    metadata["mode"] = metadata.get("mode") or detected_mode
    metadata["trap"] = metadata.get("trap") or detected_trap
    df_out.attrs.update(metadata)
    df_out.attrs['detected_mode'] = detected_mode
    df_out.attrs['detected_trap'] = detected_trap
    if extract_ms1 and xic_mode != "off":
        df_out.attrs['ms1_rts'] = np.array(ms1_rts)
        df_out.attrs['ms1_scans'] = ms1_scans
        if len(ms1_rts) > 0 and len(df_out) > 0:
            xic_results = []
            for r in range(len(df_out)):
                xic_results.append(
                    extract_xic_peak(
                        df_out['m/z'].values[r],
                        df_out.RT.values[r],
                        df_out.attrs['ms1_rts'],
                        ms1_scans
                    )
                )
            for column in xic_results[0]:
                df_out[column] = [result[column] for result in xic_results]
    return df_out


def _match_mz(i, df_mz, df, ms_error=MASS_TOLERANCE, retention=False, implicit_single_charge=False):
    if implicit_single_charge:
        mz = df.precursor_mass.values.tolist()[i]
    else:
        mz = df["m/z"].values.tolist()[i]
    idx = [k for k in range(len(df_mz.Mass.values.tolist())) if
           df_mz.Mass.values.tolist()[k] - ms_error < mz < df_mz.Mass.values.tolist()[k] + ms_error]
    if retention:
        rt = df.peak_RT.values.tolist()[i] if 'peak_RT' in df else df.RT.values.tolist()[i]
        idx = [k for k in idx if
               df_mz.RT.values.tolist()[k] - RT_TOLERANCE < rt < df_mz.RT.values.tolist()[k] + RT_TOLERANCE]
    glycans = [df_mz.glycan.values.tolist()[k] for k in idx]
    return list(set(glycans))


def _validate_mz_charge(glycan_mass, observed_mz, charge, ion_mode, tolerance=MASS_TOLERANCE):
    """
    Check if observed m/z is consistent with theoretical m/z for a given glycan,
    charge state, and ion mode.
    Returns True if valid or cannot be validated, False if clearly inconsistent.
    """
    if charge is None or pd.isna(charge):
        return True
    charge = int(charge)
    proton_mass = 1.007276
    if ion_mode == 'positive':
        if charge <= 0:
            return False
        theo_mz = (glycan_mass + charge * proton_mass) / charge
    elif ion_mode == 'negative':
        if charge >= 0:
            return False
        abs_charge = abs(charge)
        theo_mz = (glycan_mass - abs_charge * proton_mass) / abs_charge
    else:
        return True

    return abs(observed_mz - theo_mz) <= tolerance


def _get_match_rt(row):
    if 'peak_RT' in row.index and not pd.isna(row['peak_RT']):
        return row['peak_RT']
    return row['RT']


def _xic_confidence_penalty(row):
    if 'xic_peak_found' not in row.index:
        return 0.0
    if not bool(row.get('xic_peak_found', False)):
        return XIC_NO_PEAK_PENALTY
    if bool(row.get('on_peak', False)):
        return 0.0
    rt_delta = row.get('xic_rt_delta', np.nan)
    if pd.isna(rt_delta):
        return XIC_NOT_ON_PEAK_PENALTY
    return XIC_NOT_ON_PEAK_PENALTY + 0.1 * min(float(rt_delta) / max(RT_TOLERANCE, 1e-12), 2.0)


def _candidate_glycan_matches(i, df_mz, df, ms_error=MASS_TOLERANCE, retention=False,
                              implicit_single_charge=False, ion_mode=None):
    row = df.iloc[i]
    if implicit_single_charge:
        observed_mz = row['precursor_mass']
    else:
        observed_mz = row['m/z']

    match_rt = _get_match_rt(row)
    charge = row.get('precursor_charge', None)
    candidates = []

    for k, glycan_mass in enumerate(df_mz.Mass.values.tolist()):
        mass_error = abs(float(glycan_mass) - float(observed_mz))
        if mass_error > ms_error:
            continue

        rt_error = 0.0
        if retention and 'RT' in df_mz.columns:
            mapping_rt = df_mz.RT.values.tolist()[k]
            if pd.isna(match_rt) or pd.isna(mapping_rt):
                continue
            rt_error = abs(float(mapping_rt) - float(match_rt))
            if rt_error > RT_TOLERANCE:
                continue

        if not _validate_mz_charge(glycan_mass, observed_mz, charge, ion_mode, tolerance=ms_error):
            continue

        score = mass_error / max(ms_error, 1e-12)
        if retention:
            score += rt_error / max(RT_TOLERANCE, 1e-12)
        score += _xic_confidence_penalty(row)

        candidates.append({
            'glycan': df_mz.glycan.values.tolist()[k],
            'score': score,
            'mass_error': mass_error,
            'rt_error': rt_error,
        })

    best_by_glycan = {}
    for candidate in candidates:
        glycan = candidate['glycan']
        if glycan not in best_by_glycan or candidate['score'] < best_by_glycan[glycan]['score']:
            best_by_glycan[glycan] = candidate
    return list(best_by_glycan.values())


def _select_scored_glycan(i, df_mz, df, ms_error=MASS_TOLERANCE, retention=False,
                          implicit_single_charge=False, ion_mode=None,
                          score_margin=None):
    score_margin = XIC_SCORE_MARGIN if score_margin is None else score_margin
    candidates = _candidate_glycan_matches(
        i,
        df_mz,
        df,
        ms_error=ms_error,
        retention=retention,
        implicit_single_charge=implicit_single_charge,
        ion_mode=ion_mode,
    )
    match_count = len(candidates)
    if match_count == 0:
        return np.nan, match_count, np.nan, np.nan, 'no_match'

    candidates = sorted(candidates, key=lambda candidate: candidate['score'])
    best = candidates[0]
    if match_count == 1:
        return best['glycan'], match_count, best['score'], np.nan, 'unique'

    margin = candidates[1]['score'] - best['score']
    if margin >= score_margin:
        return best['glycan'], match_count, best['score'], margin, 'scored'
    return np.nan, match_count, best['score'], margin, 'ambiguous'


def _print_match_diagnostics(label, df_before, df_after, xic_mode):
    print(f"    {label} diagnostics:")
    print(f"      Rows considered: {len(df_before)}")
    if 'xic_peak_found' in df_before:
        peak_found = int(df_before['xic_peak_found'].fillna(False).sum())
        on_peak = int(df_before['on_peak'].fillna(False).sum()) if 'on_peak' in df_before else 0
        print(f"      XIC peaks found: {peak_found}")
        print(f"      MS2 scans on XIC peak: {on_peak}")
    if 'glycan_match_count' in df_before:
        match_counts = df_before['glycan_match_count'].fillna(0)
        print(f"      Rows with 0 candidate glycans: {int((match_counts == 0).sum())}")
        print(f"      Rows with 1 candidate glycan: {int((match_counts == 1).sum())}")
        print(f"      Rows with multiple candidate glycans: {int((match_counts > 1).sum())}")
    if 'glycan_match_status' in df_before:
        status_counts = df_before['glycan_match_status'].value_counts(dropna=False).to_dict()
        print(f"      Match status counts: {status_counts}")
    print(f"      Rows retained after {xic_mode} matching: {len(df_after)}")


def _glycofy_df(df, df_mz, filename, glycopost_id, ms_error=MASS_TOLERANCE, retention=False,
                implicit_single_charge=False, ion_mode=None, xic_mode="score"):
    """
    Match MS2 spectra to glycans. XIC score mode keeps XIC confidence as a
    ranking signal instead of using it as a hard filter.
    """
    mzml_metadata = extract_mzml_metadata(attrs=df.attrs)
    if xic_mode == "filter" and 'on_peak' in df:
        df = df[df.on_peak].reset_index(drop=True)

    if xic_mode == "score":
        scored_matches = [
            _select_scored_glycan(
                k,
                df_mz,
                df,
                ms_error=ms_error,
                retention=retention,
                implicit_single_charge=implicit_single_charge,
                ion_mode=ion_mode,
            )
            for k in range(len(df))
        ]
        df['glycan'] = [match[0] for match in scored_matches]
        df['glycan_match_count'] = [match[1] for match in scored_matches]
        df['glycan_score'] = [match[2] for match in scored_matches]
        df['glycan_score_margin'] = [match[3] for match in scored_matches]
        df['glycan_match_status'] = [match[4] for match in scored_matches]
        df_out = df.dropna(subset=['glycan']).reset_index(drop=True)
    else:
        df['glycan'] = [_match_mz(k, df_mz, df, ms_error=ms_error,
                                  retention=retention,
                                  implicit_single_charge=implicit_single_charge)
                        for k in range(len(df))]
        df['glycan_match_count'] = [len(list(set(matches))) for matches in df.glycan.values.tolist()]

        # Keep only rows that matched exactly one glycan.
        idx = [k for k in range(len(df)) if len(list(set(df.glycan.values.tolist()[k]))) == 1]
        df_out = df.iloc[idx, :].reset_index(drop=True)
        df_out.glycan = [k[0] for k in df_out.glycan.values.tolist()]
        df_out = df_out.dropna(subset=['glycan']).reset_index(drop=True)

    # CHECK: validate m/z vs. glycan mass & charge
    valid_rows = []
    for i, row in df_out.iterrows():
        glycan = row['glycan']
        # get neutral mass of the matched glycan
        mass_row = df_mz[df_mz['glycan'] == glycan]
        if mass_row.empty:
            continue
        glycan_mass = mass_row['Mass'].values[0]
        observed_mz = row['m/z']
        charge = row['precursor_charge']
        if _validate_mz_charge(glycan_mass, observed_mz, charge, ion_mode, tolerance=ms_error):
            valid_rows.append(i)
        else:
            print(f"    Discarding invalid assignment: {glycan} @ m/z {observed_mz:.2f}, charge {charge}")

    df_out = df_out.iloc[valid_rows, :].reset_index(drop=True)
    _print_match_diagnostics("Training", df, df_out, xic_mode)

    # Continue with original code
    df_out['filename'] = [filename] * len(df_out)
    df_out['GlycoPost_ID'] = [glycopost_id] * len(df_out)
    extract_mzml_metadata(metadata=mzml_metadata, df=df_out)
    if implicit_single_charge:
        df_out.drop(['precursor_mass'], axis=1)
    return df_out


def _match_mz_pretrain(i, df_mz, df, ms_error=MASS_TOLERANCE, retention=False, implicit_single_charge=False):
    if implicit_single_charge:
        mz = df.precursor_mass.values.tolist()[i]
    else:
        mz = df["m/z"].values.tolist()[i]
    idx = [k for k in range(len(df_mz.Mass.values.tolist())) if
           df_mz.Mass.values.tolist()[k] - ms_error < mz < df_mz.Mass.values.tolist()[k] + ms_error]
    if retention:
        rt = df.peak_RT.values.tolist()[i] if 'peak_RT' in df else df.RT.values.tolist()[i]
        idx = [k for k in idx if
               df_mz.RT.values.tolist()[k] - RT_TOLERANCE < rt < df_mz.RT.values.tolist()[k] + RT_TOLERANCE]
    return len(idx) > 0


def _glycofy_df_pretrain(df, df_mz, filename, glycopost_id, glycan_class, ms_error=MASS_TOLERANCE, retention=False,
                         implicit_single_charge=False, xic_mode="score"):
    mzml_metadata = extract_mzml_metadata(attrs=df.attrs)
    if xic_mode == "filter" and 'on_peak' in df:
        df = df[df.on_peak].reset_index(drop=True)

    df['glycan_match_count'] = [
        len(_candidate_glycan_matches(k, df_mz, df, ms_error=ms_error,
                                      retention=retention,
                                      implicit_single_charge=implicit_single_charge))
        for k in range(len(df))
    ]
    df['glycan_type'] = [glycan_class if count > 0 else np.nan
                         for count in df['glycan_match_count'].values.tolist()]
    df_out = df.dropna(subset='glycan_type').reset_index(drop=True)
    _print_match_diagnostics("Pre-training", df, df_out, xic_mode)
    df_out['filename'] = [filename] * len(df_out)
    df_out['GlycoPost_ID'] = [glycopost_id] * len(df_out)
    extract_mzml_metadata(metadata=mzml_metadata, df=df_out)
    if implicit_single_charge:
        df_out.drop(['precursor_mass'], axis=1)
    return df_out


def _wrap_all(filepath, glycopost_id, df_mz, num_peaks=NUMBER_PEAKS, ms_error=MASS_TOLERANCE, retention=False,
              implicit_single_charge=False, xic_mode="score"):
    if 'Downloads' in filepath:
        filename = filepath.split('Downloads')[-1]
    else:
        filename = os.path.basename(filepath)

    df_out = _process_mzML_stack(
        filepath,
        num_peaks=num_peaks,
        extract_ms1=xic_mode != "off",
        xic_mode=xic_mode,
    )
    ion_mode = df_out.attrs.get('detected_mode')   # positive or negative

    df_out = _glycofy_df(df_out, df_mz, filename, glycopost_id,
                         ms_error=ms_error, retention=retention,
                         implicit_single_charge=implicit_single_charge,
                         ion_mode=ion_mode, xic_mode=xic_mode)
    return df_out


def _wrap_all_pretrain(filepath, glycopost_id, df_mz, glycan_class, num_peaks=NUMBER_PEAKS, ms_error=MASS_TOLERANCE, retention=False,
                       implicit_single_charge=False, xic_mode="score"):
    if 'Downloads' in filepath:
        filename = filepath.split('Downloads')[-1]
    else:
        filename = os.path.basename(filepath)
    df_out = _process_mzML_stack(
        filepath,
        num_peaks=num_peaks,
        extract_ms1=xic_mode != "off",
        xic_mode=xic_mode,
    )
    df_out = _glycofy_df_pretrain(df_out, df_mz, filename, glycopost_id, glycan_class, ms_error=ms_error,
                                  retention=retention, implicit_single_charge=implicit_single_charge,
                                  xic_mode=xic_mode)
    return df_out


def _clear_dataframe_attrs(df):
    """Remove nonessential pandas attrs before concatenation."""
    df.attrs.clear()
    return df


def data_extraction(mzml_files, glycopost_id, df_mz_total, out_path, glycan_class=1, retention=True,
                    xic_mode="score", mass_tolerance=MASS_TOLERANCE):
    """Process mzML files for data extraction"""
    if xic_mode not in XIC_MODES:
        raise ValueError(f"xic_mode must be one of {sorted(XIC_MODES)}, got {xic_mode!r}")
    os.makedirs(out_path, exist_ok=True)
    print("\n" + "=" * 60)
    print("DATA EXTRACTION - TRAINING")
    print("=" * 60)
    print(f"XIC mode: {xic_mode}")
    print(f"Mass tolerance: {mass_tolerance}")
    print(f"RT tolerance: {RT_TOLERANCE}")
    print(f"XIC m/z tolerance: {XIC_MZ_TOLERANCE}")
    print(f"XIC score margin: {XIC_SCORE_MARGIN}")
    dfs_train = []
    for k in mzml_files:
        df_mz = df_mz_total.copy()
        for j in df_mz_total.columns.tolist()[3:]:
            if j in k:
                df_mz = df_mz_total[df_mz_total[j] > 0].reset_index(drop=True)
                break

        try:
            df_train_part = _wrap_all(k, glycopost_id, df_mz, ms_error=mass_tolerance,
                                      retention=retention, xic_mode=xic_mode)
            dfs_train.append(_clear_dataframe_attrs(df_train_part))
            print(f"Processed {os.path.basename(k)} for training")
        except Exception as e:
            print(f"Failed to process {os.path.basename(k)} for training: {e}")

    if dfs_train:
        df_train = pd.concat(dfs_train).reset_index(drop=True)
        train_output = os.path.join(out_path, f'{glycopost_id}_train.xlsx')
        df_train.to_excel(train_output, index=False)
        unique_glycans = df_train['glycan'].dropna().nunique() if 'glycan' in df_train else 0
        print(f"\n Training data saved to {train_output}")
        print(f"  Number of Training datapoints: {len(df_train)}")
        print(f"  Number of unique glycans extracted: {unique_glycans}")
    else:
        print("  No training data generated")

    # For pre-training
    print("\n" + "=" * 60)
    print("DATA EXTRACTION  - PRE-TRAINING")
    print("=" * 60)
    dfs_pretrain = []
    for k in mzml_files:
        df_mz = df_mz_total.copy()
        for j in df_mz_total.columns.tolist()[3:]:
            if j in k:
                df_mz = df_mz_total[df_mz_total[j] > 0].reset_index(drop=True)
                break

        try:
            df_pretrain_part = _wrap_all_pretrain(k, glycopost_id, df_mz, glycan_class,
                                                  ms_error=mass_tolerance,
                                                  retention=retention, xic_mode=xic_mode)
            dfs_pretrain.append(_clear_dataframe_attrs(df_pretrain_part))
            print(f"✓ Processed {os.path.basename(k)} for pre-training")
        except Exception as e:
            print(f"✗ Failed to process {os.path.basename(k)} for pre-training: {e}")

    if dfs_pretrain:
        df_pretrain = pd.concat(dfs_pretrain).reset_index(drop=True)
        pretrain_output = os.path.join(out_path, f'{glycopost_id}_pretrain.xlsx')
        df_pretrain.to_excel(pretrain_output, index=False)
        print(f"\n✓ Pre-training data saved to {pretrain_output}")
        print(f"  Number of Pre-Training datapoints: {len(df_pretrain)}")
    else:
        print("  No pre-training data generated")

    print(f"\n✓ Final Report:")
    print(f"  Number of Training datapoints: {len(df_train)}")
    print(f"  Number of unique glycans extracted: {unique_glycans}")
    print(f"  Number of Pre-Training datapoints: {len(df_pretrain)}")


def main(args):
    global MASS_TOLERANCE, RT_TOLERANCE, XIC_MZ_TOLERANCE, XIC_SCORE_MARGIN
    MASS_TOLERANCE = args.mass_tolerance
    RT_TOLERANCE = args.rt_tolerance
    XIC_MZ_TOLERANCE = args.xic_mz_tolerance
    XIC_SCORE_MARGIN = args.xic_score_margin

    glco_id = args.glcopost_id
    output_dir = args.output_dir
    mapping_file = args.mapping_file
    glycan_class = args.glycan_class
    retention = args.retention
    files_downloaded = args.files_downloaded
    files_path = args.files_path
    xic_mode = args.xic_mode

    mzml_files = []

    if files_downloaded:
        print("=" * 60)
        print("STEP 1: PROCESSING LOCAL FILES")
        print("=" * 60)
        file_names = get_all_files_name_from_dir(path = files_path)
        if file_names:
            mzml_files = process_local_files(output_dir=files_path, file_names=file_names)
        else:
            print("No valid files found in the specified directory.")

    else:
        if not glco_id:
            print("Error: When files_downloaded is False, glcopost_id is required.")
            return

        print("=" * 60)
        print("STEP 1: EXTRACTING FILES FROM GLYCOPOST")
        print("=" * 60)
        file_names, glycopost_location = get_all_files_from_glycopost_api(entry_id=glco_id)
        download_base_url = f"https://glycopost.glycosmos.org/data/{urllib.parse.quote(glycopost_location, safe='')}"

        print("\n" + "=" * 60)
        print(f" SUCCESS! Found {len(file_names)} total files")
        print(f" GlycoPOST data location: {glycopost_location}")
        print("=" * 60)
        for i, f in enumerate(file_names, 1):
            print(f"{i:3}. {f}")

        print("\n" + "=" * 60)
        print("STEP 2: DOWNLOADING AND PROCESSING FILES")
        print("=" * 60)
        print("Supported formats: .raw, .mzML, .mzXML, .zip")
        print("Files will be converted to mzML format")
        mzml_files = download_and_process(
            download_base_url=download_base_url,
            output_dir=output_dir,
            file_names=file_names,
        )

    if mzml_files and mapping_file:
        print("\n" + "=" * 60)
        print(f"STEP 3: PROCESSING {len(mzml_files)} mzML FILES")
        print("=" * 60)
        df_mz_total = pd.read_csv(mapping_file)
        out_path = os.path.join("Final_Results", f'{glco_id}_results')
        os.makedirs(out_path, exist_ok=True)
        data_extraction(
            mzml_files=mzml_files,
            glycopost_id=glco_id,
            df_mz_total=df_mz_total,
            out_path=out_path,
            glycan_class=glycan_class,
            retention=retention,
            xic_mode=xic_mode,
            mass_tolerance=MASS_TOLERANCE
        )
    elif not mzml_files:
        print("\n️  No mzML files were generated. data extraction skipped.")
    elif not mapping_file:
        print("\n  No mapping file provided. data extraction skipped.")

    print("\n" + "=" * 60)
    print("ALL PROCESSING COMPLETED SUCCESSFULLY!")
    print(f"  - Processed files saved in: {output_dir}")
    if mapping_file and mzml_files:
        print(f"  - Results saved in: {os.path.join('./Final_Results', f'{glco_id}_results')}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download, convert, and process GlycoPOST files for CandyCrunch using Docker msconvert.")
    parser.add_argument("--glcopost-id", type=str, required=True,
                        help="GlycoPost ID (e.g., GPST000338)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory where processed files and results will be saved")
    parser.add_argument("--mapping-file", type=str, required=True,
                        help="Path to the mapping CSV file (df_mz_GPSTXXXXXX.csv)")
    parser.add_argument("--glycan-class", type=int, default=1,
                        help="Glycan class (0: O-linked, 1: N-linked, 2: free/lipid, 3: everything else)")
    parser.add_argument("--retention", type=_str_to_bool, default=True,
                        help="Use retention time information (True/False, default: True)")
    parser.add_argument("--files-path", type=str, required=False,
                        help="Path to directory containing downloaded files")
    parser.add_argument("--files-downloaded", type=_str_to_bool, default=False,
                        help="If .raw, mzML or mzXML has been downloaded already (True/False, default: False)")
    parser.add_argument("--xic-mode", type=str, choices=sorted(XIC_MODES), default="score",
                        help="XIC behavior: off, annotate, filter, or score (default: score)")
    parser.add_argument("--mass-tolerance", type=float, default=MASS_TOLERANCE,
                        help=f"Mass matching tolerance (default: {MASS_TOLERANCE})")
    parser.add_argument("--rt-tolerance", type=float, default=RT_TOLERANCE,
                        help=f"Retention-time matching tolerance in minutes (default: {RT_TOLERANCE})")
    parser.add_argument("--xic-mz-tolerance", type=float, default=XIC_MZ_TOLERANCE,
                        help=f"m/z window for XIC extraction (default: {XIC_MZ_TOLERANCE})")
    parser.add_argument("--xic-score-margin", type=float, default=XIC_SCORE_MARGIN,
                        help=f"Minimum score gap required to choose among multiple glycan candidates (default: {XIC_SCORE_MARGIN})")
    args = parser.parse_args()
    main(args)
