from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import time
import argparse
import subprocess
import pymzml
import numpy as np
import pandas as pd
import zipfile
import shutil
import urllib.parse


def _extract_files_from_page(driver):
    """Extract files from current page - combines case-insensitive XPath with fallback methods"""
    files = set()
    allowed_extensions = {'.raw', '.mzml', '.mzxml', '.zip'}
    for ext in allowed_extensions:
        links = driver.find_elements(By.XPATH,
                                     f"//a[contains(translate(@href, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{ext}')]")
        download_links = driver.find_elements(By.CSS_SELECTOR, "a[download]")
        links.extend(download_links)

        for link in links:
            href = link.get_attribute('href')
            if href and ext.lower() in href.lower():
                filename = href.split('/')[-1].split('?')[0]
                filename = urllib.parse.unquote(filename)
                ext_lower = ext.lower()
                if ext_lower in filename.lower():
                    ext_pos = filename.lower().find(ext_lower) + len(ext_lower)
                    filename = filename[:ext_pos]
                if filename.lower().endswith(ext_lower) and not filename.startswith('3A'):
                    files.add(filename)
                    print(f"    Found via link: {filename}")

    if not files:
        table_rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        for row in table_rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if cells and len(cells) > 0:
                filename = cells[0].text.strip()

                # Check if filename directly ends with allowed extension
                for ext in allowed_extensions:
                    if filename.lower().endswith(ext):
                        files.add(filename)
                        print(f"    Found via table text: {filename}")
                        break
                if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
                    import re
                    for ext in allowed_extensions:
                        pattern = rf'\b(?!3A)([A-Za-z0-9][\w\-_,]+{ext})\b'
                        match = re.search(pattern, filename, re.IGNORECASE)
                        if match:
                            clean_filename = match.group(1)
                            if not clean_filename.startswith('3A'):
                                files.add(clean_filename)
                                print(f"    Found via pattern: {clean_filename}")
                                break

    return files


def get_all_files_with_next_button(entry_id=None):
    """Get all downloadable files from GlycoPOST using next button navigation"""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(options=options)
    all_files = set()
    page_num = 1

    try:
        url = f"https://glycopost.glycosmos.org/entry/{entry_id}"
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        time.sleep(2)

        while True:
            print(f"📄 Processing page {page_num}...")
            files_on_page = _extract_files_from_page(driver)
            all_files.update(files_on_page)

            print(f"  Found {len(files_on_page)} files on this page")
            print(f"  Total unique files so far: {len(all_files)}")
            next_found = False
            try:
                next_page_num = page_num + 1
                page_link = driver.find_element(By.XPATH, f"//a[text()='{next_page_num}']")
                if page_link and page_link.is_enabled():
                    driver.execute_script("arguments[0].scrollIntoView();", page_link)
                    time.sleep(0.5)
                    page_link.click()
                    next_found = True
                    page_num += 1
                    time.sleep(2)
                    wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
            except:
                pass
            if not next_found:
                try:
                    next_button = driver.find_element(By.XPATH,
                                                      "//a[contains(text(), 'Next') or contains(text(), '>') or contains(text(), '»')]")
                    if next_button and next_button.is_enabled():
                        driver.execute_script("arguments[0].scrollIntoView();", next_button)
                        time.sleep(0.5)
                        next_button.click()
                        next_found = True
                        page_num += 1
                        time.sleep(2)
                        wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
                except:
                    pass
            if not next_found:
                print(f"  No more pages available. Stopping at page {page_num}")
                break
            if page_num > 50:
                print("  Reached safety limit of 50 pages. Stopping.")
                break

    finally:
        driver.quit()

    return sorted(list(all_files))


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
                os.remove(file_path)
                print(f"  Removed original RAW file")
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


def download_and_process(base_url=None, output_dir=None, file_names=None):
    """Download files and process them based on their extensions"""
    all_mzml_files = []
    os.makedirs(output_dir, exist_ok=True)
    original_cwd = os.getcwd()
    os.chdir(output_dir)
    base_url = base_url.rstrip('/')
    for file_name in file_names:
        url = f"{base_url}/{file_name}"
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


def _process_mzML_stack(filepath, num_peaks=1000,
                        ms_level=2, intensity=False, extract_ms1=False):
    """function extracting all MS/MS spectra from .mzML file"""
    run = pymzml.run.Reader(filepath)
    highest_i_dict = {}
    rts, intensities, mzs, charges = [], [], [], []
    detected_mode, detected_trap = None, None
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
            if detected_mode is None:
                ns_uri = '{http://psi.hupo.org/ms/mzml}'
                for cv in spectrum.element.iter(f'{ns_uri}cvParam'):
                    acc = cv.get('accession', '')
                    if acc == 'MS:1000129':
                        detected_mode = 'negative'
                    elif acc == 'MS:1000130':
                        detected_mode = 'positive'
                    elif acc == 'MS:1000512':
                        filt = cv.get('value', '')
                        if filt.startswith('ITMS'):
                            detected_trap = 'linear'
                        elif filt.startswith('FTMS'):
                            detected_trap = 'orbitrap'
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
                charges.append(abs(int(raw_charge)) if raw_charge is not None else None)
                if intensity:
                    inty = spectrum.selected_precursors[0].get('i', np.nan)
                    intensities.append(inty)
    for key in highest_i_dict.keys():
        highest_i_dict[key] = dict(sorted(highest_i_dict[key].items(), key=lambda x: x[1], reverse=True))
    df_out = pd.DataFrame({
        'reducing_mass': mzs,
        'peak_d': list(highest_i_dict.values()),
        'RT': rts,
        'precursor_charge': charges,
    })
    if intensity:
        df_out['intensity'] = intensities
    df_out.attrs['detected_mode'] = detected_mode
    df_out.attrs['detected_trap'] = detected_trap
    if extract_ms1:
        df_out.attrs['ms1_rts'] = np.array(ms1_rts)
        df_out.attrs['ms1_scans'] = ms1_scans
    return df_out


def _match_mz(i, df_mz, df, ms_error=0.5, retention=False, implicit_single_charge=False):
    if implicit_single_charge:
        mz = df.precursor_mass.values.tolist()[i]
    else:
        mz = df.reducing_mass.values.tolist()[i]
    idx = [k for k in range(len(df_mz.Mass.values.tolist())) if
           df_mz.Mass.values.tolist()[k] - ms_error < mz < df_mz.Mass.values.tolist()[k] + ms_error]
    if retention:
        rt = df.RT.values.tolist()[i]
        idx = [k for k in idx if df_mz.RT.values.tolist()[k] - 2 < rt < df_mz.RT.values.tolist()[k] + 2]
    glycans = [df_mz.glycan.values.tolist()[k] for k in idx]
    return list(set(glycans))


def _glycofy_df(df, df_mz, filename, glycopost_id, ms_error=0.5, retention=False, implicit_single_charge=False):
    df['glycan'] = [_match_mz(k, df_mz, df, ms_error=ms_error,
                              retention=retention, implicit_single_charge=implicit_single_charge) for k in
                    range(len(df))]
    idx = [k for k in range(len(df)) if len(list(set(df.glycan.values.tolist()[k]))) == 1]
    df_out = df.iloc[idx, :].reset_index(drop=True)
    df_out.glycan = [k[0] for k in df_out.glycan.values.tolist()]
    df_out = df_out.dropna(subset=['glycan']).reset_index(drop=True)
    df_out['filename'] = [filename] * len(df_out)
    df_out['GlycoPost_ID'] = [glycopost_id] * len(df_out)
    if implicit_single_charge:
        df_out.drop(['precursor_mass'], axis=1)
    return df_out


def _match_mz_pretrain(i, df_mz, df, ms_error=0.5, retention=False, implicit_single_charge=False):
    if implicit_single_charge:
        mz = df.precursor_mass.values.tolist()[i]
    else:
        mz = df.reducing_mass.values.tolist()[i]
    idx = [k for k in range(len(df_mz.Mass.values.tolist())) if
           df_mz.Mass.values.tolist()[k] - ms_error < mz < df_mz.Mass.values.tolist()[k] + ms_error]
    if retention:
        rt = df.RT.values.tolist()[i]
        idx = [k for k in idx if df_mz.RT.values.tolist()[k] - 2 < rt < df_mz.RT.values.tolist()[k] + 2]
    return len(idx) > 0


def _glycofy_df_pretrain(df, df_mz, filename, glycopost_id, glycan_class, ms_error=0.5, retention=False,
                         implicit_single_charge=False):
    df['glycan_type'] = [glycan_class if _match_mz_pretrain(k, df_mz, df, ms_error=ms_error,
                                                            retention=retention,
                                                            implicit_single_charge=implicit_single_charge) else np.nan
                         for k in range(len(df))]
    df_out = df.dropna(subset='glycan_type').reset_index(drop=True)
    df_out['filename'] = [filename] * len(df_out)
    df_out['GlycoPost_ID'] = [glycopost_id] * len(df_out)
    if implicit_single_charge:
        df_out.drop(['precursor_mass'], axis=1)
    return df_out


def _wrap_all(filepath, glycopost_id, df_mz, num_peaks=1000, ms_error=0.5, retention=False,
              implicit_single_charge=False):
    if 'Downloads' in filepath:
        filename = filepath.split('Downloads')[-1]
    else:
        filename = os.path.basename(filepath)
    df_out = _process_mzML_stack(filepath, num_peaks=num_peaks)
    df_out = _glycofy_df(df_out, df_mz, filename, glycopost_id, ms_error=ms_error,
                         retention=retention, implicit_single_charge=implicit_single_charge)
    return df_out


def _wrap_all_pretrain(filepath, glycopost_id, df_mz, glycan_class, num_peaks=1000, ms_error=0.5, retention=False,
                       implicit_single_charge=False):
    if 'Downloads' in filepath:
        filename = filepath.split('Downloads')[-1]
    else:
        filename = os.path.basename(filepath)
    df_out = _process_mzML_stack(filepath, num_peaks=num_peaks)
    df_out = _glycofy_df_pretrain(df_out, df_mz, filename, glycopost_id, glycan_class, ms_error=ms_error,
                                  retention=retention, implicit_single_charge=implicit_single_charge)
    return df_out


def data_extraction(mzml_files, glycopost_id, df_mz_total, out_path, glycan_class=1, retention=True):
    """Process mzML files for data extraction"""
    os.makedirs(out_path, exist_ok=True)
    print("\n" + "=" * 60)
    print("DATA EXTRACTION - TRAINING")
    print("=" * 60)
    dfs_train = []
    for k in mzml_files:
        df_mz = df_mz_total.copy()
        for j in df_mz_total.columns.tolist()[3:]:
            if j in k:
                df_mz = df_mz_total[df_mz_total[j] > 0].reset_index(drop=True)
                break

        try:
            dfs_train.append(_wrap_all(k, glycopost_id, df_mz, retention=retention))
            print(f"✓ Processed {os.path.basename(k)} for training")
        except Exception as e:
            print(f"✗ Failed to process {os.path.basename(k)} for training: {e}")

    if dfs_train:
        df_train = pd.concat(dfs_train).reset_index(drop=True)
        train_output = os.path.join(out_path, f'{glycopost_id}_train.xlsx')
        df_train.to_excel(train_output, index=False)
        print(f"\n✓ Training data saved to {train_output}")
        print(f"  Number of Training datapoints: {len(df_train)}")
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
            dfs_pretrain.append(_wrap_all_pretrain(k, glycopost_id, df_mz, glycan_class, retention=retention))
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


def main(args):
    base_url = args.base_url
    glco_id = args.glcopost_id
    output_dir = args.output_dir
    mapping_file = args.mapping_file
    glycan_class = args.glycan_class
    retention = args.retention

    print("=" * 60)
    print("STEP 1: EXTRACTING FILES FROM GLYCOPOST")
    print("=" * 60)
    file_names = get_all_files_with_next_button(entry_id=glco_id)

    print("\n" + "=" * 60)
    print(f"✅ SUCCESS! Found {len(file_names)} total files")
    print("=" * 60)
    for i, f in enumerate(file_names, 1):
        print(f"{i:3}. {f}")

    print("\n" + "=" * 60)
    print("STEP 2: DOWNLOADING AND PROCESSING FILES")
    print("=" * 60)
    print("Supported formats: .raw, .mzML, .mzXML, .zip")
    print("Files will be converted to mzML format")

    mzml_files = download_and_process(base_url=base_url, output_dir=output_dir, file_names=file_names)

    if mzml_files and mapping_file:
        print("\n" + "=" * 60)
        print(f"STEP 3: PROCESSING {len(mzml_files)} mzML FILES")
        print("=" * 60)
        df_mz_total = pd.read_csv(mapping_file)
        out_path = os.path.join("./Final_Results", f'{glco_id}_results')
        os.makedirs(out_path, exist_ok=True)
        data_extraction(
            mzml_files=mzml_files,
            glycopost_id=glco_id,
            df_mz_total=df_mz_total,
            out_path=out_path,
            glycan_class=glycan_class,
            retention=retention
        )
    elif not mzml_files:
        print("\n⚠️  No mzML files were generated. data extraction skipped.")
    elif not mapping_file:
        print("\n⚠️  No mapping file provided. data extraction skipped.")

    print("\n" + "=" * 60)
    print("ALL PROCESSING COMPLETED SUCCESSFULLY!")
    print(f"  - Processed files saved in: {output_dir}")
    if mapping_file and mzml_files:
        print(f"  - Results saved in: {os.path.join('./Final_Results', f'{glco_id}_results')}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download, convert, and process GlycoPOST files for CandyCrunch using Docker msconvert.")
    parser.add_argument("--base-url", type=str, required=True,
                        help="Base URL for downloading files (copy link from one file and remove the filename)")
    parser.add_argument("--glcopost-id", type=str, required=True,
                        help="GlycoPost ID (e.g., GPST000338)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory where processed files and results will be saved")
    parser.add_argument("--mapping-file", type=str, required=True,
                        help="Path to the mapping CSV file (df_mz_GPSTXXXXXX.csv)")
    parser.add_argument("--glycan-class", type=int, default=1,
                        help="Glycan class (0: O-linked, 1: N-linked, 2: free/lipid, 3: everything else)")
    parser.add_argument("--retention", type=lambda x: x.lower() == 'true', default=True,
                        help="Use retention time information (True/False, default: True)")
    args = parser.parse_args()
    main(args)
