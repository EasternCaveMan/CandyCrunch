from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import os
import time
import argparse
import subprocess
import pymzml
import numpy as np
import pandas as pd
from collections import defaultdict


def get_all_raw_files_universal(entry_id=None):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(options=options)
    all_files = set()

    try:
        url = f"https://glycopost.glycosmos.org/entry/{entry_id}"
        driver.get(url)

        # Wait for table to load
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        time.sleep(2)

        # Detect total number of pages
        total_pages = detect_total_pages(driver)
        print(f"📊 Detected {total_pages} total page(s)")

        # Process all pages dynamically
        current_page = 1

        while current_page <= total_pages:
            if current_page > 1:
                # Click on the page number link
                try:
                    page_link = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, str(current_page))))
                    driver.execute_script("arguments[0].scrollIntoView();", page_link)
                    time.sleep(0.5)
                    page_link.click()
                    time.sleep(2)
                    wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
                except Exception as e:
                    print(f"  Warning: Could not click page {current_page}: {e}")
                    break

            print(f"📄 Processing page {current_page}/{total_pages}...")

            # Extract files from current page
            files_on_page = extract_files_from_page(driver)
            before_count = len(all_files)
            all_files.update(files_on_page)

            print(f"  Found {len(files_on_page)} files on this page")
            print(f"  Total unique files so far: {len(all_files)} (+{len(all_files) - before_count})")

            current_page += 1

    finally:
        driver.quit()

    return sorted(list(all_files))


def detect_total_pages(driver):
    """Detect total number of pages from pagination controls"""
    try:
        status_text = driver.find_element(By.CSS_SELECTOR, ".pager_status, [class*='pager']").text
        match = re.search(r'/\s*(\d+)', status_text)
        if match:
            total_items = int(match.group(1))
            items_per_page = 20
            total_pages = (total_items + items_per_page - 1) // items_per_page
            print(f"  Detected {total_items} total items, {total_pages} pages")
            return total_pages
    except:
        pass

    try:
        page_links = driver.find_elements(By.XPATH,
                                          "//a[text()='1' or text()='2' or text()='3' or text()='4' or text()='5']")
        if page_links:
            page_numbers = []
            for link in page_links:
                try:
                    num = int(link.text)
                    page_numbers.append(num)
                except:
                    pass
            if page_numbers:
                max_page = max(page_numbers)
                print(f"  Detected {max_page} pages from page links")
                return max_page
    except:
        pass

    try:
        next_button = driver.find_element(By.XPATH, "//a[contains(text(), 'Next') or contains(text(), '>')]")
        if next_button:
            print("  Detected 'Next' button, assuming multiple pages")
            return 999
    except:
        pass

    print("  No pagination detected, assuming single page")
    return 1


def extract_files_from_page(driver):
    """Extract .raw filenames from current page"""
    files = set()
    table_rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")

    for row in table_rows:
        cells = row.find_elements(By.TAG_NAME, "td")
        if cells and len(cells) > 0:
            filename = cells[0].text.strip()
            match = re.search(r'\b(?!3A)([A-Za-z0-9][\w\-_,]+\.raw)\b', filename, re.IGNORECASE)
            if match:
                clean_filename = match.group(1)
                if not clean_filename.startswith('3A') and clean_filename.endswith('.raw'):
                    files.add(clean_filename)

    if not files:
        download_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='.raw'], a[download]")
        for link in download_links:
            href = link.get_attribute('href')
            if href and '.raw' in href.lower():
                filename = href.split('/')[-1].split('?')[0]
                filename = filename.replace('%2C', ',').replace('%2C', ',')
                if filename.endswith('.raw') and not filename.startswith('3A'):
                    files.add(filename)

    return files


def get_all_raw_files_with_next_button(entry_id=None):
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
            files_on_page = extract_files_from_page(driver)
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


def process_mzML_stack(filepath, num_peaks=1000,
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


def match_mz(i, df_mz, df, ms_error=0.5, retention=False, implicit_single_charge=False):
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


def glycofy_df(df, df_mz, filename, glycopost_id, ms_error=0.5, retention=False, implicit_single_charge=False):
    df['glycan'] = [match_mz(k, df_mz, df, ms_error=ms_error,
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


def match_mz_pretrain(i, df_mz, df, ms_error=0.5, retention=False, implicit_single_charge=False):
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


def glycofy_df_pretrain(df, df_mz, filename, glycopost_id, glycan_class, ms_error=0.5, retention=False,
                        implicit_single_charge=False):
    df['glycan_type'] = [glycan_class if match_mz_pretrain(k, df_mz, df, ms_error=ms_error,
                                                           retention=retention,
                                                           implicit_single_charge=implicit_single_charge) else np.nan
                         for k in range(len(df))]
    df_out = df.dropna(subset='glycan_type').reset_index(drop=True)
    df_out['filename'] = [filename] * len(df_out)
    df_out['GlycoPost_ID'] = [glycopost_id] * len(df_out)
    if implicit_single_charge:
        df_out.drop(['precursor_mass'], axis=1)
    return df_out


def wrap_all(filepath, glycopost_id, df_mz, num_peaks=1000, ms_error=0.5, retention=False,
             implicit_single_charge=False):
    if 'Downloads' in filepath:
        filename = filepath.split('Downloads')[-1]
    else:
        filename = filepath
    df_out = process_mzML_stack(filepath, num_peaks=num_peaks)
    df_out = glycofy_df(df_out, df_mz, filename, glycopost_id, ms_error=ms_error,
                        retention=retention, implicit_single_charge=implicit_single_charge)
    return df_out


def wrap_all_pretrain(filepath, glycopost_id, df_mz, glycan_class, num_peaks=1000, ms_error=0.5, retention=False,
                      implicit_single_charge=False):
    if 'Downloads' in filepath:
        filename = filepath.split('Downloads')[-1]
    else:
        filename = filepath
    df_out = process_mzML_stack(filepath, num_peaks=num_peaks)
    df_out = glycofy_df_pretrain(df_out, df_mz, filename, glycopost_id, glycan_class, ms_error=ms_error,
                                 retention=retention, implicit_single_charge=implicit_single_charge)
    return df_out





def convert_raw_to_mzml_docker(raw_file_path, output_dir):
    """
    Convert a RAW file to mzML using the Docker container of msconvert

    Args:
        raw_file_path: Path to the RAW file (can be relative or absolute)
        output_dir: Directory where mzML file will be saved

    Returns:
        Path to the generated mzML file or None if conversion failed
    """
    # Get absolute paths
    raw_abs_path = os.path.abspath(raw_file_path)
    output_abs_path = os.path.abspath(output_dir)

    # Get the directory containing the RAW file and the filename
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

        # The output mzML file will be in the same directory as the RAW file
        mzml_filename = raw_filename.replace('.raw', '.mzML')
        mzml_path = os.path.join(raw_dir, mzml_filename)

        # Check if the file was created
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


def down_conv_raw(base_url=None, output_dir=None, raw_names=None):
    """Download and convert RAW files to mzML using Docker msconvert"""
    mzml_files = []

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Change to output directory for downloads (but don't double-add it to paths)
    original_cwd = os.getcwd()
    os.chdir(output_dir)

    for i in raw_names:
        url = f"{base_url}/{i}"
        print(f"\n{'=' * 60}")
        print(f"Processing {i}")
        print(f"{'=' * 60}")

        # Step 1: Download the RAW file (saves to current directory which is output_dir)
        print(f"1. Downloading {i}...")
        try:
            subprocess.run(["wget", "-c", url, "-O", i], check=True)
            print(f"✓ Successfully downloaded {i}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to download {i}: {e}")
            continue

        # Get the full path to the downloaded RAW file (it's in the current directory)
        raw_file_path = os.path.abspath(i)

        # Step 2: Convert RAW to mzML using Docker msconvert
        print(f"2. Converting {i} to mzML using Docker msconvert...")
        mzml_path = convert_raw_to_mzml_docker(raw_file_path, output_dir)

        if mzml_path and os.path.exists(mzml_path):
            mzml_files.append(mzml_path)
            print(f"  ✓ mzML file saved at: {mzml_path}")
        else:
            print(f"✗ Failed to convert {i} to mzML")
            continue

        # Step 3: Remove RAW file to save space
        try:
            os.remove(raw_file_path)
            print(f"  Removed {i} to save space")
        except OSError as e:
            print(f"  Warning: Could not remove {i}: {e}")

        time.sleep(1)

    # Change back to original directory
    os.chdir(original_cwd)

    return mzml_files

def process_with_candycrunch(mzml_files, glycopost_id, df_mz_total, out_path, glycan_class=1, retention=True):
    """Process mzML files with CandyCrunch pipeline"""
    os.makedirs(out_path, exist_ok=True)

    # For training
    print("\n" + "=" * 60)
    print("CANDYCRUNCH PROCESSING - TRAINING")
    print("=" * 60)
    dfs_train = []
    for k in mzml_files:
        # Find matching glycan mapping if applicable
        df_mz = df_mz_total.copy()
        for j in df_mz_total.columns.tolist()[3:]:
            if j in k:
                df_mz = df_mz_total[df_mz_total[j] > 0].reset_index(drop=True)
                break

        try:
            dfs_train.append(wrap_all(k, glycopost_id, df_mz, retention=retention))
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
    print("CANDYCRUNCH PROCESSING - PRE-TRAINING")
    print("=" * 60)
    dfs_pretrain = []
    for k in mzml_files:
        # Find matching glycan mapping if applicable
        df_mz = df_mz_total.copy()
        for j in df_mz_total.columns.tolist()[3:]:
            if j in k:
                df_mz = df_mz_total[df_mz_total[j] > 0].reset_index(drop=True)
                break

        try:
            dfs_pretrain.append(wrap_all_pretrain(k, glycopost_id, df_mz, glycan_class, retention=retention))
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
    print("STEP 1: EXTRACTING .RAW FILES FROM GLYCOPOST")
    print("=" * 60)
    raw_names = get_all_raw_files_with_next_button(entry_id=glco_id)

    print("\n" + "=" * 60)
    print(f"✅ SUCCESS! Found {len(raw_names)} total .raw files")
    print("=" * 60)
    for i, f in enumerate(raw_names, 1):
        print(f"{i:3}. {f}")

    print("\n" + "=" * 60)
    print("STEP 2: DOWNLOADING AND CONVERTING RAW FILES USING DOCKER MSCONVERT")
    print("=" * 60)
    mzml_files = down_conv_raw(base_url=base_url, output_dir=output_dir, raw_names=raw_names)

    if mzml_files and mapping_file:
        print("\n" + "=" * 60)
        print("STEP 3: PROCESSING WITH CANDYCRUNCH")
        print("=" * 60)
        # Load mapping file
        df_mz_total = pd.read_csv(mapping_file)

        candycrunch_out_path = os.path.join("./Final_Results", f'{glco_id}_results')
        os.makedirs(candycrunch_out_path, exist_ok=True)

        # Process with CandyCrunch
        process_with_candycrunch(
            mzml_files=mzml_files,
            glycopost_id=glco_id,
            df_mz_total=df_mz_total,
            out_path=candycrunch_out_path,
            glycan_class=glycan_class,
            retention=retention
        )

    print("\n" + "=" * 60)
    print("ALL PROCESSING COMPLETED SUCCESSFULLY!")
    print(f"  - RAW files downloaded and converted to mzML in: {output_dir}")
    if mapping_file:
        print(f"  - CandyCrunch results saved in: {os.path.join('./Final_Results', f'{glco_id}_results')}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download, convert, and process GlycoPOST RAW files with CandyCrunch using Docker msconvert.")
    parser.add_argument("--base-url", type=str, required=True,
                        help="Base URL for downloading RAW files (copy link from one RAW file and remove the filename)")
    parser.add_argument("--glcopost-id", type=str, required=True,
                        help="GlycoPost ID (e.g., GPST000338)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory where mzML files and results will be saved")
    parser.add_argument("--mapping-file", type=str, required=True,
                        help="Path to the mapping CSV file (df_mz_GPSTXXXXXX.csv)")
    parser.add_argument("--glycan-class", type=int, default=1,
                        help="Glycan class (0: O-linked, 1: N-linked, 2: free/lipid, 3: everything else)")
    parser.add_argument("--retention", type=lambda x: x.lower() == 'true', default=True,
                        help="Use retention time information (True/False, default: True)")

    args = parser.parse_args()
    main(args)