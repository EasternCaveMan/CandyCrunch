import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

# Read your input CSV (adjust filename)
input_df = pd.read_excel("UniCarbMSP.xlsx")
input_df=input_df[:3]
# Store extracted data
extracted_rows = []

# Process each row
for idx, row in input_df.iterrows():
    # Clean and fix the URL
    raw_url = str(row['Link']).strip()

    # Fix common URL issues
    if 'expasy.orgmsdata' in raw_url:
        raw_url = raw_url.replace('expasy.orgmsdata', 'expasy.org/msData')
    if raw_url.startswith('http://'):
        raw_url = raw_url.replace('http://', 'https://')
    if not raw_url.startswith('https://'):
        raw_url = 'https://' + raw_url

    # Ensure proper path format
    if '/msData/' not in raw_url and '/msData' in raw_url:
        raw_url = raw_url.replace('/msData', '/msData/')

    compound_id = row['Compound ID']
    retention_time = row['Retention Time (min)']
    neutral_mass = row['Neutral Mass']
    observed_mass = row['Observed Mass']
    description = row['Description']
    formula = row['Formula']

    print(f"\nProcessing Compound {compound_id}: {raw_url}")

    try:
        # Fetch the page with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(raw_url, timeout = 30, headers = headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # --- Extract page metadata (same for all fragments from this page) ---
        reducing_mass = "Not found"
        persubstitution = "Not found"
        column = "Not found"
        method = "Not found"
        mass_spec = "Not found"

        # Find "Structure details" section - look for text patterns
        page_text = soup.get_text()

        # Extract Reducing Mass
        rm_match = re.search(r'Reducing Mass\s+([\d\.]+)', page_text)
        if rm_match:
            reducing_mass = rm_match.group(1)

        # Extract Persubstitution
        ps_match = re.search(r'Persubstitution\s+(\w+)', page_text)
        if ps_match:
            persubstitution = ps_match.group(1)

        # Extract Column
        col_match = re.search(r'Column\s+(.+?)(?:\n|$)', page_text)
        if col_match:
            column = col_match.group(1).strip()

        # Extract Method (multi-line)
        method_match = re.search(r'Method\s+(.+?)(?=Mass Spec|\n\n)', page_text, re.DOTALL)
        if method_match:
            method = method_match.group(1).strip().replace('\n', ' ')

        # Extract Mass Spec
        ms_match = re.search(r'Mass Spec\s+(.+?)(?:\n|$)', page_text)
        if ms_match:
            mass_spec = ms_match.group(1).strip()

        print(f"  Metadata extracted: Reducing Mass={reducing_mass}, Column={column[:50]}...")

        # --- Extract peak table: Peak, Intensity, Annotations ---
        peaks_found = []
        intensities_found = []
        annotations_found = []

        # Method: Parse the page more systematically
        # Find all div or pre content that might contain the peak list
        peak_pattern = re.compile(r'(\d+\.\d+|\d+)\s+(\d+\.\d+|\d+)')
        annotation_pattern = re.compile(r'Annotations:\s*\n\s*(.+?)(?=\n\s*\d|$)', re.DOTALL)

        # Look for the peak table in the page
        lines = page_text.split('\n')
        in_peak_table = False
        current_annotations = []

        for i, line in enumerate(lines):
            # Detect start of peak table
            if 'Peak' in line and 'Intensity' in line:
                in_peak_table = True
                continue

            if in_peak_table:
                # Check for peak line (number followed by number)
                peak_match = re.match(r'^\s*(\d+\.?\d*)\s+(\d+\.?\d*)\s*$', line)
                if peak_match:
                    peak = peak_match.group(1)
                    intensity = peak_match.group(2)
                    peaks_found.append(peak)
                    intensities_found.append(intensity)

                    # Store annotations for previous peak if any
                    if current_annotations:
                        annotations_found.append('; '.join(current_annotations))
                        current_annotations = []
                    else:
                        annotations_found.append('')

                # Check for annotation line
                elif 'Annotations:' in line:
                    # Look ahead for annotation text
                    if i + 1 < len(lines):
                        annotation_text = lines[i + 1].strip()
                        if annotation_text and not re.match(r'^\d', annotation_text):
                            current_annotations.append(annotation_text)

                # Stop at next major section
                elif 'Structure Format' in line or 'Biological Context' in line or 'Spectra' in line and i > 50:
                    break

        # Handle last annotations
        if current_annotations and len(annotations_found) < len(peaks_found):
            annotations_found.append('; '.join(current_annotations))
        elif len(annotations_found) < len(peaks_found):
            annotations_found.extend([''] * (len(peaks_found) - len(annotations_found)))

        # If no peaks found with first method, try alternative parsing
        if not peaks_found:
            print(f"  Warning: No peaks found with first method, trying alternative...")
            # Find all number pairs that look like peaks (m/z and intensity)
            all_numbers = re.findall(r'(\d+\.?\d*)\s+(\d+\.?\d*)', page_text)
            # Filter to likely peaks (m/z between 100-2000, intensity reasonable)
            for mz, intensity in all_numbers:
                if 100 < float(mz) < 2000 and float(intensity) > 0:
                    if mz not in peaks_found:  # Avoid duplicates
                        peaks_found.append(mz)
                        intensities_found.append(intensity)
                        annotations_found.append('')

        print(f"  Found {len(peaks_found)} peaks")

        # Create a row for each peak
        for p_idx in range(len(peaks_found)):
            extracted_rows.append({
                'Source_Compound_ID': compound_id,
                'Retention_Time': retention_time,
                'Neutral_Mass': neutral_mass,
                'Observed_Mass': observed_mass,
                'Description': description,
                'Formula': formula,
                'Reducing_Mass': reducing_mass,
                'Persubstitution': persubstitution,
                'Column': column,
                'Method': method,
                'Mass_Spec': mass_spec,
                'Peak_mz': peaks_found[p_idx],
                'Intensity': intensities_found[p_idx] if p_idx < len(intensities_found) else '',
                'Annotations': annotations_found[p_idx] if p_idx < len(annotations_found) else ''
            })

    except Exception as e:
        print(f"  Error processing {raw_url}: {e}")
        # Still add a row with error info
        extracted_rows.append({
            'Source_Compound_ID': compound_id,
            'Retention_Time': retention_time,
            'Neutral_Mass': neutral_mass,
            'Observed_Mass': observed_mass,
            'Description': description,
            'Formula': formula,
            'Error': str(e),
            'URL': raw_url
        })

    # Be polite to the server
    time.sleep(1)

# Create DataFrame and save
output_df = pd.DataFrame(extracted_rows)
output_df.to_csv('unicarb_extracted_data.csv', index = False)
print(f"\n✅ Done! Saved {len(extracted_rows)} rows to 'unicarb_extracted_data.csv'")
print(f"   Processed {input_df.shape[0]} compounds")
print(f"   Total fragment rows extracted: {len(extracted_rows)}")

# Print summary
if output_df.shape[0] > 0:
    print(f"\n📊 Summary:")
    print(f"   Columns: {list(output_df.columns)}")
    print(f"   Preview of first row:")
    print(output_df.iloc[0].to_dict())



