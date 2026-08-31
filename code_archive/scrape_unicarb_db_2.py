
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

# Read your input CSV (adjust filename)
# changed to read_csv based on your snippet, switch back to read_excel if needed

input_df = pd.read_excel("UniCarbMSP_2.xlsx")
# input_df = input_df[:5]
extracted_rows = []

# Process each row
for idx, row in input_df.iterrows():
    # Clean and fix the URL
    raw_url = str(row['Link']).strip()

    if 'expasy.orgmsdata' in raw_url:
        raw_url = raw_url.replace('expasy.orgmsdata', 'expasy.org/msData')
    if raw_url.startswith('http://'):
        raw_url = raw_url.replace('http://', 'https://')
    if not raw_url.startswith('https://'):
        raw_url = 'https://' + raw_url
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
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(raw_url, timeout = 30, headers = headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # --- Extract Page Metadata ---
        page_text = soup.get_text()

        reducing_mass = "Not found"
        persubstitution = "Not found"
        column = "Not found"
        method = "Not found"
        mass_spec = "Not found"

        rm_match = re.search(r'Reducing Mass\s+([\d\.]+)', page_text)
        if rm_match: reducing_mass = rm_match.group(1)

        ps_match = re.search(r'Persubstitution\s+(\w+)', page_text)
        if ps_match: persubstitution = ps_match.group(1)

        col_match = re.search(r'Column\s+(.+?)(?:\n|$)', page_text)
        if col_match: column = col_match.group(1).strip()

        # Improved multi-line extraction for Method block
        method_match = re.search(r'Method\s*(.*?)\s*Mass Spec', page_text, re.DOTALL)
        if method_match:
            method = " ".join([line.strip() for line in method_match.group(1).split('\n') if line.strip()])

        ms_match = re.search(r'Mass Spec\s+(.+?)(?:\n|$)', page_text)
        if ms_match: mass_spec = ms_match.group(1).strip()

        # --- Robust HTML-Based Peak & Annotation Extraction ---
        peaks_found = []
        intensities_found = []
        annotations_found = []

        # Find the table containing "Peak" and "Intensity" headers
        table = None
        for t in soup.find_all('table'):
            if t.find(text = re.compile('Peak')) and t.find(text = re.compile('Intensity')):
                table = t
                break

        if table:
            current_peak = None
            current_intensity = None
            current_annotations = []

            # Step through each table row chronologically
            for tr in table.find_all('tr'):
                cells = [td.get_text(strip = True) for td in tr.find_all(['td', 'th'])]
                if not cells or 'Peak' in cells:
                    continue  # Skip empty or header rows
                if 'Annotations:' in cells[0] or any('Annotations:' in c for c in cells):
                    # 1. Grab the text annotation as you already do
                    annotation_text = tr.get_text(strip = True).replace('Annotations:', '').strip()
                    # 2. Look for the glycan structure image inside this row
                    img_tag = tr.find('img', class_ = 'sugar_image')
                    if img_tag and img_tag.get('src'):
                        from urllib.parse import urlparse, parse_qs
                        img_src = img_tag['src']
                        # Parse the URL and extract the 'structure' query parameter
                        parsed_url = urlparse(img_src)
                        query_params = parse_qs(parsed_url.query)
                        # if 'structure' exists in the URL, extract it
                        if 'structure' in query_params:
                            iupac_structure = query_params['structure'][0]
                            # Combine the text annotation and the IUPAC string
                            annotation_text = f"{annotation_text} [{iupac_structure}]"
                    if annotation_text:
                        current_annotations.append(annotation_text)

                # Condition B: It is a standard numerical Peak/Intensity row
                elif len(cells) >= 2 and re.match(r'^\d', cells[0]):
                    # Before moving to the next peak, save the accumulated data of the prior peak
                    if current_peak is not None:
                        peaks_found.append(current_peak)
                        intensities_found.append(current_intensity)
                        annotations_found.append(" | ".join(current_annotations))

                    # Reset variables for the new peak
                    current_peak = cells[0]
                    current_intensity = cells[1]
                    current_annotations = []

            # Append the absolute last peak in the table loop
            if current_peak is not None:
                peaks_found.append(current_peak)
                intensities_found.append(current_intensity)
                annotations_found.append(" | ".join(current_annotations))

        # Fallback to text parsing only if HTML structure completely failed
        if not peaks_found:
            print("  ⚠️ HTML table parsing yielded no peaks. Resorting to text alignment...")
            lines = [line.strip() for line in page_text.split('\n') if line.strip()]

            i = 0
            while i < len(lines):
                # Look for a Peak value followed by an Intensity value
                if re.match(r'^\d+\.\d+$', lines[i]) and i + 1 < len(lines) and re.match(r'^\d+\.\d+$', lines[i + 1]):
                    peaks_found.append(lines[i])
                    intensities_found.append(lines[i + 1])

                    # Look ahead to see if the next elements contain an annotation
                    annotations = []
                    forward_idx = i + 2
                    while forward_idx < len(lines):
                        if 'Annotations:' in lines[forward_idx]:
                            if forward_idx + 1 < len(lines) and not re.match(r'^\d', lines[forward_idx + 1]):
                                annotations.append(lines[forward_idx + 1])
                            forward_idx += 2
                        elif re.match(r'^\d+\.\d+$', lines[forward_idx]):
                            # Reached next peak structure, stop looking ahead
                            break
                        else:
                            forward_idx += 1

                    annotations_found.append(" | ".join(annotations))
                    i = forward_idx
                else:
                    i += 1

        print(f"  Found {len(peaks_found)} peaks mapped safely with annotations.")

        # --- Compile into individual rows ---
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
                'Intensity': intensities_found[p_idx],
                'Annotations': annotations_found[p_idx]
            })

    except Exception as e:
        print(f"  Error processing {raw_url}: {e}")
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

    time.sleep(1)

# Export Data
output_df = pd.DataFrame(extracted_rows)
output_df.to_csv('unicarb_extracted_data.csv', index = False)
print(f"\n✅ Done! Saved {len(extracted_rows)} entries to 'unicarb_extracted_data.csv'")