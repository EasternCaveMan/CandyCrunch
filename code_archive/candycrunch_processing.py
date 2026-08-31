import pymzml
import numpy as np
import os
import pandas as pd
from collections import defaultdict
# from candycrunch.prediction import process_mzML_stack

###NEEDS TO BE FILLED
#specify filelist
filelist = [
    '/home/daniel/Downloads/GPST000707/Feline_IgG.mzML',
    '/home/daniel/Downloads/GPST000707/Feline_Serum.mzML'
]
#specify filepath for the mapping file
df_mz_total = pd.read_csv("/home/daniel/Downloads/GPST000707/df_mz_GPST000707.csv")

#specify GlycoPOST ID
glycopost_id = 'GPST000707'

#specify filepath for saving the results
out_path = '/home/daniel/Downloads/GPST000707_results/'

# Create output directory if it doesn't exist
os.makedirs(out_path, exist_ok=True)

#specify glycan class (O-linked: 0, N-linked: 1, free/lipid: 2, everything else: 3)
glycan_class = 1

#only change this to False if you do not have retention information
retention = True



def process_mzML_stack(filepath, num_peaks=1000,
                       ms_level=2, intensity=False, extract_ms1=False):
  """function extracting all MS/MS spectra from .mzML file\n
 | Arguments:
 | :-
 | filepath (string): absolute filepath to the .mzML file
 | num_peaks (int): max number of peaks to extract from spectrum; default:1000
 | ms_level (int): which MS^n level to extract; default:2
 | intensity (bool): whether to extract precursor ion intensity from spectra; default:False
 | extract_ms1 (bool): whether to extract MS1 data for XIC area quantification; default:False\n
 | Returns:
 | :-
 | Returns a pandas dataframe of spectra with m/z, peak dictionary, retention time, charge, and intensity if True
 """
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
        # Vendor software can default to charge=1 when undetermined;
        # only trust explicit multiply-charged assignments
        if raw_charge is not None and abs(int(raw_charge)) == 1:
          raw_charge = None
        charges.append(abs(int(raw_charge)) if raw_charge is not None else None)
        if intensity:
          inty = spectrum.selected_precursors[0].get('i', np.nan)
          intensities.append(inty)
  # Sort the highest_i_dict by values
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

def match_mz(i, df_mz, df, ms_error = 0.5, retention = False, implicit_single_charge = False):
  if implicit_single_charge:
    mz = df.precursor_mass.values.tolist()[i]
  else:
    mz = df.reducing_mass.values.tolist()[i]
  idx = [k for k in range(len(df_mz.Mass.values.tolist())) if df_mz.Mass.values.tolist()[k] - ms_error < mz < df_mz.Mass.values.tolist()[k] + ms_error]
  if retention:
    rt = df.RT.values.tolist()[i]
    idx = [k for k in idx if df_mz.RT.values.tolist()[k] - 2 < rt < df_mz.RT.values.tolist()[k] + 2]
  glycans = [df_mz.glycan.values.tolist()[k] for k in idx]
  return list(set(glycans))

def glycofy_df(df, df_mz, filename, glycopost_id, ms_error = 0.5, retention = False, implicit_single_charge = False):
  df['glycan'] = [match_mz(k, df_mz, df, ms_error = ms_error,
                          retention = retention, implicit_single_charge = implicit_single_charge) for k in range(len(df))]
  idx = [k for k in range(len(df)) if len(list(set(df.glycan.values.tolist()[k]))) == 1]
  df_out = df.iloc[idx,:].reset_index(drop = True)
  df_out.glycan = [k[0] for k in df_out.glycan.values.tolist()]
  df_out = df_out.dropna(subset=['glycan']).reset_index(drop=True)
  df_out['filename'] = [filename]*len(df_out)
  df_out['GlycoPost_ID'] = [glycopost_id]*len(df_out)
  if implicit_single_charge:
    df_out.drop(['precursor_mass'], axis = 1)
  return df_out


def wrap_all(filepath, glycopost_id, df_mz, num_peaks = 1000, ms_error = 0.5, retention = False, implicit_single_charge = False):
  if 'Downloads' in filepath:
      filename = filepath.split('Downloads')[-1]
  else:
      filename = filepath
  # Remove implicit_single_charge from process_mzML_stack call
  df_out = process_mzML_stack(filepath, num_peaks = num_peaks)
  df_out = glycofy_df(df_out, df_mz, filename, glycopost_id, ms_error = ms_error,
                      retention = retention, implicit_single_charge = implicit_single_charge)
  return df_out
def match_mz_pretrain(i, df_mz, df, ms_error = 0.5, retention = False, implicit_single_charge = False):
  if implicit_single_charge:
    mz = df.precursor_mass.values.tolist()[i]
  else:
    mz = df.reducing_mass.values.tolist()[i]
  idx = [k for k in range(len(df_mz.Mass.values.tolist())) if df_mz.Mass.values.tolist()[k] - ms_error < mz < df_mz.Mass.values.tolist()[k] + ms_error]
  if retention:
    rt = df.RT.values.tolist()[i]
    idx = [k for k in idx if df_mz.RT.values.tolist()[k] - 2 < rt < df_mz.RT.values.tolist()[k] + 2]
  return len(idx)>0

def glycofy_df_pretrain(df, df_mz, filename, glycopost_id, glycan_class, ms_error = 0.5, retention = False, implicit_single_charge = False):
  df['glycan_type'] = [glycan_class if match_mz_pretrain(k, df_mz, df, ms_error = ms_error,
                          retention = retention, implicit_single_charge = implicit_single_charge) else np.nan for k in range(len(df))]
  df_out = df.dropna(subset='glycan_type').reset_index(drop=True)
  df_out['filename'] = [filename]*len(df_out)
  df_out['GlycoPost_ID'] = [glycopost_id]*len(df_out)
  if implicit_single_charge:
    df_out.drop(['precursor_mass'], axis = 1)
  return df_out



def wrap_all_pretrain(filepath, glycopost_id, df_mz, glycan_class, num_peaks = 1000, ms_error = 0.5, retention = False, implicit_single_charge = False):
  if 'Downloads' in filepath:
      filename = filepath.split('Downloads')[-1]
  else:
      filename = filepath
  # Remove implicit_single_charge from process_mzML_stack call
  df_out = process_mzML_stack(filepath, num_peaks = num_peaks)
  df_out = glycofy_df_pretrain(df_out, df_mz, filename, glycopost_id, glycan_class, ms_error = ms_error,
                      retention = retention, implicit_single_charge = implicit_single_charge)
  return df_out
#for training
dfs = []
for k in filelist:
  for j in df_mz_total.columns.tolist()[3:]:
    if j in k:
      df_mz = df_mz_total[df_mz_total[j]>0].reset_index(drop=True)
    else:
      df_mz = df_mz_total
  try:
    dfs.append(wrap_all(k, glycopost_id, df_mz, retention = retention))
  except:
    print(k)
df_train = pd.concat(dfs).reset_index(drop=True)
df_train.to_excel(out_path + glycopost_id + '_train.xlsx', index = False)
print("Number of Training datapoints: " + str(len(df_train)))



#for pretraining
dfs = []
for k in filelist:
  for j in df_mz_total.columns.tolist()[3:]:
    if j in k:
      df_mz = df_mz_total[df_mz_total[j]>0].reset_index(drop=True)
    else:
      df_mz = df_mz_total
  try:
    dfs.append(wrap_all_pretrain(k, glycopost_id, df_mz, glycan_class, retention = retention))
  except:
      print(k)
df_pretrain = pd.concat(dfs).reset_index(drop=True)
df_pretrain.to_excel(out_path + glycopost_id + '_pretrain.xlsx', index = False)
print("Number of Pre-Training datapoints: " + str(len(df_pretrain)))
###############################################################################