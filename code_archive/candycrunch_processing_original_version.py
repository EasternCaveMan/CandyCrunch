import pymzml
import numpy as np
import pandas as pd
from collections import defaultdict


###NEEDS TO BE FILLED
#specify filelist
filelist = filelist = []
#specify filepath for the mapping file
df_mz_total = pd.read_csv()
#specify GlycoPOST ID
glycopost_id = ''
#specify filepath for saving the results
out_path = ''
#specify glycan class (O-linked: 0, N-linked: 1, free/lipid: 2, everything else: 3)
glycan_class =""
#only change this to False if you do not have retention information
retention = True

def process_mzML_stack(filepath, num_peaks = 1000, ms_level = 2, implicit_single_charge = False):
  run = pymzml.run.Reader(filepath)
  highest_i_dict = defaultdict(dict)
  number_of_peaks_to_extract = num_peaks
  rts = []
  charges = []
  prev_len = 0
  for spectrum in run:
    if spectrum.ms_level == ms_level:
      if len(spectrum.peaks("raw")) < number_of_peaks_to_extract:
        ex_num = len(spectrum.peaks("raw"))
      else:
        ex_num = number_of_peaks_to_extract
      try:
        temp = spectrum.highest_peaks(2)
        for mz, i in spectrum.highest_peaks(ex_num):
          highest_i_dict[str(spectrum.ID) + '_' + str(spectrum.selected_precursors[0]['mz'])][mz] = i
          if implicit_single_charge:
            charges.append(spectrum.selected_precursors[0]['charge'])
        if len(highest_i_dict.keys())>prev_len:
          rts.append(spectrum.scan_time_in_minutes())
          prev_len = len(highest_i_dict.keys())
      except:
        pass
  reducing_mass = [float(k.split('_')[-1]) for k in list(highest_i_dict.keys())]
  if implicit_single_charge:
    reducing_mass2 = [reducing_mass[k]*charges[k] for k in range(len(charges))]
  peak_d = list(highest_i_dict.values())
  df_out = pd.DataFrame([reducing_mass, peak_d]).T
  df_out.columns = ['reducing_mass', 'peak_d']
  df_out['RT'] = rts
  if implicit_single_charge:
    df_out['precursor_mass'] = reducing_mass2
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
  df_out = process_mzML_stack(filepath, num_peaks = num_peaks, implicit_single_charge = implicit_single_charge)
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
  df_out = process_mzML_stack(filepath, num_peaks = num_peaks, implicit_single_charge = implicit_single_charge)
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
