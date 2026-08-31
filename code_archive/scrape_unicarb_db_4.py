# import re
#
# def clean_gws(gws):
#     return re.sub(r"/#[A-Za-z]+cleavage(?:_\d+_\d+)?", "", gws)
#
#
# print(clean_gws("?1D-GalNAc,o/#bcleavage--?b1D-GlcNAc,p(--??1S)--?b1D-Gal,p--2a1L-Fuc,p/#ycleavage$MONO,Und,-H,0,redEnd"))

import h5py

import h5py

import pandas as pd

file_path = "all_glyco_corpus_2026_1gb_part1.h5"

with h5py.File(file_path, "r") as f:

    n = f["label"].shape[0]

    # Start with the simple 1D columns

    df = pd.DataFrame({

        "RT": f["RT"][:],

        "kept_peaks_n": f["kept_peaks_n"][:],

        "label": f["label"][:],

        "precursor_mz": f["precursor_mz"][:],

        "raw_peaks_n": f["raw_peaks_n"][:],

        "source_file": f["source_file"][:],

        "source_filename": f["source_filename"][:],

        "source_id": f["source_id"][:],

        "source_row": f["source_row"][:],

    })

    # Convert byte strings to normal strings, if needed

    for col in ["label", "source_file", "source_filename", "source_id"]:

        if df[col].dtype == object:

            df[col] = df[col].apply(

                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x

            )

    # Optional: add spectrum as two array-like columns

    # spectrum[i, 0, :] = m/z values

    # spectrum[i, 1, :] = intensity values

    spectrum = f["spectrum"][:]

    df["spectrum_mz"] = list(spectrum[:, 0, :])

    df["spectrum_intensity"] = list(spectrum[:, 1, :])

print(df.head())

print(df.shape)

# Save to pickle; best for preserving array columns

df.to_pickle("all_glyco_corpus_part1_dataframe.pkl")

# Optional CSV, but array columns become long strings

df.to_csv("all_glyco_corpus_part1_dataframe.csv", index=False)