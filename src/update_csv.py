from pathlib import Path
import pandas as pd
import csv
import numpy as np

file_path = str(Path(__file__).resolve().parents[1] / "data/manual_review/translation_review_200_chunks.csv")

# Read the CSV
df = pd.read_csv(file_path, dtype=str)

# Find rows where is_valid is '0'
mask = df['is_valid'] == '0'

# Ganti isi kolom indonesian_text dengan teks yang ada di kolom revision
df.loc[mask, 'indonesian_text'] = df.loc[mask, 'revision']

# Ubah nilai is_valid menjadi 1
df['is_valid'] = '1'

# Kosongkan isi kolom revision dan notes
df['revision'] = np.nan
df['notes'] = np.nan

# Save back to the file with quoting to preserve the original format
df.to_csv(file_path, index=False, quoting=csv.QUOTE_ALL)

print("Data has been successfully updated!")
