#%%
import numpy as np
from h5r_functions import h5r_to_df
import sparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# functions for the analysis
def expand_coords_by_window(filtered_coords_0, filtered_coords_1):
    """Expand coordinates by a 9x9 window."""
    offsets = [(i, j) for i in range(-5, 4) for j in range(-5, 4)]
    expanded_coords = set()
    for x, y in zip(filtered_coords_0, filtered_coords_1):
        for dx, dy in offsets:
            expanded_coords.add((x + dx, y + dy))
    return expanded_coords

def vectorized_in_cutout(df, filtered_coords_0, filtered_coords_1):
    """Determine if each localization is in the cutout."""
    coords_set = expand_coords_by_window(filtered_coords_0, filtered_coords_1)
    xy_tuples = list(zip(df['fitResults_x0'].astype(int), df['fitResults_y0'].astype(int)))
    in_cutout = [1 if xy in coords_set else 0 for xy in tqdm(xy_tuples, desc="Checking localizations")]
    return in_cutout


#%%
# Load the sparse matrix from .npz file
s = sparse.load_npz('/Volumes/T7/compression_data/sparz_nir/guy_peaks_bp1.npz')
coords = s.coords

# Load localizations file
df = h5r_to_df('/Volumes/T7/compression_data/data_compression_localizations/Nir_et_al/raw/Nir_et_al_thres-12_deb-8_syn-PSF.h5r')
df = df[(df["fitError_x0"] > 0) & 
                 (df["fitError_x0"] < 30) & 
                 (df["fitError_y0"] > 0) & 
                 (df["fitError_y0"] < 30) & 
                 (df["fitResults_A"] > 5) & 
                 (df["fitResults_A"] < 100000000)]
#%%
# Select frames that are present in the df with the localizations
select_num = df['tIndex'].unique()
valid_indices = np.isin(coords[2], select_num)
filtered_coords_0 = coords[0][valid_indices]
filtered_coords_1 = coords[1][valid_indices]
filtered_data = data[valid_indices]

#%%
# Adjust x and y coordinates
df.loc[:, 'fitResults_x0'] /= 90
df.loc[:, 'fitResults_y0'] /= 90

#%%
# Determine if each localization is in the cutout
df['in_cutout'] = vectorized_in_cutout(df, filtered_coords_0, filtered_coords_1)

#%%
# Count the number of localizations in the cutout
print(df['in_cutout'].value_counts())


#%%
# Plot localizations and color by in_cutout
sns.scatterplot(data=df, x='fitResults_x0', y='fitResults_y0', hue='in_cutout', s=1, alpha=0.5)
plt.show()

#%%
# Optional: Save the 'in_cutout' column as a csv
# df['in_cutout'].to_csv('/Volumes/T7/compression_data/sparz_nir/guy_peaks_bp1_in_cutout.csv', index=False)
