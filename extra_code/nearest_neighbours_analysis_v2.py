#%%
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
from h5r_functions import h5r_to_df
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import distance
from glob import glob

#%%
import multiprocessing
num_cores = multiprocessing.cpu_count()
num_cores = int(round(num_cores * 0.6, 0))

print('Will be using',num_cores, 'cores')

#%%
from joblib import Parallel, delayed

def compute_single_row(row, gt, n_neighbors):
    distances = np.array([distance.euclidean(row, gt_row) for gt_row in gt.values])
    sorted_indices = np.argsort(distances)
    return distances[sorted_indices[:n_neighbors]], sorted_indices[:n_neighbors]


def compute_nearest_neighbours_parallel(df, gt, n_neighbors=1, n_jobs=-1):
    # Initialize arrays to hold the results
    min_distances = np.zeros((df.shape[0], n_neighbors))
    min_indices = np.zeros((df.shape[0], n_neighbors), dtype=int)

    # Compute the minimum distances and indices for each row in df
    results = Parallel(n_jobs=n_jobs)(delayed(compute_single_row)(row, gt, n_neighbors) for _, row in df.iterrows())

    # Assign the results to the arrays
    for i, (distances, indices) in enumerate(results):
        min_distances[i] = distances
        min_indices[i] = indices

    return min_distances, min_indices


#%%
# find all files in a directory .h5r with glob but do not include the raw files
path = '/Users/alioutas/Dropbox (HMS)/data_compression/simulated_tubulin_data Laura Breimann/'
files = glob(path + '**/*.h5r')
# find the raw files and save them seperately
raw_files = [file for file in files if 'raw' in file]
files = [file for file in files if 'raw' not in file]

# %%
# Create a loop to Load each file with h5r_to_df if it doesnt contain the word raw
# load each raw filr with h5r_to_df
# then use the compute_nearest_neighbours_parallel function to find the nearest neighbours between raw and files
# for every file save the results in a dataframe distance will be the calculated distances 
# and the name of the folder pasted to the name of the file will be saved in the codec column
# save the results in a dataframe

#%%
# load raw file
raw_df = h5r_to_df(filepath=raw_files[0])

col_select = ['fitResults_x0',	'fitResults_y0', 'fitResults_z0']
raw = raw_df.reset_index()[col_select]
n_neighbors = 1
# distances, indices = compute_nearest_neighbours(df_query, gt, n_neighbors)


#%%

# create an empty dataframe to save the results
df_out = pd.DataFrame(columns=['distance', 'codec'])
print(type(df_out))

# data name to append to the codec column
data_name = 'nir_et_al'

#%%
for file in files:
    print(file)
    # df_out_temp = pd.DataFrame(columns=['distance', 'codec'])
    locs = h5r_to_df(filepath=file)
    # apply universal filter of localizations as this seems to be an issue created during localization with PyME
    locs = locs[(locs["fitError_x0"] > 0) & (locs["fitError_x0"] < 30) & (locs["fitError_y0"] > 0) & (locs["fitError_y0"] < 30) & (locs["fitResults_A"] > 5) & (locs["fitResults_A"] < 100000)]
    df_query = locs.reset_index()[col_select]
    distances, indices = compute_nearest_neighbours_parallel(df_query, raw, n_neighbors=1, n_jobs=num_cores)
    # append to df_out the distances results to the 'distance' column
    # df_out_temp['distance'] = distances.flatten()
    # append the codec name to the 'codec' column
    # df_out_temp['codec'] = os.path.basename(os.path.dirname(file)) + '_' + data_name
    # df_out = df_out.append(df_out_temp, ignore_index=True)
    distances_flat = distances.flatten()
    codec_name = os.path.basename(os.path.dirname(file)) + '_' + data_name
    df_to_append = pd.DataFrame({'distance': distances_flat, 'codec': [codec_name]*len(distances_flat)})
    df_out = pd.concat([df_out, df_to_append], ignore_index=True)

#%%
df_out
# %%
# plot kdeplot as a histogram of the distances
import seaborn as sns
sns.set_theme(style="whitegrid")
sns.kdeplot(data=df_out, hue="codec", bw_adjust=.1, x="distance", color ="codec" , fill=False, common_norm=True, alpha=.4, linewidth=2, log_scale=True)
plt.axvline(x=40, color='r', linestyle='--', linewidth=2)
# save plot
plt.savefig(os.path.join(os.getcwd()+ '/output/', data_name+ '_kdeplot.png'), dpi=300)
# %%
# plot the boxplot of df
import seaborn as sns
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="codec", y="distance", data=df_out)
ax.set(yscale="log")
ax.set_ylabel('Distance (log)')
plt.axhline(y=40, color='r', linestyle='--', linewidth=2)
plt.show()

#save plot
plt.savefig(os.path.join(os.getcwd()+ '/output/', data_name+ '_boxplot.png'), dpi=300)

# %%

# %%
