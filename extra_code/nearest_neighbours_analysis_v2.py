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
import seaborn as sns
import multiprocessing
import plotly.express as px

#%%
# Determine the number of available cores and use 60% of them for the NN analysis
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
# data name for naming the output files
data_name = 'synthTub'
# find the raw files and save them seperately
raw_files = [file for file in files if 'raw' in file]
files = [file for file in files if 'raw' not in file]

#Check if there is an output folder, and if there isnt then create one
if not os.path.exists(os.path.join(path+ '/output/')):
    os.makedirs(os.path.join(path+ '/output/'))


#%%
# load raw file
raw_df = h5r_to_df(filepath=raw_files[0])
raw_df = raw_df[(raw_df["fitError_x0"] > 0) & (raw_df["fitError_x0"] < 30) & (raw_df["fitError_y0"] > 0) & (raw_df["fitError_y0"] < 30) & (raw_df["fitResults_A"] > 5) & (raw_df["fitResults_A"] < 100000)]

col_select = ['fitResults_x0',	'fitResults_y0', 'fitResults_z0']
raw = raw_df.reset_index()[col_select]
n_neighbors = 1

print('Raw file contains', raw.shape[0], 'localizations')

# make 2d plot of the localizations colored by the distance with plotly
fig = px.scatter(raw_df, x='fitResults_x0', y='fitResults_y0', opacity=0.6)
# save plot to file
fig.write_image(os.path.join(path+ '/output/', data_name+ '_'+os.path.basename(raw_files[0])+'_localizations.png'), width=800, height=800, scale=2)
fig.show()

#%%
# create an empty dataframe to save the results
df_out = pd.DataFrame(columns=['distance', 'codec'])

# loop over the files and compute the nearest neighbours
for file in files:
    # df_out_temp = pd.DataFrame(columns=['distance', 'codec'])
    locs = h5r_to_df(filepath=file)
    print(os.path.basename(file), "has", locs.shape[0], "localizations")
    # apply universal filter of localizations as this seems to be an issue created during localization with PyME
    locs = locs[(locs["fitError_x0"] > 0) & (locs["fitError_x0"] < 30) & (locs["fitError_y0"] > 0) & (locs["fitError_y0"] < 30) & (locs["fitResults_A"] > 5) & (locs["fitResults_A"] < 100000)]
    df_query = locs.reset_index()[col_select]
    distances, indices = compute_nearest_neighbours_parallel(df_query, raw, n_neighbors=1, n_jobs=num_cores)
    df_query['distances'] = distances.flatten()
    distances_flat = distances.flatten()
    codec_name = os.path.basename(os.path.dirname(file)) + '_' + data_name
    df_to_append = pd.DataFrame({'distance': distances_flat, 'codec': [codec_name]*len(distances_flat)})
    df_out = pd.concat([df_out, df_to_append], ignore_index=True)

    # make 2d plot of the localizations colored by the distance with plotly
    fig = px.scatter(df_query, x='fitResults_x0', y='fitResults_y0', color='distances', color_continuous_scale='viridis', opacity=0.6)
    # save plot to file
    fig.write_image(os.path.join(path+ '/output/', data_name+ '_'+ os.path.basename(os.path.dirname(file))+'_localizations.png'), width=800, height=800, scale=2)
    fig.show()

#%%
df_out
# %%
# plot kdeplot as a histogram of the distances
sns.set_theme(style="whitegrid")
sns.kdeplot(data=df_out, hue="codec", bw_adjust=.1, x="distance", color ="codec" , fill=False, common_norm=True, alpha=.4, linewidth=2, log_scale=True)
plt.axvline(x=40, color='r', linestyle='--', linewidth=2)
# save plot
plt.savefig(os.path.join(os.getcwd()+ '/output/', data_name+ '_'+os.path.basename(os.path.dirname(files[0])) + '_kdeplot.png'), dpi=300)
plt.savefig(os.path.join(path+ '/output/', data_name+ '_'+os.path.basename(os.path.dirname(files[0])) + '_kdeplot.png'), dpi=300)

#%%
fig = px.scatter(df_query, x='fitResults_x0', y='fitResults_y0', color=distances.flatten(), color_continuous_scale='viridis')
fig.update_layout(
    legend_title="Legend Title"
)

fig.show()

# %%
# plot the boxplot of df
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="codec", y="distance", data=df_out)
ax.set(yscale="log")
ax.set_ylabel('Distance (log)')
plt.axhline(y=40, color='r', linestyle='--', linewidth=2)

#save plot
plt.savefig(os.path.join(os.getcwd()+ '/output/', data_name+'_' + '_boxplot.png'), dpi=300)
plt.savefig(os.path.join(path+ '/output/', data_name+'_'+ '_boxplot.png'), dpi=300)

plt.show()

# %%
