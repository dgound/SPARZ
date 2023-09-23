#%%
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/Users/alioutas/Library/CloudStorage/GoogleDrive-alioutas@gmail.com/My Drive/GitHub/SPARZ3/extra_code')
from h5r_functions import h5r_to_df
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import distance
from glob import glob
import seaborn as sns
import multiprocessing
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import entropy
from joblib import Parallel, delayed
from tqdm import tqdm
import glob

#%%
from scipy.stats import wasserstein_distance

# Earth Mover's Distance (EMD) or Wasserstein distance: This metric can be thought of as the minimum amount of "work" required to transform one point cloud into another, where work is measured as point movement times the distance moved.
def compute_wasserstein_distance_1D(u, v):
    u = u.values if isinstance(u, pd.DataFrame) else u
    v = v.values if isinstance(v, pd.DataFrame) else v
    
    # Compute 1D Wasserstein distance
    wd_1D = wasserstein_distance(u, v)
    
    return wd_1D

#%%
def pointcloud_to_voxels(data, voxel_size):
    """
    Convert point cloud data to a set of occupied voxels.
    
    Parameters:
    - data: DataFrame or numpy array containing x, y, z coordinates.
    - voxel_size: The size of the voxel used for binning the points.
    
    Returns:
    - A set of tuples representing the occupied voxel indices.
    """
    # Compute voxel indices for each point
    voxel_indices = (data / voxel_size).astype(int)
    
    # Convert to set for unique voxel indices
    occupied_voxels = set([tuple(row) for row in voxel_indices])
    
    return occupied_voxels


def visualize_voxels(voxel_indices_set, voxel_size=0.5, color = 'black', alpha = 0.5):
    """
    Visualize voxels in a 3D plot.
    
    Parameters:
    - voxel_indices_set: Set of tuples containing voxel indices.
    - voxel_size: The size of each voxel (cube).
    
    Returns:
    - A 3D plot showing the occupied voxels.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    for voxel in voxel_indices_set:
        # Extract the corner of the voxel based on index and voxel size
        corner = np.array(voxel) * voxel_size
        # Plot a cube at the corner position with given voxel size
        ax.bar3d(corner[0], corner[1], corner[2], 
                 voxel_size, voxel_size, voxel_size, 
                 shade=True,
                 color = color,
                 alpha=alpha)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

def visualize_voxels_two_datasets(voxel_indices_set1, voxel_indices_set2, voxel_size=0.5, alpha = 0.5):
    """
    Visualize two sets of voxels in a 3D plot.
    
    Parameters:
    - voxel_indices_set1: Set of tuples containing voxel indices for the first dataset.
    - voxel_indices_set2: Set of tuples containing voxel indices for the second dataset.
    - voxel_size: The size of each voxel (cube).
    
    Returns:
    - A 3D plot showing the occupied voxels for both datasets.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot voxels for the first dataset in magenta
    for voxel in voxel_indices_set1:
        corner = np.array(voxel) * voxel_size
        ax.bar3d(corner[0], corner[1], corner[2], 
                 voxel_size, voxel_size, voxel_size, 
                 shade=True, color='magenta',alpha=alpha)
    
    # Plot voxels for the second dataset in green
    for voxel in voxel_indices_set2:
        corner = np.array(voxel) * voxel_size
        ax.bar3d(corner[0], corner[1], corner[2], 
                 voxel_size, voxel_size, voxel_size, 
                 shade=True, color='green',alpha=alpha)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

def compute_jaccard_similarity_manual(u, v, voxel_size=0.5):
    """
    Compute the Jaccard Similarity between two sets of points using manual voxelization and alternative intersection computation.
    """
    
    # Convert dataframes to voxel sets
    set_u = pointcloud_to_voxels(u, voxel_size)
    set_v = pointcloud_to_voxels(v, voxel_size)
    
    # Compute intersection using an alternative method
    intersection_count = sum(1 for voxel in set_u if voxel in set_v)
    
    union_count = len(set_u) + len(set_v) - intersection_count
    
    if union_count == 0:
        return 0.0
    else:
        return intersection_count / union_count


# Histogram Comparison: Create histograms of distances (or other properties) for both datasets and then compare these histograms using methods like the Bhattacharyya distance or the Kullback-Leibler divergence.

def histogram_comparison(distances_raw, distances_condition, bins=30, comparison_metric="bhattacharyya"):
    """
    Compare histograms of nearest neighbor distances for raw and condition datasets.

    Parameters:
    - distances_raw: Nearest neighbor distances for the raw dataset.
    - distances_condition: Nearest neighbor distances for the condition dataset.
    - bins: Number of bins to use for the histograms.
    - comparison_metric: Either "bhattacharyya" or "kl_divergence" for Kullback-Leibler divergence.

    Returns:
    - A similarity or divergence score based on the chosen metric.
    """

    # Create histograms
    hist_raw, bin_edges = np.histogram(distances_raw, bins=bins, density=True)
    hist_condition, _ = np.histogram(distances_condition, bins=bin_edges, density=True)
    
    # Ensure histograms are normalized (sum to 1)
    hist_raw = hist_raw / np.sum(hist_raw)
    hist_condition = hist_condition / np.sum(hist_condition)
    
    # Compute Bhattacharyya distance
    if comparison_metric == "bhattacharyya":
        bc_coefficient = np.sum(np.sqrt(hist_raw * hist_condition))
        return -np.log(bc_coefficient)  # Return Bhattacharyya distance
    
    # Compute Kullback-Leibler divergence with aggressive smoothing
    elif comparison_metric == "kl_divergence":
        # Add a larger smoothing value to avoid log(0) and division by zero
        smoothing_value = 0.01
        hist_raw = (hist_raw + smoothing_value) / (1.0 + smoothing_value * len(hist_raw))
        hist_condition = (hist_condition + smoothing_value) / (1.0 + smoothing_value * len(hist_condition))
        
        kl_div = entropy(hist_raw, hist_condition)
        
        # Check and replace 'inf' with a large number (although this should now be unlikely)
        if np.isinf(kl_div):
            kl_div = 1e10
        
        return kl_div
    
    else:
        raise ValueError("Invalid comparison_metric. Choose either 'bhattacharyya' or 'kl_divergence'.")

from scipy.stats import wasserstein_distance

# Earth Mover's Distance (EMD) or Wasserstein distance: This metric can be thought of as the minimum amount of "work" required to transform one point cloud into another, where work is measured as point movement times the distance moved.
def compute_wasserstein_distance(u, v):
    u = u.values if isinstance(u, pd.DataFrame) else u
    v = v.values if isinstance(v, pd.DataFrame) else v
    
    # Compute 1D Wasserstein distance for each dimension (x, y, z) and average them
    wd_x = wasserstein_distance(u[:, 0], v[:, 0])
    wd_y = wasserstein_distance(u[:, 1], v[:, 1])
    wd_z = wasserstein_distance(u[:, 2], v[:, 2])
    
    return (wd_x + wd_y + wd_z) / 3



#%%
# Determine the number of available cores and use % of them for the NN analysis
num_cores = multiprocessing.cpu_count()
num_cores = int(round(num_cores * 0.9, 0))

print('Will be using',num_cores, 'cores')

#%%

# def compute_single_row(row, gt, n_neighbors):
#     distances = np.array([distance.euclidean(row, gt_row) for gt_row in gt.values])
#     sorted_indices = np.argsort(distances)
#     return distances[sorted_indices[:n_neighbors]], sorted_indices[:n_neighbors]

# def compute_nearest_neighbours_parallel(df, gt, n_neighbors=1, n_jobs=-1):
#     # Initialize arrays to hold the results
#     min_distances = np.zeros((df.shape[0], n_neighbors))
#     min_indices = np.zeros((df.shape[0], n_neighbors), dtype=int)

#     # Compute the minimum distances and indices for each row in df
#     results = Parallel(n_jobs=n_jobs)(delayed(compute_single_row)(row, gt, n_neighbors) for _, row in df.iterrows())

#     # Assign the results to the arrays
#     for i, (distances, indices) in enumerate(results):
#         min_distances[i] = distances
#         min_indices[i] = indices

#     return min_distances, min_indices

#%%
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

def compute_nearest_neighbours(df1, df2, n_neighbors=1, algorithm='ball_tree'):
    # Determine the number of rows in each dataframe
    len_df1 = len(df1)
    len_df2 = len(df2)

    print(f"Original lengths: df1={len_df1}, df2={len_df2}")

    # If the dataframes have different number of rows
    if len_df1 != len_df2:
        # Determine which dataframe is smaller
        if len_df1 < len_df2:
            # Pad df1 with dummy rows
            df1 = pd.concat([df1, pd.DataFrame(np.zeros((len_df2 - len_df1, df1.shape[1])), columns=df1.columns)], ignore_index=True)
        else:
            # Pad df2 with dummy rows
            df2 = pd.concat([df2, pd.DataFrame(np.zeros((len_df1 - len_df2, df2.shape[1])), columns=df2.columns)], ignore_index=True)

    print(f"Padded lengths: df1={len(df1)}, df2={len(df2)}")

    # Compute the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm).fit(df1)
    distances, indices = nbrs.kneighbors(df2)

    print(f"Distances and indices lengths: distances={len(distances)}, indices={len(indices)}")

    # If df1 was padded, remove the dummy rows from the indices
    if len_df1 < len_df2:
        indices = indices[:len_df1]
    # If df2 was padded, remove the dummy rows from the distances and indices
    elif len_df1 > len_df2:
        distances = distances[:len_df2]
        indices = indices[:len_df2]

    print(f"Final lengths: distances={len(distances)}, indices={len(indices)}")

    return distances, indices




#%%
# find all files in a directory .h5r with glob but do not include the raw files
# path = '/Users/alioutas/Dropbox (HMS)/data_compression/data_compression_localizations/Nir_et_al/' #Nir_et_al #simulated_tubulin_data
path = '/Volumes/T7/compression_data/data_compression_localizations/Nir_et_al/'
files = glob.glob(path + '**/*.h5r')
# data name for naming the output files
data_name = 'nir_etal_vs_raw'
date = '_20230922'
# find the raw files and save them seperately
raw_files = [file for file in files if 'raw_16bit' in file]
files = [file for file in files if 'raw_16bit' not in file]
# raw_files = '/Users/alioutas/Dropbox (HMS)/data_compression/Nir_et_al Laura Breimann/raw/Nir_et_al_thres-12_deb-8_syn-PSF.h5r'

# DELETE THIS
files = files[1:]

#Check if there is an output folder, and if there isnt then create one
if not os.path.exists(os.path.join(path+ '/output/')):
    os.makedirs(os.path.join(path+ '/output/'))


#%%
# load raw file
raw_dff = h5r_to_df(filepath=raw_files[0])

# Synth data filtering
# max_accuracy = 100000

# Nir et al data filtering
max_accuracy = 1000000

raw_df = raw_dff[(raw_dff["fitError_x0"] > 0) & (raw_dff["fitError_x0"] < 30) & (raw_dff["fitError_y0"] > 0) & (raw_dff["fitError_y0"] < 30) & (raw_dff["fitResults_A"] > 5) & (raw_dff["fitResults_A"] < max_accuracy)]


col_select = ['fitResults_x0',	'fitResults_y0', 'fitResults_z0']
raw = raw_df.reset_index()[col_select]
n_neighbors = 1

print('Raw file contains', raw.shape[0], 'localizations')

# make 2d plot of the localizations colored by the distance with plotly
fig = px.scatter(raw_df, x='fitResults_x0', y='fitResults_y0', opacity=0.6, title=data_name+ date+'_'+os.path.basename(raw_files[0])) #marker_size=2
fig.update_traces(marker=dict(size=1))
# save plot to file
fig.write_image(os.path.join(path+ '/output/', data_name+ date+'_'+os.path.basename(raw_files[0])+'_localizations.png'), width=800, height=800, scale=2)
fig.show()

#%%
# DF COMPRESSED create an empty dataframe to save the results
df_out = pd.DataFrame(columns=['distance', 'codec', 'label'])
df_stats_out = pd.DataFrame(columns=['metric', 'codec', 'label', 'value'])
dist_threshold = 40

# loop over the files and compute the nearest neighbours
for file in tqdm(files):
    # df_out_temp = pd.DataFrame(columns=['distance', 'codec'])
    locs = h5r_to_df(filepath=file)
    print(os.path.basename(file), "has", locs.shape[0], "localizations")
    # apply universal filter of localizations as this seems to be an issue created during localization with PyME
    locs = locs[(locs["fitError_x0"] > 0) & (locs["fitError_x0"] < 30) & (locs["fitError_y0"] > 0) & (locs["fitError_y0"] < 30) & (locs["fitResults_A"] > 5) & (locs["fitResults_A"] < max_accuracy)]
    df_query = locs.reset_index()[col_select]
    # distances, indices = compute_nearest_neighbours_parallel(df_query, raw, n_neighbors=1, n_jobs=num_cores)
    distances, indices = compute_nearest_neighbours(raw, df_query, n_neighbors=1)
    df_query['distances'] = distances.flatten()
    distances_flat = distances.flatten()
    distances_flat = distances_flat[~np.isnan(distances_flat)]
    codec_label = os.path.basename(os.path.dirname(file))
    codec_name = os.path.basename(os.path.dirname(file)) + '_' + data_name + date
    df_to_append = pd.DataFrame({'distance': distances_flat, 'codec': [codec_name]*len(distances_flat),'label': [codec_label]*len(distances_flat)})
    df_out = pd.concat([df_out, df_to_append], ignore_index=True)

    # compute the metrics
    # histogram comparison
    hist_comp = histogram_comparison(np.zeros(len(distances_flat)), distances_flat, bins=10, comparison_metric="bhattacharyya")
    hist_comp_to_append = pd.DataFrame({'metric': ['Histogram'], 'codec': [codec_name], 'value': [hist_comp],'label': [codec_label]})
    # jaccard
    jaccard = compute_jaccard_similarity_manual(df_query.iloc[:,:3].values, raw.values, voxel_size=20)
    jaccard_to_append = pd.DataFrame({'metric': ['Jaccard'], 'codec': [codec_name], 'value': [jaccard],'label': [codec_label]})
    # wasserstein
    wasserstein = compute_wasserstein_distance(df_query.iloc[:,:3], raw)
    wasserstein_to_append = pd.DataFrame({'metric': ['Wasserstein coordinates'], 'codec': [codec_name], 'value': [wasserstein],'label': [codec_label]})

    zeros = np.zeros(len(distances_flat))
    wasserstein_dist = compute_wasserstein_distance_1D(zeros, distances.flatten())
    wasserstein_dist_to_append = pd.DataFrame({'metric': ['Wasserstein NN distances'], 'codec': [codec_name], 'value': [wasserstein_dist],'label': [codec_label]})

    # append the metrics to the df
    df_stats_out = pd.concat([df_stats_out, jaccard_to_append, wasserstein_to_append, wasserstein_dist_to_append, hist_comp_to_append], ignore_index=True)

    # make 2d plot of the localizations colored by the distance with plotly
    fig = px.scatter(df_query, x='fitResults_x0', y='fitResults_y0', color='distances', color_continuous_scale='viridis', opacity=0.6, title = data_name+ date +'_'+ os.path.basename(os.path.dirname(file)))
    fig.update_traces(marker=dict(size=1))
    fig.update_layout(legend= {'itemsizing': 'constant'})
    # save plot to file
    fig.write_image(os.path.join(path+ '/output/', data_name+ date +'_'+ os.path.basename(os.path.dirname(file))+'_localizations.png'), width=800, height=800, scale=2)
    fig.show()

    # plot that colors the localizations based on the distance threshold
    df_query['distance_threshold'] = np.where(df_query['distances'] > dist_threshold, '>'+str(dist_threshold), '<'+str(dist_threshold))
    fig = px.scatter(df_query, x='fitResults_x0', y='fitResults_y0', color='distance_threshold', opacity=0.6, title = data_name+ date +'_'+ os.path.basename(os.path.dirname(file)), color_discrete_sequence=["grey", "red"])
    fig.update_traces(marker=dict(size=1))
    fig.update_layout(legend= {'itemsizing': 'constant'})
    fig.write_image(os.path.join(path+ '/output/', data_name+ date +'_'+ os.path.basename(os.path.dirname(file))+'_localizations_'+str(dist_threshold)+'_thresholded.png'), width=800, height=800, scale=2)
    fig.show()

# save df_out to csv
df_out.to_csv(os.path.join(path+ '/output/', data_name+ date+'_'+os.path.basename(os.path.dirname(file))+'_distances.csv'), index=False)
# count the number of localizations per codec
df_out.groupby('label').count().to_csv(os.path.join(path+ '/output/', data_name+ date+'_'+os.path.basename(os.path.dirname(file))+'_n_loc.csv'), index=False)

# save df_stats_out to csv
df_stats_out.to_csv(os.path.join(path+ '/output/', data_name+ date +'_'+os.path.basename(os.path.dirname(file))+'_metrics.csv'), index=False)




#%%
# DF RAW create an empty dataframe to save the results
df_out = pd.DataFrame(columns=['distance', 'codec', 'label'])
df_stats_out = pd.DataFrame(columns=['metric', 'codec','label', 'value'])
dist_threshold = 40

# loop over the files and compute the nearest neighbours
for file in tqdm(files):
    # df_out_temp = pd.DataFrame(columns=['distance', 'codec'])
    locs = h5r_to_df(filepath=file)
    # getting unique time indexes 
    # time_indexes = locs['tIndex'].unique()

    # # iterate over unique time indexes
    # for t_index in time_indexes:
    #     locs_tindex = locs[locs['tIndex']==t_index]
    #     print(f"Processing tIndex={t_index}, number of localizations: {locs_tindex.shape[0]}")

    #     # filter localizations based on selected criteria
    #     locs_tindex = locs_tindex[(locs_tindex["fitError_x0"] > 0) & (locs_tindex["fitError_x0"] < 30) & \
    #                             (locs_tindex["fitError_y0"] > 0) & (locs_tindex["fitError_y0"] < 30) & \
    #                             (locs_tindex["fitResults_A"] > 5) & (locs_tindex["fitResults_A"] < max_accuracy)]
    
    #     df_query = locs_tindex.reset_index()[col_select]

    #     # compute nearest neighbours
    #     distances, indices = compute_nearest_neighbours(df_query, raw.iloc[:, :3], n_neighbors=1)
    #     raw['distances'] = distances.flatten()
    #     distances_flat = distances.flatten()
    #     distances_flat = distances_flat[~np.isnan(distances_flat)]
    print(os.path.basename(file), "has", locs.shape[0], "localizations")
    # apply universal filter of localizations as this seems to be an issue created during localization with PyME
    locs = locs[(locs["fitError_x0"] > 0) & (locs["fitError_x0"] < 30) & (locs["fitError_y0"] > 0) & (locs["fitError_y0"] < 30) & (locs["fitResults_A"] > 5) & (locs["fitResults_A"] < max_accuracy)]
    df_query = locs.reset_index()[col_select]
    # distances, indices = compute_nearest_neighbours_parallel(df_query, raw, n_neighbors=1, n_jobs=num_cores)
    distances, indices = compute_nearest_neighbours(df_query,raw.iloc[:,:3], n_neighbors=1)
    raw['distances'] = distances.flatten()
    distances_flat = distances.flatten()
    distances_flat = distances_flat[~np.isnan(distances_flat)]
    codec_label = os.path.basename(os.path.dirname(file))
    codec_name = os.path.basename(os.path.dirname(file)) + '_' + data_name + date
    df_to_append = pd.DataFrame({'distance': distances_flat, 'codec': [codec_name]*len(distances_flat),'label': [codec_label]*len(distances_flat)})
    df_out = pd.concat([df_out, df_to_append], ignore_index=True)

    # compute the metrics
    # histogram comparison
    hist_comp = histogram_comparison(np.zeros(len(distances_flat)), distances_flat, bins=10, comparison_metric="bhattacharyya")
    hist_comp_to_append = pd.DataFrame({'metric': ['Histogram'], 'codec': [codec_name], 'value': [hist_comp],'label': [codec_label]})
    # jaccard
    jaccard = compute_jaccard_similarity_manual(raw.iloc[:,:3].values, df_query.values, voxel_size=20)
    jaccard_to_append = pd.DataFrame({'metric': ['Jaccard'], 'codec': [codec_name], 'value': [jaccard],'label': [codec_label]})
    # wasserstein
    wasserstein = compute_wasserstein_distance(raw.iloc[:,:3], df_query)
    wasserstein_to_append = pd.DataFrame({'metric': ['Wasserstein coordinates'], 'codec': [codec_name], 'value': [wasserstein],'label': [codec_label]})
    
    zeros = np.zeros(len(distances_flat))
    wasserstein_dist = compute_wasserstein_distance_1D(zeros, distances.flatten())
    wasserstein_dist_to_append = pd.DataFrame({'metric': ['Wasserstein NN distances'], 'codec': [codec_name], 'value': [wasserstein_dist],'label': [codec_label]})

    # append the metrics to the df
    df_stats_out = pd.concat([df_stats_out, jaccard_to_append, wasserstein_to_append, wasserstein_dist_to_append, hist_comp_to_append], ignore_index=True)

    # make 2d plot of the localizations colored by the distance with plotly
    fig = px.scatter(raw, x='fitResults_x0', y='fitResults_y0', color='distances', color_continuous_scale='viridis', opacity=0.6, title = data_name+ date +'_'+ os.path.basename(os.path.dirname(file)))
    fig.update_traces(marker=dict(size=1))
    fig.update_layout(legend= {'itemsizing': 'constant'})
    # save plot to file
    fig.write_image(os.path.join(path+ '/output/', data_name+ date +'_'+ os.path.basename(os.path.dirname(file))+'_localizations.png'), width=800, height=800, scale=2)
    fig.show()

    # plot that colors the localizations based on the distance threshold
    raw['distance_threshold'] = np.where(raw['distances'] > dist_threshold, '>'+str(dist_threshold), '<'+str(dist_threshold))
    fig = px.scatter(raw, x='fitResults_x0', y='fitResults_y0', color='distance_threshold', opacity=0.6, title = data_name+ date +'_'+ os.path.basename(os.path.dirname(file)), color_discrete_sequence=["grey", "red"])
    fig.update_traces(marker=dict(size=1))
    fig.update_layout(legend= {'itemsizing': 'constant'})
    fig.write_image(os.path.join(path+ '/output/', data_name+ date +'_'+ os.path.basename(os.path.dirname(file))+'_localizations_'+str(dist_threshold)+'_thresholded.png'), width=800, height=800, scale=2)
    fig.show()

# save df_out to csv
df_out.to_csv(os.path.join(path+ '/output/', data_name+ date+'_'+os.path.basename(os.path.dirname(file))+'_distances.csv'), index=False)
# count the number of localizations per codec
df_out.groupby('label').count().to_csv(os.path.join(path+ '/output/', data_name+ date+'_'+os.path.basename(os.path.dirname(file))+'_n_loc.csv'), index=False)

# save df_stats_out to csv
df_stats_out.to_csv(os.path.join(path+ '/output/', data_name+ date +'_'+os.path.basename(os.path.dirname(file))+'_metrics.csv'), index=False)



#%%
# # save df_out to csv
# df_out.to_csv(os.path.join(path+ '/output/', data_name+ date+'_'+os.path.basename(os.path.dirname(file))+'_distances.csv'), index=False)
# # count the number of localizations per codec
# df_out.groupby('label').count().to_csv(os.path.join(path+ '/output/', data_name+ date+'_'+os.path.basename(os.path.dirname(file))+'_n_loc.csv'), index=False)

# # save df_stats_out to csv
# df_stats_out.to_csv(os.path.join(path+ '/output/', data_name+ date +'_'+os.path.basename(os.path.dirname(file))+'_metrics.csv'), index=False)




#####################################################################
# read in saved dataframes
#####################################################################
#%%
import pandas as pd

df_out = pd.read_csv('/Volumes/T7/compression_data/data_compression_localizations/Nir_et_al/output/nir_etal_vs_raw_by_time_20230918_sparz_lev1_k11_rt55_distances.csv')
df_stats_out = pd.read_csv('/Volumes/T7/compression_data/data_compression_localizations/Nir_et_al/output/nir_etal_vs_raw_by_time_20230918_sparz_lev1_k11_rt55_metrics.csv')

#%%
df_out
df_stats_out
# %%
# plot kdeplot as a histogram of the distances
sns.set_theme(style="whitegrid")
df_out_new = df_out.copy()
df_out_new.distance = round(np.log(df_out_new.distance+1),10)

#%%
# sns.kdeplot(data=df_out_new, hue="codec", bw_adjust=.1, x="distance", color ="codec" , fill=False, common_norm=True, alpha=.4, linewidth=2, log_scale=False)
sns.histplot(data=df_out_new, hue="codec", x="distance", color ="codec" , fill=False, common_norm=True, alpha=.4, linewidth=2, log_scale=False)
plt.axvline(x=40, color='r', linestyle='--', linewidth=2)
# log y scale
plt.yscale('log')
# the legend should be outside the plot at the bottom and aligned with the x axis
# plt.legend(bbox_to_anchor=(0., -0.3, 1., .102), loc='lower center', ncol=3, mode="expand", borderaxespad=0.)
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# save plot
# plt.savefig(os.path.join(os.getcwd()+ '/output/', data_name+ date +'_'+os.path.basename(os.path.dirname(files[0])) + '_kdeplot.png'), dpi=300)
plt.savefig(os.path.join(path+ '/output/', data_name+ date +'_'+os.path.basename(os.path.dirname(files[0])) + '_kdeplot.png'), dpi=300)


#%%
# make a kde plot that seperates the different codecs in different subplots
g = sns.FacetGrid(df_out_new, col="codec", hue="codec", col_wrap=3, sharex=False, sharey=False)
g.map_dataframe(sns.kdeplot, x="distance", bw_adjust=.1, fill=False, common_norm=True, alpha=.4, linewidth=2, log_scale=True)
g.add_legend()

#%%
# make a histogram plot that seperates the different codecs in different subplots with matplot lib
import matplotlib.pyplot as plt
import numpy as np

# create a figure with 3 subplots and shared x and y axes
fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, tight_layout=True)



#%%
df_out.groupby('codec').count()

#%%
import seaborn as sns
import matplotlib.pyplot as plt

coded_to_use = 'x265_nir_etal_vs_raw_by_time_20230918'
filtered_data = df_out_new[df_out_new['codec'] == coded_to_use]
sns.histplot(data=filtered_data, x="distance", fill=False, common_norm=True, alpha=.4, linewidth=2, log_scale=False)
plt.axvline(x=40, color='r', linestyle='--', linewidth=2)
plt.show()

#%%



# %%
# plot the boxplot of df
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="codec", y="distance", data=df_out)
ax.set(yscale="log")
ax.set_ylabel('Distance (log)')
plt.axhline(y=40, color='r', linestyle='--', linewidth=2)
# the name of the codec should be in 45 degrees and aligned with the x axis
plt.xticks(rotation=45, ha='right')

#save plot
# plt.savefig(os.path.join(os.getcwd()+ '/output/', data_name+ date +'_' + '_boxplot.png'), dpi=300)
plt.savefig(os.path.join(path+ '/output/', data_name+date+'_'+ '_boxplot.png'), dpi=300)

plt.show()

# %%
# create dotplot of the metrics
sns.set_theme(style="whitegrid")
ax = sns.stripplot(x="metric", y="value", hue="codec", data=df_stats_out,  dodge=True, size= 15) #jitter=True,
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set(yscale="log")
ax.set_ylabel('Value (log)')
ax.set_xlabel(' ')


#%%
# create facet based on metric
g = sns.FacetGrid(df_stats_out.dropna(), col="metric", hue="codec")
g.map_dataframe(sns.scatterplot, x="codec", y="value")
g.add_legend()




# %%
import seaborn as sns

# facet grid plot
g = sns.FacetGrid(df_stats_out.dropna(), col="metric", hue="codec", sharey=False)
g.map_dataframe(sns.stripplot, x="codec", y="value")
g.add_legend()

# set the y-axis scale for each facet
g.set(yscale="log")

# set the x-axis label for each facet
# g.set_xlabels('X Values')
g.set(xticklabels=[])
g.set(xlabel=None)
# sns.set_xlabel('')

# set the y-axis label for each facet
g.set_ylabels('Value (log)')

# set the title for each facet
g.set_titles("{col_name}")

# show the plot
plt.show()
# %%
from scipy.stats import wasserstein_distance

# Earth Mover's Distance (EMD) or Wasserstein distance: This metric can be thought of as the minimum amount of "work" required to transform one point cloud into another, where work is measured as point movement times the distance moved.
def compute_wasserstein_distance_1D(u, v):
    u = u.values if isinstance(u, pd.DataFrame) else u
    v = v.values if isinstance(v, pd.DataFrame) else v
    
    # Compute 1D Wasserstein distance
    wd_1D = wasserstein_distance(u, v)
    
    return wd_1D




# %%
# create a dataframe with only 
import numpy as np
df_zeros = pd.DataFrame(np.zeros((217474, 1)), columns=['distance'])
wd_1d = df_out.groupby('codec').apply(lambda x: compute_wasserstein_distance_1D(df_zeros['distance'].values, x['distance'].values))
wd_1d = pd.DataFrame(wd_1d)
wd_1d.columns = ['value']
# %%
# create dotplot of the metrics
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
ax = sns.stripplot(x="codec", y="value", hue="codec", data=wd_1d, dodge=True, size=15)

# Remove the legend
ax.get_legend().remove()

# Tilt the x-axis labels 45 degrees
plt.xticks(rotation=45, ha='right')

ax.set(yscale="log")
ax.set_ylabel('Value (log)')
ax.set_xlabel(' ')

plt.show()

# %%
########## Histogram comparison of distances

def histogram_comparison(distances_raw, distances_condition, bins=30, comparison_metric="bhattacharyya"):
    """
    Compare histograms of nearest neighbor distances for raw and condition datasets.

    Parameters:
    - distances_raw: Nearest neighbor distances for the raw dataset.
    - distances_condition: Nearest neighbor distances for the condition dataset.
    - bins: Number of bins to use for the histograms.
    - comparison_metric: Either "bhattacharyya" or "kl_divergence" for Kullback-Leibler divergence.

    Returns:
    - A similarity or divergence score based on the chosen metric.
    """

    # Create histograms
    hist_raw, bin_edges = np.histogram(distances_raw, bins=bins, density=True)
    hist_condition, _ = np.histogram(distances_condition, bins=bin_edges, density=True)
    
    # Ensure histograms are normalized (sum to 1)
    hist_raw = hist_raw / np.sum(hist_raw)
    hist_condition = hist_condition / np.sum(hist_condition)
    
    # Compute Bhattacharyya distance
    if comparison_metric == "bhattacharyya":
        bc_coefficient = np.sum(np.sqrt(hist_raw * hist_condition))
        return -np.log(bc_coefficient)  # Return Bhattacharyya distance
    
    # Compute Kullback-Leibler divergence with aggressive smoothing
    elif comparison_metric == "kl_divergence":
        # Add a larger smoothing value to avoid log(0) and division by zero
        smoothing_value = 0.01
        hist_raw = (hist_raw + smoothing_value) / (1.0 + smoothing_value * len(hist_raw))
        hist_condition = (hist_condition + smoothing_value) / (1.0 + smoothing_value * len(hist_condition))
        
        kl_div = entropy(hist_raw, hist_condition)
        
        # Check and replace 'inf' with a large number (although this should now be unlikely)
        if np.isinf(kl_div):
            kl_div = 1e10
        
        return kl_div
    
    else:
        raise ValueError("Invalid comparison_metric. Choose either 'bhattacharyya' or 'kl_divergence'.")
# %%
import numpy as np
df_zeros = pd.DataFrame(np.zeros((217474, 1)), columns=['distance'])
hc_1d = df_out.groupby('codec').apply(lambda x: histogram_comparison(df_zeros['distance'].values, x['distance'].values, bins=30, comparison_metric="bhattacharyya"))
hc_1d = pd.DataFrame(wd_1d)
hc_1d.columns = ['value']
# %%
# create dotplot of the metrics
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
ax = sns.stripplot(x="codec", y="value", hue="codec", data=hc_1d, dodge=True, size=15)

# Remove the legend
ax.get_legend().remove()

# Tilt the x-axis labels 45 degrees
plt.xticks(rotation=45, ha='right')

ax.set(yscale="log")
ax.set_ylabel('Value (log)')
ax.set_xlabel(' ')

plt.show()
# %%
