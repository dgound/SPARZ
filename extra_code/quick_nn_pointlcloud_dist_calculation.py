#%%
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
#from h5r_functions import h5r_to_df
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import distance
from glob import glob
import seaborn as sns
import multiprocessing
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import entropy
from joblib import Parallel, delayed

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


# %%

raw = pd.read_csv('/Users/alioutas/Downloads/raw_locs.csv')

# %%
comp = pd.read_csv('/Users/alioutas/Downloads/comp_locs.csv')
# %%
distances, indices = compute_nearest_neighbours(raw.dropna(),comp.dropna(), n_neighbors=1)


# %%
# make a histogram of the distances with log scale and vertical line at 40
plt.hist(distances.flatten(), bins=100, log=True)
plt.axvline(40, color='red')
plt.show()
# %%
