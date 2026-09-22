import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def pointcloud_to_voxels(coords, voxel_size):
    """
    coords: Nx2 or Nx3 numpy array, or DataFrame
    voxel_size: bin size (float)
    Returns: set of voxel (bin) indices as tuples
    """
    coords = np.asarray(coords)
    if coords.shape[1] == 2:  # XY
        indices = np.floor(coords / voxel_size).astype(int)
        return set(map(tuple, indices))
    elif coords.shape[1] == 3:  # XYZ
        indices = np.floor(coords / voxel_size).astype(int)
        return set(map(tuple, indices))
    else:
        raise ValueError('Points must be Nx2 or Nx3 array')

def compute_jaccard_similarity_manual(u, v, voxel_size=0.5):
    """
    Compute the Jaccard Similarity between two sets of points
    using manual voxelization and intersection computation.
    
    Parameters:
    - u, v: DataFrames or numpy arrays with x, y, z coordinates.
    - voxel_size: The size of the voxel used for binning the points.
    
    Returns:
    - Jaccard similarity (float)
    """
    set_u = pointcloud_to_voxels(u, voxel_size)
    set_v = pointcloud_to_voxels(v, voxel_size)
    intersection_count = len(set_u & set_v)
    union_count = len(set_u | set_v)
    return intersection_count / union_count if union_count != 0 else 0.0

def compute_nearest_neighbours(df1, df2, n_neighbors=1, algorithm='ball_tree'):
    """
    Compute nearest neighbors between two sets of points.
    
    Parameters:
    - df1, df2: DataFrames or numpy arrays.
    - n_neighbors: Number of neighbors to compute.
    - algorithm: Algorithm for NearestNeighbors.
    
    Returns:
    - distances, indices: Arrays of distances and indices.
    """
    df1 = np.asarray(df1)
    df2 = np.asarray(df2)
    len_df1 = df1.shape[0]
    len_df2 = df2.shape[0]

    # Pad the smaller array if necessary
    if len_df1 < len_df2:
        df1 = np.vstack([df1, np.zeros((len_df2 - len_df1, df1.shape[1]))])
    elif len_df2 < len_df1:
        df2 = np.vstack([df2, np.zeros((len_df1 - len_df2, df2.shape[1]))])

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm).fit(df1)
    distances, indices = nbrs.kneighbors(df2)
    
    # Remove results related to dummy rows, if any
    distances = distances[:min(len_df1, len_df2)]
    indices = indices[:min(len_df1, len_df2)]
    
    return distances, indices