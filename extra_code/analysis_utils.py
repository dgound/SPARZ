"""
Utility functions for SPARZ compression analysis and figure generation.
"""

import os
import re
import numpy as np
import cv2
import tifffile
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from natsort import natsorted


# ============================================================================
# Image Processing Utilities
# ============================================================================

def normalize_image_16bit_to_8bit(img, min_val, max_val):
    """
    Normalize 16-bit image to 8-bit based on provided min/max values.
    
    Parameters:
    -----------
    img : np.ndarray
        Input image
    min_val : int
        Minimum value for normalization
    max_val : int
        Maximum value for normalization
        
    Returns:
    --------
    np.ndarray
        Normalized 8-bit image
    """
    img_16bit = img.astype(np.uint16)
    img_norm = (img_16bit - min_val) / (max_val - min_val) * 65535
    img_norm = np.clip(img_norm, 0, 65535)
    img_norm_8bit = (img_norm / 256).astype(np.uint8)
    return img_norm_8bit


def read_tiff_frame(image_path, frame_number):
    """
    Read a specific frame from a multi-frame TIFF file.
    
    Parameters:
    -----------
    image_path : str
        Path to TIFF file
    frame_number : int
        Frame index to extract
        
    Returns:
    --------
    np.ndarray
        Grayscale image frame
    """
    with Image.open(image_path) as img:
        img.seek(frame_number)
        frame = np.array(img)
        if frame.ndim == 2:
            return frame
        else:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


# ============================================================================
# Point Cloud and Similarity Metrics
# ============================================================================

def pointcloud_to_voxels(data, voxel_size):
    """
    Convert point cloud data to a set of occupied voxels.
    
    Parameters:
    -----------
    data : pd.DataFrame or np.ndarray
        Point cloud with x, y, z coordinates
    voxel_size : float
        Size of voxel for binning
        
    Returns:
    --------
    set
        Set of tuples representing occupied voxel indices
    """
    voxel_indices = (data / voxel_size).astype(int)
    occupied_voxels = set(map(tuple, voxel_indices.values))
    return occupied_voxels


def compute_jaccard_similarity_manual(u, v, voxel_size=0.5):
    """
    Compute Jaccard Similarity between two point clouds using voxelization.
    
    Parameters:
    -----------
    u, v : pd.DataFrame
        Point clouds to compare
    voxel_size : float, optional
        Voxel size for discretization
        
    Returns:
    --------
    float
        Jaccard similarity score
    """
    set_u = pointcloud_to_voxels(u, voxel_size)
    set_v = pointcloud_to_voxels(v, voxel_size)
    
    intersection_count = len(set_u.intersection(set_v))
    union_count = len(set_u.union(set_v))
    
    return intersection_count / union_count if union_count != 0 else 0.0

def calculate_jaccard_for_condition(raw_locs, compressed_locs, voxel_size=0.5):
    """
    Calculate 3D Jaccard similarity between raw and compressed localizations.
    
    Parameters:
    -----------
    raw_locs : pd.DataFrame
        Raw localizations with x, y, z columns
    compressed_locs : pd.DataFrame
        Compressed localizations with x, y, z columns
    voxel_size : float, optional
        Voxel size for discretization (in nm)
        
    Returns:
    --------
    float
        Jaccard similarity score
    """
    print(f"  Raw localizations: {len(raw_locs)}")
    print(f"  Compressed localizations: {len(compressed_locs)}")
    
    # Calculate Jaccard similarity
    jaccard = compute_jaccard_similarity_manual(raw_locs, compressed_locs, voxel_size=voxel_size)
    
    return jaccard

def compute_nearest_neighbours(df1, df2, n_neighbors=1, algorithm='ball_tree'):
    """
    Compute nearest neighbors between two point clouds with automatic padding.
    
    Parameters:
    -----------
    df1, df2 : pd.DataFrame
        Point clouds to compare
    n_neighbors : int, optional
        Number of neighbors to find
    algorithm : str, optional
        Algorithm for neighbor search
        
    Returns:
    --------
    tuple
        (distances, indices) arrays
    """
    len_df1 = len(df1)
    len_df2 = len(df2)

    print(f"Original lengths: df1={len_df1}, df2={len_df2}")

    # Pad dataframes to equal length
    if len_df1 != len_df2:
        if len_df1 < len_df2:
            df1 = pd.concat([df1, pd.DataFrame(np.zeros((len_df2 - len_df1, df1.shape[1])), 
                                               columns=df1.columns)], ignore_index=True)
        else:
            df2 = pd.concat([df2, pd.DataFrame(np.zeros((len_df1 - len_df2, df2.shape[1])), 
                                               columns=df2.columns)], ignore_index=True)

    # Compute nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm).fit(df1)
    distances, indices = nbrs.kneighbors(df2)

    # Remove padding from results
    if len_df1 < len_df2:
        indices = indices[:len_df1]
    elif len_df1 > len_df2:
        distances = distances[:len_df2]
        indices = indices[:len_df2]

    print(f"Final lengths: distances={len(distances)}, indices={len(indices)}")
    
    return distances, indices


# ============================================================================
# SSIM Calculation
# ============================================================================

def calculate_ssim(original_file, compressed_file):
    """
    Calculate median SSIM between two TIFF files.
    
    Parameters:
    -----------
    original_file : str
        Path to original TIFF file
    compressed_file : str
        Path to compressed TIFF file
        
    Returns:
    --------
    float
        Median SSIM score across all frames
    """
    ssim_scores = []
    with tifffile.TiffFile(original_file) as tif_original:
        with tifffile.TiffFile(compressed_file) as tif_compressed:
            for original_frame, compressed_frame in zip(tif_original.series[0].pages, 
                                                        tif_compressed.series[0].pages):
                original_image = original_frame.asarray()
                compressed_image = compressed_frame.asarray()
                score = ssim(original_image, compressed_image, 
                           data_range=compressed_image.max() - compressed_image.min())
                ssim_scores.append(score)
    return np.median(ssim_scores)


def calculate_ssim_folder(original_folder, compressed_folder):
    """
    Calculate SSIM for all TIFF files in two folders.
    
    Parameters:
    -----------
    original_folder : str
        Path to folder with original files
    compressed_folder : str
        Path to folder with compressed files
        
    Returns:
    --------
    list
        List of dictionaries with SSIM results
    """
    ssim_results = []
    original_files = natsorted([f for f in os.listdir(original_folder) 
                               if f.endswith('.tiff') and not f.startswith('.')])
    compressed_files = natsorted([f for f in os.listdir(compressed_folder) 
                                 if f.endswith('.tiff') and not f.startswith('.')])
    
    for original_file, compressed_file in zip(original_files, compressed_files):
        ssim_scores = []
        try:
            with tifffile.TiffFile(os.path.join(original_folder, original_file)) as tif_original:
                with tifffile.TiffFile(os.path.join(compressed_folder, compressed_file)) as tif_compressed:
                    for original_frame, compressed_frame in zip(tif_original.series[0].pages, 
                                                                tif_compressed.series[0].pages):
                        original_image = original_frame.asarray()
                        compressed_image = compressed_frame.asarray()
                        score = ssim(original_image, compressed_image, 
                                   data_range=compressed_image.max() - compressed_image.min())
                        ssim_scores.append(score)
            if ssim_scores:
                median_ssim = np.median(ssim_scores)
                ssim_results.append({
                    'original_file': original_file, 
                    'compressed_file': compressed_file, 
                    'ssim': median_ssim
                })
        except Exception as e:
            print(f"Error processing files {original_file} and {compressed_file}: {e}")
    
    return ssim_results

def calculate_ssim_biplane(original_folder, compressed_folder, bp_positive_pattern, bp_negative_pattern):
    """
    Calculate SSIM for biplane data with two separate image files.
    
    Parameters:
    -----------
    original_folder : str
        Path to folder with original files
    compressed_folder : str
        Path to folder with compressed files
    bp_positive_pattern : str
        Pattern to match positive biplane file (e.g., 'BP+250')
    bp_negative_pattern : str
        Pattern to match negative biplane file (e.g., 'BP-250')
        
    Returns:
    --------
    dict
        Dictionary with SSIM results for both biplanes and median
    """
    import tifffile
    from skimage.metrics import structural_similarity as ssim
    import numpy as np
    import os
    
    # Find the files
    orig_files = [f for f in os.listdir(original_folder) if f.endswith('.tif') or f.endswith('.tiff')]
    comp_files = [f for f in os.listdir(compressed_folder) if f.endswith('.tif') or f.endswith('.tiff')]
    
    # Match biplane files
    orig_bp_pos = [f for f in orig_files if bp_positive_pattern in f][0]
    orig_bp_neg = [f for f in orig_files if bp_negative_pattern in f][0]
    
    comp_bp_pos = [f for f in comp_files if bp_positive_pattern in f][0]
    comp_bp_neg = [f for f in comp_files if bp_negative_pattern in f][0]
    
    results = {}
    
    # Calculate SSIM for BP+250
    ssim_scores_pos = []
    with tifffile.TiffFile(os.path.join(original_folder, orig_bp_pos)) as tif_orig:
        with tifffile.TiffFile(os.path.join(compressed_folder, comp_bp_pos)) as tif_comp:
            for orig_frame, comp_frame in zip(tif_orig.series[0].pages, tif_comp.series[0].pages):
                orig_img = orig_frame.asarray()
                comp_img = comp_frame.asarray()
                score = ssim(orig_img, comp_img, data_range=comp_img.max() - comp_img.min())
                ssim_scores_pos.append(score)
    
    # Calculate SSIM for BP-250
    ssim_scores_neg = []
    with tifffile.TiffFile(os.path.join(original_folder, orig_bp_neg)) as tif_orig:
        with tifffile.TiffFile(os.path.join(compressed_folder, comp_bp_neg)) as tif_comp:
            for orig_frame, comp_frame in zip(tif_orig.series[0].pages, tif_comp.series[0].pages):
                orig_img = orig_frame.asarray()
                comp_img = comp_frame.asarray()
                score = ssim(orig_img, comp_img, data_range=comp_img.max() - comp_img.min())
                ssim_scores_neg.append(score)
    
    results['bp_positive_file'] = comp_bp_pos
    results['bp_negative_file'] = comp_bp_neg
    results['ssim_bp_positive'] = np.median(ssim_scores_pos)
    results['ssim_bp_negative'] = np.median(ssim_scores_neg)
    # Use median of the two median values
    results['ssim_median'] = np.median([results['ssim_bp_positive'], results['ssim_bp_negative']])
    
    return results


# ============================================================================
# File and Data Utilities
# ============================================================================

def get_total_size_of_files(folder_path, file_extensions):
    """
    Calculate total size of files with specific extensions in a folder.
    
    Parameters:
    -----------
    folder_path : str
        Path to folder
    file_extensions : list
        List of file extensions to include (e.g., ['.mp4', '.npz'])
        
    Returns:
    --------
    int
        Total size in bytes
    """
    total_size = 0
    for file in os.listdir(folder_path):
        if any(file.endswith(ext) for ext in file_extensions):
            total_size += os.path.getsize(os.path.join(folder_path, file))
    return total_size


def natural_sort_key(s):
    """
    Generate a key for natural sorting of strings with numbers.
    
    Parameters:
    -----------
    s : str
        String to generate key for
        
    Returns:
    --------
    list
        Sort key
    """
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', s)]


# ============================================================================
# Parameter Extraction
# ============================================================================

def extract_k_lev(label):
    """
    Extract kernel size and compression level from folder label.
    
    Parameters:
    -----------
    label : str
        Folder name (e.g., 'sparz_k5_rt35_lev0')
        
    Returns:
    --------
    tuple
        (kernel_size, compression_level)
    """
    k_match = re.search(r'_k(\d+)_', label)
    lev_match = re.search(r'_lev(\d+)', label)
    k_value = int(k_match.group(1)) if k_match else None
    lev_value = int(lev_match.group(1)) if lev_match else None
    return k_value, lev_value


def extract_parameters(folder_name):
    """
    Extract all parameters from folder name.
    
    Parameters:
    -----------
    folder_name : str
        Folder name with parameters
        
    Returns:
    --------
    tuple
        (kernel_size, rel_threshold, compression_level)
    """
    match = re.search(r'k(\d+)_rt(\d+)_lev(\d+)', folder_name)
    if match:
        kernel_size = int(match.group(1))
        rel_threshold = float(f"0.{match.group(2)}")
        compression_level = int(match.group(3))
        return kernel_size, rel_threshold, compression_level
    else:
        return None, None, None
    

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    
    Parameters:
    -----------
    color : matplotlib color or array
        Color to lighten
    amount : float
        Amount to lighten (0=no change, 1=white)
        
    Returns:
    --------
    tuple
        RGB color tuple
    """
    import matplotlib.colors as mc
    import colorsys
    import numpy as np
    
    # Handle numpy arrays
    if isinstance(color, np.ndarray):
        color = tuple(color)
    
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])    


# ============================================================================
# SPARZ Data Reconstruction
# ============================================================================

def reconstruct_array(data):
    """
    Reconstruct full array from sparse NPZ data.
    
    Parameters:
    -----------
    data : dict
        NPZ file data with 'shape', 'fill_value', 'coords', 'data' keys
        
    Returns:
    --------
    np.ndarray
        Reconstructed full array
    """
    shape = tuple(data['shape'])
    fill_value = data['fill_value']
    coords = data['coords']
    sparse_data = data['data']

    full_array = np.full(shape, fill_value)
    for i, coord in enumerate(zip(*coords)):
        full_array[coord] = sparse_data[i]
    
    return full_array


# ============================================================================
# Grid Search Analysis
# ============================================================================

def analyze_grid_search_results(main_folder, original_files, file_extensions=['.mp4', '.npz']):
    """
    Analyze grid search results and calculate SSIM and compression ratios.
    
    Parameters:
    -----------
    main_folder : str
        Path to main folder containing grid search results
    original_files : str
        Path to folder with original files
    file_extensions : list, optional
        File extensions to include in size calculation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with analysis results
    """
    original_file_size = get_total_size_of_files(original_files, ['.tiff'])
    print(f'Original file size: {original_file_size / 1e6:.2f} MB')
    
    ssim_df_list = []
    for folder in natsorted(os.listdir(main_folder)):
        if folder.startswith('.'):
            continue
            
        print(f"Processing: {folder}")
        uncompressed_folder = os.path.join(main_folder, folder, 'uncompressed')
        
        if not os.path.exists(uncompressed_folder):
            continue
        
        ssim_results = calculate_ssim_folder(original_files, uncompressed_folder)
        total_file_size = get_total_size_of_files(os.path.join(main_folder, folder), 
                                                  file_extensions)
        file_size_percentage = (total_file_size / original_file_size) * 100
        
        k, lev = extract_k_lev(folder)
        threshold_match = re.search(r'rt(\d+)', folder)
        threshold = int(threshold_match.group(1)) if threshold_match else None
        
        for result in ssim_results:
            ssim_df_list.append({
                'Folder': folder,
                'original_file': result['original_file'],
                'compressed_file': result['compressed_file'],
                'SSIM': result['ssim'],
                'Size': total_file_size,
                'percentage_size': file_size_percentage,
                'k': k,
                'lev': lev,
                'threshold': threshold
            })
    
    return pd.DataFrame(ssim_df_list)