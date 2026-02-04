"""
Functions for reading and processing h5r (HDF5) localization files.
"""

import h5py
import numpy as np
import pandas as pd


def h5r_to_df(filepath):
    """
    Read localization data from h5r (HDF5) file and return as pandas DataFrame.
    
    This function reads PYME/PYMEVisualize FitResults data and flattens the nested
    structure into a pandas DataFrame with separate columns for each parameter.
    
    Parameters:
    -----------
    filepath : str
        Path to h5r file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing localization data with columns:
        - tIndex: time/frame index
        - fitResults_*: fitted parameters (A, x0, y0, z0, bg, br, dx, dy)
        - fitError_*: fit errors for each parameter
        - startParams_*: initial parameters for fitting
        - slicesUsed_*: ROI information for each localization
        - resultCode: fitting result status code
        - ratio: ratio value
        - nchi2: normalized chi-square value
    """
    # Open HDF5 file and read data
    with h5py.File(filepath, 'r') as f:
        dataset = f["FitResults"]
        data = dataset[:]
    
    # Define expected data structure
    fresultdtype = [
        ('tIndex', '<i4'),
        ('fitResults', [
            ('A', '<f4'), ('x0', '<f4'), ('y0', '<f4'), ('z0', '<f4'), 
            ('bg', '<f4'), ('br', '<f4'), ('dx', '<f4'), ('dy', '<f4')
        ]),
        ('fitError', [
            ('A', '<f4'), ('x0', '<f4'), ('y0', '<f4'), ('z0', '<f4'), 
            ('bg', '<f4'), ('br', '<f4'), ('dx', '<f4'), ('dy', '<f4')
        ]),
        ('startParams', [
            ('A', '<f4'), ('x0', '<f4'), ('y0', '<f4'), ('z0', '<f4'), 
            ('bg', '<f4'), ('br', '<f4'), ('dx', '<f4'), ('dy', '<f4')
        ]),
        ('resultCode', '<i4'),
        ('slicesUsed', [
            ('x', [('start', '<i4'), ('stop', '<i4'), ('step', '<i4')]),
            ('y', [('start', '<i4'), ('stop', '<i4'), ('step', '<i4')]),
            ('x2', [('start', '<i4'), ('stop', '<i4'), ('step', '<i4')]),
            ('y2', [('start', '<i4'), ('stop', '<i4'), ('step', '<i4')])
        ]),
        ('subtractedBackground', [('g', '<f4'), ('r', '<f4')]),
        ('ratio', '<f4'),
        ('nchi2', '<f4')
    ]
    
    # Create initial DataFrame with top-level columns
    df = pd.DataFrame(data, columns=[x[0] for x in fresultdtype])
    
    # Extract nested structures into separate columns
    param_names = ['A', 'x0', 'y0', 'z0', 'bg', 'br', 'dx', 'dy']
    
    # Fit results
    df_fit_results = pd.DataFrame(
        df['fitResults'].tolist(), 
        columns=['fitResults_' + x for x in param_names]
    )
    
    # Fit errors
    df_fit_error = pd.DataFrame(
        df['fitError'].tolist(), 
        columns=['fitError_' + x for x in param_names]
    )
    
    # Start parameters
    df_start_params = pd.DataFrame(
        df['startParams'].tolist(), 
        columns=['startParams_' + x for x in param_names]
    )
    
    # Slices used (ROI information)
    slice_names = ['start', 'stop', 'step']
    df_slices_used_x = pd.DataFrame(
        df['slicesUsed'].apply(lambda x: x[0]).tolist(), 
        columns=['slicesUsed_x_' + x for x in slice_names]
    )
    df_slices_used_y = pd.DataFrame(
        df['slicesUsed'].apply(lambda x: x[1]).tolist(), 
        columns=['slicesUsed_y_' + x for x in slice_names]
    )
    df_slices_used_x2 = pd.DataFrame(
        df['slicesUsed'].apply(lambda x: x[2]).tolist(), 
        columns=['slicesUsed_x2_' + x for x in slice_names]
    )
    df_slices_used_y2 = pd.DataFrame(
        df['slicesUsed'].apply(lambda x: x[3]).tolist(), 
        columns=['slicesUsed_y2_' + x for x in slice_names]
    )
    
    # Concatenate all extracted columns with the original DataFrame
    df = pd.concat([
        df,
        df_fit_results, 
        df_fit_error, 
        df_start_params,
        df_slices_used_x, 
        df_slices_used_y, 
        df_slices_used_x2, 
        df_slices_used_y2
    ], axis=1)
    
    # Convert nested column types to strings (for compatibility)
    df['fitResults'] = df['fitResults'].astype(str)
    df['fitError'] = df['fitError'].astype(str)
    df['startParams'] = df['startParams'].astype(str)
    df['slicesUsed'] = df['slicesUsed'].astype(str)
    df['subtractedBackground'] = df['subtractedBackground'].astype(str)
    
    # Drop the original nested columns (now have flattened versions)
    df = df.drop(columns=['fitResults', 'fitError', 'startParams', 'slicesUsed'])
    
    return df


def get_xyz_coordinates(filepath):
    """
    Extract just x, y, z coordinates from h5r file.
    
    Parameters:
    -----------
    filepath : str
        Path to h5r file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['x', 'y', 'z'] containing fitted coordinates
    """
    df = h5r_to_df(filepath)
    
    # Extract coordinates and rename
    coords = df[['fitResults_x0', 'fitResults_y0', 'fitResults_z0']].copy()
    coords.columns = ['x', 'y', 'z']
    
    # Remove any invalid values
    coords = coords.replace([np.inf, -np.inf], np.nan).dropna()
    
    return coords


def filter_localizations(df, result_code=0, max_error_xy=None, max_error_z=None):
    """
    Filter localizations based on fit quality criteria.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame from h5r_to_df()
    result_code : int, optional
        Keep only localizations with this result code (0 = good fit)
    max_error_xy : float, optional
        Maximum allowed localization error in x/y (nm)
    max_error_z : float, optional
        Maximum allowed localization error in z (nm)
        
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame
    """
    filtered = df.copy()
    
    # Filter by result code
    if result_code is not None:
        filtered = filtered[filtered['resultCode'] == result_code]
    
    # Filter by xy error
    if max_error_xy is not None:
        filtered = filtered[
            (filtered['fitError_x0'] < max_error_xy) & 
            (filtered['fitError_y0'] < max_error_xy)
        ]
    
    # Filter by z error
    if max_error_z is not None:
        filtered = filtered[filtered['fitError_z0'] < max_error_z]
    
    return filtered


def get_localization_stats(filepath):
    """
    Get basic statistics about localizations in h5r file.
    
    Parameters:
    -----------
    filepath : str
        Path to h5r file
        
    Returns:
    --------
    dict
        Dictionary with statistics
    """
    df = h5r_to_df(filepath)
    
    stats = {
        'total_localizations': len(df),
        'unique_frames': df['tIndex'].nunique(),
        'mean_locs_per_frame': len(df) / df['tIndex'].nunique() if df['tIndex'].nunique() > 0 else 0,
        'x_range': (df['fitResults_x0'].min(), df['fitResults_x0'].max()),
        'y_range': (df['fitResults_y0'].min(), df['fitResults_y0'].max()),
        'z_range': (df['fitResults_z0'].min(), df['fitResults_z0'].max()),
        'mean_xy_error': (df['fitError_x0'].mean() + df['fitError_y0'].mean()) / 2,
        'mean_z_error': df['fitError_z0'].mean(),
        'result_codes': df['resultCode'].value_counts().to_dict()
    }
    
    return stats


def print_localization_stats(filepath):
    """
    Print summary statistics for an h5r file.
    
    Parameters:
    -----------
    filepath : str
        Path to h5r file
    """
    stats = get_localization_stats(filepath)
    
    print(f"\nLocalization Statistics for: {filepath}")
    print("=" * 80)
    print(f"Total localizations: {stats['total_localizations']:,}")
    print(f"Unique frames: {stats['unique_frames']:,}")
    print(f"Mean localizations per frame: {stats['mean_locs_per_frame']:.1f}")
    print(f"\nSpatial range (nm):")
    print(f"  X: {stats['x_range'][0]:.1f} to {stats['x_range'][1]:.1f}")
    print(f"  Y: {stats['y_range'][0]:.1f} to {stats['y_range'][1]:.1f}")
    print(f"  Z: {stats['z_range'][0]:.1f} to {stats['z_range'][1]:.1f}")
    print(f"\nLocalization precision:")
    print(f"  Mean XY error: {stats['mean_xy_error']:.2f} nm")
    print(f"  Mean Z error: {stats['mean_z_error']:.2f} nm")
    print(f"\nResult codes:")
    for code, count in stats['result_codes'].items():
        print(f"  Code {code}: {count:,} ({100*count/stats['total_localizations']:.1f}%)")


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print_localization_stats(filepath)
    else:
        print("Usage: python h5r_functions.py <path_to_h5r_file>")