import os
import numpy as np
import imageio
import tifffile
from natsort import natsorted

def get_all_slices_from_folder(input_folder):
    """
    Load all 3D stacks from the specified folder and concatenate them slice-wise.
    
    Args:
    - input_folder (str): Path to the folder containing the 3D stacks.
    
    Returns:
    - np.array: Concatenated 2D slices from all stacks.
    """
    # Get the list of image files in the input folder, excluding system files
    stack_files = natsorted(file for file in os.listdir(input_folder) if not file.startswith('.'))
    print(stack_files)
    
    all_slices = []
    for stack_file in stack_files:
        stack_path = os.path.join(input_folder, stack_file)
        stack = imageio.volread(stack_path)
        all_slices.extend(stack)
        
    return np.array(all_slices, ) #remove dtype dtype=np.uint16

def save_stack_as_tiff(stack, output_path):
    """
    Save the provided stack as a TIFF file.
    
    Args:
    - stack (np.array): 5D array containing the image data.
    - output_path (str): Path where the TIFF file will be saved.
    """
    try:
        tifffile.imwrite(output_path, stack, dtype=np.uint16, bigtiff=True, imagej=True)
        print(f"TIFF stack saved as '{output_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Define input and output folders
    input_folder = '/Users/laurabreimann/Desktop/input_data/nir_etal_ROI/level_3_kernel_11_relThres_55/biplane'
    output_folder = '/Users/laurabreimann/Desktop/input_data/nir_etal_ROI/level_3_kernel_11_relThres_55'
    output_filename = 'Nir_et_al.tif'
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get concatenated slices from the folder
    all_slices = get_all_slices_from_folder(input_folder)
    
    # Reshape the slices to have shape (T, Z, Y, X, C)
    final_stack_shape = (len(all_slices), 1, all_slices.shape[1], all_slices.shape[2], 1)
    all_slices_reshaped = all_slices.reshape(final_stack_shape)

    # Save the reshaped slices as a TIFF file
    output_tiff_path = os.path.join(output_folder, output_filename)
    save_stack_as_tiff(all_slices_reshaped, output_tiff_path)
    
    # Print the dimensions
    print("Stack dimensions:", all_slices_reshaped.shape)
