import os
import numpy as np
import imageio
import tifffile
from natsort import natsorted
from tifffile import TiffFile  # Add this import statement for TiffFile

def load_volume_from_file(folder, filename):
    """
    Load a 3D volume from a TIFF file.

    Parameters:
        folder (str): The directory containing the TIFF file.
        filename (str): The name of the TIFF file.

    Returns:
        np.array: The 3D volume loaded from the TIFF file.
    """
    with TiffFile(os.path.join(folder, filename)) as tif:
        slices = [page.asarray() for page in tif.pages]
        volume = np.stack(slices, axis=0)
        print(f"Loaded volume from {filename}, shape: {volume.shape}")
    return volume

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
    print(f"Total stack files: {len(stack_files)}")

    all_slices = []
    for stack_file in stack_files:
        stack_path = os.path.join(input_folder, stack_file)
        # Load each stack as a 3D volume using load_volume_from_file function
        volume = load_volume_from_file(input_folder, stack_file)
        # Extract slices from the volume and append them to all_slices
        for slice_index in range(volume.shape[0]):
            all_slices.append(volume[slice_index])  # Append each slice
        print(f"Loaded stack '{stack_file}' with {volume.shape[0]} slices")

    print(f"Total slices loaded: {len(all_slices)}")

    return np.array(all_slices)

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
    input_folder = '/Users/laurabreimann/Desktop/figure_1/nir_et_al/zstd_level_0/decompressed/BP2'
    output_folder = '/Users/laurabreimann/Desktop/figure_1/nir_et_al/zstd_level_0/decompressed'
    output_filename = 'Nir_et_al_bp2.tif'
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get concatenated slices from the folder using load_volume_from_file function
    all_slices = get_all_slices_from_folder(input_folder)

    # Print the shape of all_slices before reshaping
    print("Original stack shape:", all_slices.shape)

    # Reshape the slices if needed
    num_slices = len(all_slices)
    slice_shape = all_slices[0].shape
    final_stack_shape = (num_slices,) + slice_shape
    final_stack = np.reshape(all_slices, final_stack_shape)

    # Save the reshaped slices as a TIFF file
    output_tiff_path = os.path.join(output_folder, output_filename)
    save_stack_as_tiff(final_stack, output_tiff_path)

    # Print the dimensions
    print("Stack dimensions:", final_stack.shape)
