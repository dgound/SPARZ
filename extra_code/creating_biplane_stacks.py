import os
import numpy as np
from tifffile import TiffFile, imwrite
from natsort import natsorted


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


def save_volume_to_file(volume, output_folder, filename):
    """
    Save a 3D volume to a TIFF file.

    Parameters:
        volume (np.array): The 3D volume to be saved.
        output_folder (str): The directory where the TIFF file will be saved.
        filename (str): The name of the TIFF file.

    Returns:
        None
    """
    output_path = os.path.join(output_folder, filename)
    metadata = {'axes': 'TYX'}
    imwrite(output_path, volume, metadata=metadata)
    print(f"Saved concatenated volume to {output_path}")


def process_files_in_subfolder(uncompressed_folder, output_folder):
    '''Process the files in a subfolder of the main folder.'''
    # Get all files in the uncompressed folder
    files = os.listdir(uncompressed_folder)
    files = natsorted(files)  # Sort the files

    # Identify bp1 and bp2 files
    bp1_file = None
    bp2_file = None
    for file in files:
        if '_bp1' in file:
            bp1_file = file
        elif '_bp2' in file:
            bp2_file = file

    # Check if both files were found
    if bp1_file is None or bp2_file is None:
        print(f"Missing files in folder {uncompressed_folder}. Skipping.")
        return

    # Load, process, and save the volumes
    volume1 = load_volume_from_file(uncompressed_folder, bp1_file)
    volume2 = load_volume_from_file(uncompressed_folder, bp2_file)

    # Concatenate and save the volumes
    concatenated_volume = np.concatenate([volume1, volume2], axis=2)
    # Define metadata
    
    print(f"Concatenated volume shape: {concatenated_volume.shape}")
    output_filename = "Nir_et_al.tif"
    save_volume_to_file(concatenated_volume, output_folder, output_filename)



def main(main_folder, output_main_folder):
    '''Process all subfolders in the main folder.'''
    # Get all subfolders and sort them naturally
    subfolders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]
    subfolders = natsorted(subfolders)

    # Iterate through each subfolder in the main folder
    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder)

        # Define the path to the uncompressed folder
        uncompressed_folder = os.path.join(subfolder_path, 'decompressed')
        if not os.path.exists(uncompressed_folder):
            print(f"Uncompressed folder not found in {subfolder_path}. Skipping.")
            continue

        # Define the output folder for the concatenated images
        output_folder = os.path.join(output_main_folder, subfolder)
        os.makedirs(output_folder, exist_ok=True)
        # Process the files in the uncompressed folder
        process_files_in_subfolder(uncompressed_folder, output_folder)


if __name__ == "__main__":
    main_folder = "/Volumes/T9/compression/Data_for_figures/figure_3/data_for_localization"
    output_main_folder = "/Volumes/T9/compression/Data_for_figures/figure_3/data_for_localization"
    main(main_folder, output_main_folder)