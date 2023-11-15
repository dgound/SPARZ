import os
import numpy as np
from tifffile import TiffFile, imsave
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
    imsave(output_path, volume)
    print(f"Saved concatenated volume to {output_path}")


def main(folder1, folder2, output_folder):
    """
    Main function to load pairs of 3D volumes, concatenate them, and save the concatenated volumes.

    Parameters:
        folder1 (str): Directory containing the first set of TIFF files.
        folder2 (str): Directory containing the second set of TIFF files.
        output_folder (str): Directory to save the concatenated TIFF files.

    Returns:
        None
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get the sorted list of image files in the input folders
    files1 = natsorted(os.listdir(folder1))
    print(files1)
    files2 = natsorted(os.listdir(folder2))
    print(files2)
    assert len(files1) == len(
        files2
    ), "Mismatch in number of files between the two folders."

    for i, (filename1, filename2) in enumerate(zip(files1, files2)):
        output_filename = f"concatenated_part_{i}.tiff"

        # Load the 3D volumes
        volume1 = load_volume_from_file(folder1, filename1)
        volume2 = load_volume_from_file(folder2, filename2)

        # Ensure the loaded volumes are 3D and have the expected dimensions
        assert volume1.shape == (250, 245, 245) and volume2.shape == (
            250,
            245,
            245,
        ), f"Loaded volumes from {filename1} and {filename2} have unexpected shapes: {volume1.shape}, {volume2.shape}"

        # Concatenate the volumes along the y-axis
        concatenated_volume = np.concatenate([volume1, volume2], axis=2)

        # Save the concatenated volume
        save_volume_to_file(concatenated_volume, output_folder, output_filename)


if __name__ == "__main__":
    folder1 = "/Users/laurabreimann/Desktop/input_data/nir_etal_ROI/level_3_kernel_11_relThres_55/BP1"
    folder2 = "/Users/laurabreimann/Desktop/input_data/nir_etal_ROI/level_3_kernel_11_relThres_55/BP2"
    output_folder = "/Users/laurabreimann/Desktop/input_data/nir_etal_ROI/level_3_kernel_11_relThres_55/biplane"
    main(folder1, folder2, output_folder)
