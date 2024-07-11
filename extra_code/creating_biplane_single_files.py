"""
 This script is used to create biplane images from two matching single plane images in subfolders of many images.

    The script assumes that the images are in a folder structure like this:
        main_folder
            subfolder1
                uncompressed
                    image1_bp1.tif
                    image1_bp2.tif
                    image2_bp1.tif
                    image2_bp2.tif
                    ... 
            
                    
    The script will create a new folder 'uncompressed_biplane' in each subfolder and save the concatenated images there.
    The script will also copy the metadata from one of the source images to the concatenated image.
    The script will remove '_bp1' or '_bp2' from the filename for the output file.

 
""" 

import os
import cv2
import numpy as np
import dask.array as da
from tifffile import TiffFile, imwrite



# Function to concatenate images
def concat_images(image_pair):
    image1 = da.from_array(image_pair[0], chunks='auto')
    image2 = da.from_array(image_pair[1], chunks='auto')
    return da.concatenate([image1, image2], axis=1)


# Path to the folder containing subfolders with images
main_folder = '/Volumes/T9/compression/figure_3/data_for_localization'


# Iterate over each subfolder in the main folder
for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)
    if os.path.isdir(subfolder_path):
        uncompressed_folder = os.path.join(subfolder_path, 'uncompressed')
        if os.path.exists(uncompressed_folder):
            # Get the list of image files in the uncompressed folder
            all_files = sorted(os.listdir(uncompressed_folder))
            files1 = [f for f in all_files if 'bp1' in f]
            files2 = [f for f in all_files if 'bp2' in f]

            # Pairing logic for files
            pairs = []
            for f1 in files1:
                base_name = f1.replace("bp1", "bp2")
                for f2 in files2:
                    if f2 == base_name:
                        pairs.append((f1, f2))
                        break

            # Load metadata from one of the source images
            source_metadata = None
            source_image_path = None  # Store the path for metadata copying
            for i, image_pair in enumerate(pairs):
                try:
                    # Load metadata from a TIFF image using tifffile.TiffFile
                    source_image_path = os.path.join(uncompressed_folder, image_pair[0])
                    with TiffFile(source_image_path) as tif:
                        tiff_page = tif.pages[0]
                        
                        if 'tags' in dir(tiff_page):
                            source_metadata = dict(tiff_page.tags)  # Convert tags to a dictionary
                        elif 'metadata' in dir(tiff_page):
                            source_metadata = dict(tiff_page.metadata)  # Convert metadata to a dictionary
                    
                    if source_metadata is not None:
                        source_metadata = {str(key): value for key, value in source_metadata.items()}  # Convert keys to strings
                        
                        # Print the metadata for debugging purposes
                        print(f"Metadata for image {i+1}:\n")
                        for key, value in source_metadata.items():
                            print(f"{key}: {value}")
                        
                        break  # Only need metadata from one image
                except Exception as e:
                    print(f"Error occurred while loading metadata from image {i+1}: {str(e)}")



            # Concatenate and save the images
            for i, image_pair in enumerate(pairs):
                try:
                    # Load the images using OpenCV
                    image1 = cv2.imread(os.path.join(uncompressed_folder, image_pair[0]), cv2.IMREAD_UNCHANGED)
                    image2 = cv2.imread(os.path.join(uncompressed_folder, image_pair[1]), cv2.IMREAD_UNCHANGED)

                    # Concatenate the images using Dask
                    concatenated_image = concat_images((image1, image2))

                    # Convert the Dask array to a NumPy array
                    np_array = concatenated_image.compute().astype(np.uint16)  # Convert to 16-bit

                    # Calculate the new dimensions for the concatenated image
                    new_image_width = image1.shape[1] + image2.shape[1]  # Adjust as needed
                    new_image_length = max(image1.shape[0], image2.shape[0])  # Adjust as needed

                    # Copy selected metadata attributes with adjusted dimensions
                    if source_metadata is not None:
                        # Select the attributes you want to copy
                        selected_attributes = ['BitsPerSample', 'Compression', 'PhotometricInterpretation', 'SamplesPerPixel']

                        # Create the concatenated image's metadata with selected attributes
                        concatenated_metadata = {key: source_metadata[key] for key in selected_attributes if key in source_metadata}

                        # Update dimensions and other metadata attributes
                        concatenated_metadata['ImageWidth'] = new_image_width  # Adjusted width for concatenated image
                        concatenated_metadata['ImageLength'] = new_image_length  # Adjusted length for concatenated image
                        concatenated_metadata['ImageDescription'] = 'Concatenated image'

                        # Remove '_bp1' or '_bp2' from the filename for the output file
                        output_filename = image_pair[0].replace("_bp1", "").replace("_bp2", "")
                        output_folder = os.path.join(subfolder_path, 'uncompressed_biplane')
                        os.makedirs(output_folder, exist_ok=True)
                        output_path = os.path.join(output_folder, output_filename)
                        imwrite(output_path, np_array, resolution=(1, 1), metadata=concatenated_metadata)
                        print(f"Concatenated image {i+1} saved as '{output_path}'.")
                    else:
                        print(f"Source metadata not available for image {i+1}.")
                        imwrite(output_path, np_array, resolution=(1, 1))  # Save without metadata
                        print(f"Concatenated image {i+1} saved without metadata as '{output_path}'.")
                except Exception as e:
                    print(f"Error occurred while processing image {i+1}: {str(e)}")

