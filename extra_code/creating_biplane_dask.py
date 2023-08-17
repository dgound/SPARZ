#%%

import os
import cv2
import numpy as np
import zarr
import dask.array as da
import imageio
from tifffile.tifffile import TiffWriter
#import tifffile
from tifffile import TiffFile, imwrite




# %%

def concat_images(image_pair):
    image1 = da.from_array(image_pair[0], chunks='auto')
    image2 = da.from_array(image_pair[1], chunks='auto')
    return da.concatenate([image1, image2], axis=1)


# %%

# Folder paths containing the matching images
folder1 = '/Users/laurabreimann/Desktop/simulated_tubulin_data/sparz/uncompressed/bp1/'
folder2 = '/Users/laurabreimann/Desktop/simulated_tubulin_data/sparz/uncompressed/bp2/'
output_folder ='/Users/laurabreimann/Desktop/simulated_tubulin_data/sparz/uncompressed/fused_biplane/'
zarr_folder = "/Users/laurabreimann/Desktop/simulated_tubulin_data/sparz/uncompressed/zarr/"

# %%
 
# Get the list of image files in the input folders
files1 = sorted(os.listdir(folder1))
files2 = sorted(os.listdir(folder2))


#%%
# Combine the image files from both folders
image_files = zip(files1, files2)

# %%
# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(zarr_folder, exist_ok=True)

#%%

# Load metadata from one of the source images
source_metadata = None
source_image_path = None  # Store the path for metadata copying
for i, image_pair in enumerate(image_files):
    try:
        # Load metadata from a TIFF image using tifffile.TiffFile
        source_image_path = os.path.join(folder1, image_pair[0])
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



# %%

# Concatenate and save the images
for i, image_pair in enumerate(zip(files1, files2)):

    try:
        # Load the images using OpenCV
        image1 = cv2.imread(os.path.join(folder1, image_pair[0]), cv2.IMREAD_UNCHANGED)
        image2 = cv2.imread(os.path.join(folder2, image_pair[1]), cv2.IMREAD_UNCHANGED)

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
            
            # Save the image as a TIFF file with metadata using tifffile
            output_path = os.path.join(output_folder, os.path.basename(image_pair[0]))
            imwrite(output_path, np_array, resolution=(1, 1), metadata=concatenated_metadata)
            print(f"Concatenated image {i+1} saved as '{output_path}'.")
        else:
            print(f"Source metadata not available for image {i+1}.")
            imwrite(output_path, np_array, resolution=(1, 1))  # Save without metadata
            print(f"Concatenated image {i+1} saved without metadata as '{output_path}'.")
    except Exception as e:
        print(f"Error occurred while processing image {i+1}: {str(e)}")


# %%
