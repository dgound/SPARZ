#%%

import os
import cv2
import numpy as np
import zarr
import dask.array as da
import imageio

# %%
def concat_images(image_pair):
    image1 = da.from_array(image_pair[0], chunks='auto')
    image2 = da.from_array(image_pair[1], chunks='auto')
    return da.concatenate([image1, image2], axis=1)


# %%

# Folder paths containing the matching images
folder1 = '/Users/laurabreimann/Library/CloudStorage/Dropbox-HMS/x265_lossless_decoded_bp1/'
folder2 = '/Users/laurabreimann/Library/CloudStorage/Dropbox-HMS/x265_lossless_decoded_bp2/'
output_folder ='/Users/laurabreimann/Desktop/Guys_data/x265fused_biplane/'
zarr_folder = "/Users/laurabreimann/Desktop/Guys_data/x265/zarr/"

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


# Concatenate and save the images
for i, image_pair in enumerate(image_files):
    try:
        # Load the images using OpenCV
        image1 = cv2.imread(os.path.join(folder1, image_pair[0]), cv2.IMREAD_UNCHANGED)
        image2 = cv2.imread(os.path.join(folder2, image_pair[1]), cv2.IMREAD_UNCHANGED)

        # Concatenate the images using Dask
        concatenated_image = concat_images((image1, image2))

        # Save the concatenated image as a Zarr store
        output_path = os.path.join(zarr_folder, f"image_{i}.zarr")
        concatenated_image.to_zarr(output_path)

        # Convert the Zarr store data to a NumPy array
        np_array = zarr.open(output_path, mode='r')[:]

        # Convert the NumPy array to the correct image format (if necessary)
        image_data = np_array.squeeze().astype(np.uint16)  # Convert to 16-bit

    

# Save the image as a TIFF file with the name of the corresponding image in folder1
        output_path = os.path.join(output_folder, files1[i])
        imageio.imwrite(output_path, image_data)

        print(f"Concatenated image {i+1} saved as '{output_path}'.")
    except Exception as e:
        print(f"Error occurred while processing image {i+1}: {str(e)}")



# %%
