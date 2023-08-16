#%%
import os
import zarr
import numpy as np
import imageio


#%%
# Load zarr datasets
raw_data_bp1 = zarr.open('/Users/laurabreimann/Library/CloudStorage/Dropbox-HMS/simulated/raw_data_bp1.zarr', mode='r')
raw_data_bp2 = zarr.open('/Users/laurabreimann/Library/CloudStorage/Dropbox-HMS/simulated/raw_data_bp2.zarr', mode='r')

#%%

# Assuming the images have the same number of slices
num_slices, height, width = raw_data_bp1.shape

# Specify the desired output folder
output_folder = '/Users/laurabreimann/Desktop/simulated_tubulin_data/zarr/biplane_images/'
os.makedirs(output_folder, exist_ok=True)

# Iterate over each slice and combine the images
for idx in range(num_slices):
    image1 = raw_data_bp1[idx, :, :]
    image2 = raw_data_bp2[idx, :, :]

    # Combine images side by side
    combined_image = np.concatenate((image1, image2), axis=1)

    # Convert to 16-bit and scale the values
    combined_image_16bit = (combined_image / np.max(combined_image) * (2**16 - 1)).astype(np.uint16)

    # Save the combined image as a TIFF file using imageio
    output_filename = os.path.join(output_folder, f'combined_image_{idx}.tiff')
    imageio.imwrite(output_filename, combined_image_16bit, format='TIFF')

print(f"Combined images saved to '{output_folder}' folder.")

# %%
