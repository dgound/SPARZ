#%% 
import os
import h5py
import numpy as np
import imageio


#%%

# Load HDF5 datasets
hdf5_file1 = h5py.File('/Users/laurabreimann/Library/CloudStorage/Dropbox-HMS/simulated/tubulin_bp1.hdf5', 'r')
hdf5_file2 = h5py.File('/Users/laurabreimann/Library/CloudStorage/Dropbox-HMS/simulated/tubulin_bp2.hdf5', 'r')


#%%

# Assuming the datasets have the same number of slices
num_slices, height, width = hdf5_file1['slices'].shape

# Specify the desired output folder
output_folder = '/Users/laurabreimann/Desktop/simulated_tubulin_data/hdf5/fused_biplane/'
os.makedirs(output_folder, exist_ok=True)

# Iterate over each slice and combine the images
for idx in range(num_slices):
    image1 = hdf5_file1['slices'][idx, :, :]
    image2 = hdf5_file2['slices'][idx, :, :]

    # Combine images side by side
    combined_image = np.concatenate((image1, image2), axis=1)

    # Convert to 16-bit and scale the values
    combined_image_16bit = (combined_image / np.max(combined_image) * (2**16 - 1)).astype(np.uint16)

    # Save the combined image as a TIFF file using imageio
    output_filename = os.path.join(output_folder, f'combined_image_{idx}.tiff')
    imageio.imwrite(output_filename, combined_image_16bit, format='TIFF')

print(f"Combined images saved to '{output_folder}' folder.")
