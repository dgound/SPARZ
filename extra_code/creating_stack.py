#%%

#import relevant packages 
import os
import numpy as np
import imageio
from tifffile import imsave
import tifffile


#%%

# Define input and output folders
input_folder = '/Users/laurabreimann/Desktop/simulated_tubulin_data/sparz/uncompressed/fused_biplane/'
output_stack_folder = '/Users/laurabreimann/Desktop/simulated_tubulin_data/sparz/uncompressed/'

# Create the output folder for the stack
os.makedirs(output_stack_folder, exist_ok=True)


# %%

# Get the list of image files in the input folder, excluding .DS_Store
image_files = sorted(file for file in os.listdir(input_folder) if not file.startswith('.DS_Store'))

# Load all images and store in a list
images = []
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = imageio.imread(image_path)
    images.append(image)
    
# Convert the list of images to a NumPy array
images = np.array(images, dtype=np.uint16)


# Get the dimensions of the images
Y, X = images[0].shape  # Assuming all images have the same dimensions
C = 1  # Assuming single channel images

# Get the number of images (time frames)
T = len(image_files)

# Assuming Z is 1 
Z_values = 1  # Choose the appropriate Z value based on your needs

# Create an array with the correct shape
stack_shape = (T, Z_values, 1, Y, X, C)  # Add a singleton Z dimension
stack = np.zeros(stack_shape, dtype=images.dtype)

# Populate the stack with the image data
for i, image in enumerate(images):
    stack[i, 0, 0, :, :, 0] = image

# Save the stack as a TIFF file
output_tiff_path = os.path.join(output_stack_folder, "Nir_et_al.tif")
metadata = {
    'axes': 'TZCYXS',  # Reorder axes to TZCYXS
    'shape': stack_shape,
}
tifffile.imsave(output_tiff_path, stack, dtype=np.uint16, bigtiff=True, imagej=True, metadata=metadata)

print(f"TIFF stack saved as '{output_tiff_path}'.")


# %%
# Print the dimensions
print("Stack dimensions:", stack.shape)


# %%
