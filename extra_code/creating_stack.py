# This script is used to create a TIFF stack from a folder containing single images.


import os
import numpy as np
import imageio
import tifffile

def load_images_from_folder(input_folder):
    """
    Load all images from the specified folder.
    
    Args:
    - input_folder (str): Path to the folder containing the images.
    
    Returns:
    - np.array: Array containing all the loaded images.
    """
    # Exclude system files and load image files
    image_files = sorted(file for file in os.listdir(input_folder) if not file.startswith('.'))
    
    images = [imageio.imread(os.path.join(input_folder, img_file)) for img_file in image_files]
    
    return np.array(images, dtype=np.uint16)

def convert_images_to_stack(images):
    """
    Convert a list of images into a single TIFF stack with shape (T, Z, C, Y, X).
    
    Args:
    - images (np.array): Array containing the loaded images.
    
    Returns:
    - np.array: 5D TIFF stack.
    """
    T, Y, X = len(images), *images[0].shape
    stack_shape = (T, 1, 1, Y, X, 1)  # Single channel, single Z value
    stack = np.zeros(stack_shape, dtype=images.dtype)

    for i, image in enumerate(images):
        stack[i, 0, 0, :, :, 0] = image
        
    return stack

if __name__ == '__main__':
    main_folder = '/Users/laurabreimann/Desktop/simulated_tubulin_data/raw'
    
    # Iterate over each subfolder in the main folder
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        
        if os.path.isdir(subfolder_path):
            # Define input and output folders within the subfolder
            input_folder = os.path.join(subfolder_path, 'uncompressed_biplane')
            output_folder = os.path.join(subfolder_path)
            output_filename = 'sequence-MT0.N1.HD-BP.tif'
            
            # Ensure output folder exists
            os.makedirs(output_folder, exist_ok=True)

            # Load images and convert to a single TIFF stack
            if os.path.exists(input_folder):
                images = load_images_from_folder(input_folder)
                stack = convert_images_to_stack(images)
                
                # Save the stack as a TIFF in the output folder
                output_tiff_path = os.path.join(output_folder, output_filename)
                metadata = {'axes': 'TZCYXS', 'shape': stack.shape}
                tifffile.imsave(output_tiff_path, stack, dtype=np.uint16, bigtiff=True, imagej=True, metadata=metadata)
                
                print(f"TIFF stack saved as '{output_tiff_path}'.")
                print("Stack dimensions:", stack.shape)