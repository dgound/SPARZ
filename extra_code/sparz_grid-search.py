"""
This script runs a grid search over the parameters of SPARZ (kernel size, relative threshold, compression level).
It uses the SPARZIP and SPARUNZIP classes from SPARZ.py.

"""



from SPARZ import SPARZIP, SPARUNZIP
import os
from tqdm import tqdm

# Paths for input files
path1 = '/Users/laurabreimann/Documents/Postdoc/Colabs/Compression/sequence-as-stack-MT0.N1.HD-BP+250.tif'
path2 = '/Users/laurabreimann/Documents/Postdoc/Colabs/Compression/sequence-as-stack-MT0.N1.HD-BP-250.tif'

stem= 'MT0'

# Define ranges for grid search
kernel_sizes =  range(5, 12, 2) #  5, 7, 9, 11 
rel_thresholds = [0.35, 0.45, 0.55, 0.65]  # 0.35, 0.45, 0.55, 0.65
compression_levels = [0, 1, 2, 3]  # 0, 1, 2, 3


# Extract base filenames without extension
base_filename1 = os.path.splitext(os.path.basename(path1))[0]
base_filename2 = os.path.splitext(os.path.basename(path2))[0]


# Total number of iterations for progress bar
total_iterations = len(kernel_sizes) * len(rel_thresholds) * len(compression_levels)


# Iterate over all combinations of parameters
with tqdm(total=total_iterations, desc="Grid Search Progress", unit="iteration") as pbar: # Initialize tqdm progress bar
    for kernel_size in kernel_sizes:
        for rel_thresh in rel_thresholds:
            for compression_level in compression_levels:
                # Create output paths based on parameters
                output_path = f'/Users/laurabreimann/Desktop/microtubule_data/sparz_k{kernel_size}_rt{int(rel_thresh*100)}_lev{compression_level}'
                uncompressed_path = os.path.join(output_path, 'uncompressed')
                os.makedirs(uncompressed_path, exist_ok=True)

                # SPARZIP
                z = SPARZIP(path1, stem, output_path, path2, relative_threshold = rel_thresh, kernel_size =kernel_size)
                z.run(codec='x265', compression_level=compression_level)

                # Output files for SPARUNZIP
                path_sparse_bp1 = os.path.join(output_path, f'{base_filename1}.npz')
                path_sparse_bp2 = os.path.join(output_path, f'{base_filename2}.npz')
                mp4_1 = os.path.join(output_path, f'{base_filename1}_compression_level_{compression_level}.mp4')
                mp4_2 = os.path.join(output_path, f'{base_filename2}_compression_level_{compression_level}.mp4')

                # SPARUNZIP
                u= SPARUNZIP(path_sparse_bp1= path_sparse_bp1,
                            path_encoded_bp1= mp4_1,
                            stem= stem,
                            output_path= uncompressed_path,
                            path_sparse_bp2= path_sparse_bp2,
                            path_encoded_bp2= mp4_2,
                            use_roi = True,
                            chunk_size = 10) 
                u.run()

                # Update progress bar
                pbar.update(1)


