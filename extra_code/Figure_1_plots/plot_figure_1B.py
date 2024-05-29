#%% IMPORTS
import os
import re
import numpy as np
import tifffile
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %% FUNCTIONS
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def calculate_ssim(original_file, compressed_file):
    ssim_scores = []
    with tifffile.TiffFile(original_file) as tif_original, tifffile.TiffFile(compressed_file) as tif_compressed:
        for original_frame, compressed_frame in zip(tif_original.series[0].pages, tif_compressed.series[0].pages):
            original_image = original_frame.asarray()
            compressed_image = compressed_frame.asarray()
            score = ssim(original_image, compressed_image, data_range=compressed_image.max() - compressed_image.min())
            ssim_scores.append(score)
    return np.median(ssim_scores)

def get_total_size_of_files(folder_path, file_extensions):
    total_size = 0
    compressed_folder_path = os.path.join(folder_path, "compressed")
    for file in os.listdir(compressed_folder_path):
        if any(file.endswith(ext) for ext in file_extensions):
            total_size += os.path.getsize(os.path.join(compressed_folder_path, file))
    return total_size

#%% PATHS

original_file = '/Users/alioutas/Dropbox/Dropbox (HMS)/data_compression/data_compression_localizations/figure_1_data/synth_tubulin_for_SSIM/sequence-MT0.N1.HD-BP.tif'
main_folder = '/Users/alioutas/Dropbox/Dropbox (HMS)/data_compression/data_compression_localizations/figure_1_data/synth_tubulin_for_SSIM/'

original_file_size = 41360118  # in bytes

#%% CALCULATE SSIM AND FILE SIZE PERCENTAGE
results = {}
for folder in sorted(os.listdir(main_folder), key=natural_sort_key):
    folder_path = os.path.join(main_folder, folder)
    if os.path.isdir(folder_path):
        compressed_file = os.path.join(folder_path, 'sequence-MT0.N1.HD-BP.tif')
        median_ssim = calculate_ssim(original_file, compressed_file)
        total_file_size = get_total_size_of_files(folder_path, ['.mp4', '.npz', '.avi', '.mov', '.zst'])
        file_size_percentage = (total_file_size / original_file_size) * 100

        results[folder] = {
            'median_ssim': median_ssim,
            'file_size_percentage': file_size_percentage
        }


#%% PLOT
# Define color palette
color_palette = {
    # 'SPARZ': '#0E92EE',
    'av1': '#8ECAE6',
    'ffv1': '#219EBC',
    'h264': '#023047',
    'prores': '#817425',
    'x265': '#FFB703',
    'zstd': '#FB8500'
}


# Create DataFrame from results
ssim_output = pd.DataFrame(results).T.reset_index()
ssim_output.columns = ['label', 'median_ssim', 'file_size_percentage']

# Extract label and map colors
ssim_output['label'] = ssim_output['label'].str.split('_').str[0]
ssim_output['label'] = ssim_output['label'].replace('AV1', 'av1')
ssim_output['color_palette'] = ssim_output['label'].map(color_palette)

# Round file size percentage
ssim_output['file_size_percentage'] = np.round(ssim_output['file_size_percentage'], 2)

ssim_output.to_csv('/Users/alioutas/Dropbox/Dropbox (HMS)/data_compression/data_compression_localizations/figure_1_data/synth_tubulin_for_SSIM/output/SSIM_filesize_20240529.csv', index=False)

# Define the order of labels
order = ['av1', 'ffv1', 'prores', 'h264', 'x265', 'zstd']

# Create the stripplot
sns.stripplot(data=ssim_output, x='file_size_percentage', y='median_ssim', hue='label', palette=color_palette, size=15, hue_order=order)
plt.ylabel('Median SSIM Score')
plt.ylim(0.7, 1.009)
plt.xlabel('')
plt.legend(title='Codec')

# save plot  as png
output_file = '/Users/alioutas/Dropbox/Dropbox (HMS)/data_compression/data_compression_localizations/figure_1_data/synth_tubulin_for_SSIM/output/Figure1B_SSIM_filesize_20240529.pdf'
plt.savefig(output_file, format='pdf')

# Show the plot
plt.show()
