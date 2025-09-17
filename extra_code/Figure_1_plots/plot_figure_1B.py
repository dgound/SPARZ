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

original_file = '/Users/alioutas/HMS Dropbox/Antonios Lioutas/data_compression/data_compression_localizations/figure_1_data/synth_tubulin_for_SSIM//sequence-MT0.N1.HD-BP.tif'
main_folder = '/Users/alioutas/HMS Dropbox/Antonios Lioutas/data_compression/data_compression_localizations/figure_1_data/synth_tubulin_for_SSIM/'


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


#%%
# Define color palette

# color_palette = {
#      'SPARZ': '#0E92EE',
#     'h264': '#8ECAE6',
#     'prores': '#219EBC',
#     'av1': '#023047',
#     'x265': '#817425',
#     'ffv1': '#FFB703',
#     'zstd': '#FB8500'
# }
# New simplified color palette
color_palette = {
    'SPARZ': '#0E92EE',
    'h264': '#817425',
    'prores': '#817425',
    'av1': '#817425',
    'x265': '#817425',
    'ffv1': '#FB8500',
    'zstd': '#FB8500'
}
#%%
# Create DataFrame from results
ssim_output = pd.DataFrame(results).T.reset_index()
ssim_output.columns = ['label', 'median_ssim', 'file_size_percentage']

# Extract label and map colors
ssim_output['label'] = ssim_output['label'].str.split('_').str[0]
ssim_output['label'] = ssim_output['label'].replace('AV1', 'av1')
ssim_output['color_palette'] = ssim_output['label'].map(color_palette)

# Round file size percentage
ssim_output['file_size_percentage'] = np.round(ssim_output['file_size_percentage'], 2)
# save the datatable
ssim_output.to_csv('/Users/alioutas/Dropbox/Dropbox (HMS)/data_compression/data_compression_localizations/figure_1_data/synth_tubulin_for_SSIM/output/SSIM_filesize_20240529.csv', index=False)


#%% PLOT

#%%
# read ssim results
ssim_output = pd.read_csv("/Users/alioutas/HMS Dropbox/Antonios Lioutas/data_compression/data_compression_localizations/figure_1_data/synth_tubulin_for_SSIM/output/SSIM_filesize_20240529.csv")



#%%
# Define the order of labels
order = [#'SPARZ',
    'h264', 'prores', 'av1', 'x265', 'ffv1', 'zstd']
# Create the stripplot
sns.stripplot(data=ssim_output, x='file_size_percentage', y='median_ssim', hue='label', palette=color_palette, size=15, hue_order=order)
plt.ylabel('Median SSIM Score')
plt.xlabel('File Size (%)')
plt.ylim(0.7, 1.009)
plt.legend(title='Codec')

# save plot  as png
output_file = '/Users/alioutas/Dropbox/HMS Dropbox/data_compression/data_compression_localizations/figure_1_data/synth_tubulin_for_SSIM/output/Figure1B_SSIM_filesize_20240529.pdf'
plt.savefig(output_file, format='pdf')

# Show the plot
plt.show()

# %%
#Plotting only SSIM
plt.figure(figsize=(6, 3))
sns.stripplot(data=ssim_output, x='label', y='median_ssim', hue='label', palette=color_palette, size=15, order=order)
plt.ylabel('Median SSIM Score')
plt.xlabel(' ')
plt.ylim(0, 1.09)
plt.legend([],[],frameon=False)  # Disable legend
# plt.legend(title='')  
# move legend outside the plot
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# %%
# Plotting only size
plt.figure(figsize=(6, 3))
sns.stripplot(data=ssim_output, x='label', y='file_size_percentage', hue='label', palette=color_palette, size=15, order=order)
plt.ylabel('File Size (%)')
plt.xlabel(' ')
plt.ylim(0, 100)
plt.legend([],[],frameon=False)  # Disable legend
# move legend outside the plot
# %%
