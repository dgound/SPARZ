#%% IMPORTS
from datetime import date
from glob import glob
import os
import re
import numpy as np
import tifffile
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#%% FUNCTIONS
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
    for file in os.listdir(folder_path):
        if any(file.endswith(ext) for ext in file_extensions):
            total_size += os.path.getsize(os.path.join(folder_path, file))
    return total_size

#%% PATHS

# original_file = '/Users/alioutas/HMS Dropbox/Antonios Lioutas/data_compression/data_compression_localizations/figure_1_data/synth_tubulin_for_SSIM//sequence-MT0.N1.HD-BP.tif'
original_files = glob('/Users/alioutas/HMS Dropbox/Antonios Lioutas/data_compression/data_compression_localizations/figure_1_data/synth_tubulin_data/raw/*.tif')
main_folder = '/Users/alioutas/HMS Dropbox/Antonios Lioutas/data_compression/data_compression_localizations/figure_1_data/synth_tubulin_data/'


# original_file_size = 41360118  # in bytes
# calcuate the total original file size of all original files
original_file_size = sum(os.path.getsize(f) for f in original_files)

folders_level_0 = sorted([f for f in os.listdir(main_folder) if f.endswith('level_0')], key=natural_sort_key)

#%% CALCULATE SSIM AND FILE SIZE PERCENTAGE
results = {}
for folder in folders_level_0:
    folder_path = os.path.join(main_folder, folder)
    print(f"\nProcessing folder: {folder_path}")
    if os.path.isdir(folder_path):
        # Find all .tif or .tiff files in the compressed folder
        compressed_files = sorted(
            glob(os.path.join(folder_path, '*.tif')) + glob(os.path.join(folder_path, '*.tiff')),
            key=natural_sort_key
        )
        print(f"Compressed files found: {[os.path.basename(f) for f in compressed_files]}")
        ssim_scores = []
        # Comparison between "-250"/"+250" matching biplanes:
        orig_minus = next((f for f in original_files if "-250" in os.path.basename(f)), None)
        orig_plus = next((f for f in original_files if "+250" in os.path.basename(f)), None)
        comp_minus = next((f for f in compressed_files if "-250" in os.path.basename(f)), None)
        comp_plus = next((f for f in compressed_files if "+250" in os.path.basename(f)), None)
        if orig_minus and comp_minus:
            print(f"Comparing {orig_minus} with {comp_minus}")
            score_minus = calculate_ssim(orig_minus, comp_minus)
            ssim_scores.append(score_minus)
        if orig_plus and comp_plus:
            print(f"Comparing {orig_plus} with {comp_plus}")
            score_plus = calculate_ssim(orig_plus, comp_plus)
            ssim_scores.append(score_plus)
        if ssim_scores:
            median_ssim = np.median(ssim_scores)
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
    'SPARZ lossless': '#0E92EE',
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
ssim_output['label'] = ssim_output['label'].replace('sparz', 'SPARZ')
ssim_output['label'] = ssim_output['label'].replace('sparzffv1', 'SPARZ lossless')
ssim_output['color_palette'] = ssim_output['label'].map(color_palette)

# Round file size percentage
ssim_output['file_size_percentage'] = np.round(ssim_output['file_size_percentage'], 2)
# Ensure output directory exists before saving
output_dir = '/Users/alioutas/HMS Dropbox/Antonios Lioutas/data_compression/data_compression_localizations/figure_1_data/synth_tubulin_data/output'
os.makedirs(output_dir, exist_ok=True)
# save the datatable
ssim_output.to_csv(f'{output_dir}/SSIM_filesize_{date.today().strftime("%Y%m%d")}.csv', index=False)


#%% PLOT

#%%
# read ssim results
ssim_output = pd.read_csv('/Users/alioutas/HMS Dropbox/Antonios Lioutas/data_compression/data_compression_localizations/figure_1_data/synth_tubulin_data/output/SSIM_filesize_20251001.csv')



#%%
# Define the order of labels
order = ['SPARZ', 'h264', 'prores', 'av1', 'x265','SPARZ lossless' ,'ffv1', 'zstd']
# Create the stripplot
sns.stripplot(data=ssim_output, x='file_size_percentage', y='median_ssim', hue='label', palette=color_palette, size=15, hue_order=order, dodge=True)
plt.ylabel('Median SSIM Score')
plt.xlabel('File Size (%)')
plt.ylim(0.7, 1.009)
plt.legend(title='Codec', loc='lower right')

# save plot  as png
output_file = '/Users/alioutas/HMS Dropbox/Antonios Lioutas/data_compression/data_compression_localizations/figure_1_data/synth_tubulin_for_SSIM/output/Figure1B_SSIM_filesize_20240529.pdf'
plt.savefig(output_file, format='pdf')

# Show the plot
plt.show()

# %%
#Plotting only SSIM
plt.figure(figsize=(6, 2))
sns.stripplot(data=ssim_output, x='label', y='median_ssim', hue='label', palette=color_palette, size=15, order=order)
plt.ylabel('Median SSIM Score', fontsize=12)
plt.xlabel(' ')
plt.ylim(0, 1.09)
plt.xticks(rotation=45)
plt.legend([],[],frameon=False)  # Disable legend
plt.axvline(x=4.5, ymin=0, ymax=1, color='black', lw=2)
# plt.legend(title='')  
# move legend outside the plot
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# %%
# Plotting only size
plt.figure(figsize=(6, 2))
sns.stripplot(data=ssim_output, x='label', y='file_size_percentage', hue='label', palette=color_palette, size=15, order=order)
plt.ylabel('File Size (%)', fontsize=12)
plt.xlabel(' ')
plt.xticks(rotation=45)
plt.ylim(0, 100)
plt.legend([],[],frameon=False)  # Disable legend
plt.axvline(x=4.5, ymin=0, ymax=1, color='black', lw=2)

# move legend outside the plot
# %%
