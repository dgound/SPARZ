#%%
import os
import re
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


#%%
# color pallete
# color_palette = {
#     # 'SPARZ': '#0E92EE',
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

order = ['SPARZ', 'h264', 'prores', 'av1', 'x265', 'ffv1', 'zstd']


#%%
# Toy processing time values
# fake_processing_time = pd.DataFrame({'codec': ['av1', 'ffv1', 'h264', 'prores', 'x265', 'zstd'],
#                                     'processing_time': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
#                                     'color_palette' : color_palette.values()})


processing_time = pd.read_csv("https://www.dropbox.com/scl/fi/wwf57wq5sduwm9hkqjhyn/run_times_no_ROI_tubulin.csv?rlkey=est6xcba0xlevj5lowe9t2irj&dl=1")

# read in the prosessing time with and extract the information for SPARZ (x265+ROI)
processing_time_plus_roi = pd.read_csv("https://www.dropbox.com/scl/fi/wwf57wq5sduwm9hkqjhyn/run_times_with_ROI_tubulin.csv?rlkey=est6xcba0xlevj5lowe9t2irj&dl=1")

SPARZ_row = processing_time_plus_roi[processing_time_plus_roi['codec'] == 'x265'].copy()
SPARZ_row['codec'] = SPARZ_row['codec'].replace('x265', 'SPARZ')
# append this row to processing_time
processing_time = pd.concat([processing_time, SPARZ_row], ignore_index=True)

processing_time['codec'] = processing_time['codec'].replace('x264', 'h264')

processing_time['color_palette'] = processing_time['codec'].map(color_palette)

# Sort the data by the specified order
processing_time['codec'] = pd.Categorical(processing_time['codec'], categories=order, ordered=True)
processing_time = processing_time.sort_values('codec')
processing_time


# %% PROCESSING  time ** WITHOUT ** ROI
# Create two subplot barplots with the compression and decompression times
fig, (ax0, ax1) = plt.subplots(figsize=(10, 7), ncols=2, sharey=False)

# Defining error for each bar
error_compression = processing_time['compression_std_dev']
error_decompression = processing_time['decompression_std_dev']

# Defining overhead errors
error_overhead_compression = processing_time['compression_overhead_std_dev']
error_overhead_decompression = processing_time['decompression_overhead_std_dev']

# Compression - no change in y range
ax0.bar(processing_time['codec'], processing_time['compression_mean'], color=[color_palette[x] for x in processing_time['codec']], yerr=error_compression, capsize=5)
ax0.set_title('Compression', fontsize=12)
ax0.set_ylabel('Time (seconds)', fontsize=12)

# Decompression - adjusting y range from 0 to 6 for better subplot positioning
ax1.bar(processing_time['codec'], processing_time['decompression_mean'], color=[color_palette[x] for x in processing_time['codec']], yerr=error_decompression, capsize=5)
ax1.set_ylim([0, 6])
ax1.set_title('Decompression', fontsize=12)

# Insert subplots for overhead at the center top
inset_ax0 = inset_axes(ax0, width="30%", height="30%", loc='upper center')
inset_ax1 = inset_axes(ax1, width="30%", height="30%", loc='upper center')

# Bar plot for overhead with errors, using the same colors and rotating labels
inset_ax0.bar(processing_time['codec'], processing_time['compression_overhead_mean'], color=[color_palette[x] for x in processing_time['codec']], yerr=error_overhead_compression, capsize=5)
inset_ax1.bar(processing_time['codec'], processing_time['decompression_overhead_mean'], color=[color_palette[x] for x in processing_time['codec']], yerr=error_overhead_decompression, capsize=5)

# Labeling y-axis and removing titles for the inset axes
inset_ax0.set_ylabel('Overhead Time (seconds)')
inset_ax1.set_ylabel('Overhead Time (seconds)')

# Rotating x-axis labels for inset plots
inset_ax0.tick_params(axis='x', rotation=45)
inset_ax1.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()



# %%
