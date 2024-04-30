#%%
import os
import re
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import pandas as pd

#%%
# color pallete
color_palette = {'av1' : '#8ECAE6',
                'ffv1' : '#219EBC',
                'x264' : '#023047',
                'prores' : '#817425',
                'x265' : '#FFB703',
                'zstd' : '#FB8500'
                }


#%%
# Toy processing time values
# fake_processing_time = pd.DataFrame({'codec': ['av1', 'ffv1', 'h264', 'prores', 'x265', 'zstd'],
#                                     'processing_time': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
#                                     'color_palette' : color_palette.values()})


processing_time = pd.read_csv("/Users/alioutas/Downloads/run_times_no_ROI_tubulin.csv")
processing_time['color_palette'] = processing_time['codec'].map(color_palette)

#%%
# dot plot the processing time
fig, ax = plt.subplots()
ax.scatter(data=processing_time, x='codec', y='compression_mean', color='color_palette', marker='o', s =250)  # Remove duplicate 'color' keyword argument

ax.set_ylabel('Processing time (s)')
# ax.set_yticks([])
ax.set_title('Processing times codec only (no ROI)')
plt.show()


# %%
# Create two subplot barplots with the compression and decompression times
fig, (ax0, ax1) = plt.subplots(figsize=(10, 7), ncols=2, sharey=True)

# Defining error for each bar
error_compression = processing_time['compression_overhead_std_dev']
error_decompression = processing_time['decompression_overhead_std_dev']

# Additional standard deviation errors
error_compression_mean = processing_time['compression_std_dev']
error_decompression_mean = processing_time['decompression_std_dev']

# Compression
ax0.bar(processing_time['codec'], processing_time['compression_mean'], color=[color_palette[x] for x in processing_time['codec']], label='Compression Mean', yerr=error_compression_mean, capsize=5)
ax0.bar(processing_time['codec'], processing_time['compression_overhead_mean'], bottom=processing_time['compression_mean'], color='gray', label='Compression Overhead', yerr=error_compression)

# Decompression
ax1.bar(processing_time['codec'], processing_time['decompression_mean'], color=[color_palette[x] for x in processing_time['codec']], label='Decompression Mean', yerr=error_decompression_mean, capsize=5)
ax1.bar(processing_time['codec'], processing_time['decompression_overhead_mean'], bottom=processing_time['decompression_mean'], color='gray', label='Decompression Overhead', yerr=error_decompression)

ax1.set_title('Decompression')
ax1.legend("")

ax0.set_ylabel('Time (seconds)')
ax0.set_title('Compression')
ax0.legend("")

# ax0.set_yscale('log')

plt.show()


# %%

processing_time = pd.read_csv("/Users/alioutas/Downloads/run_times_with_ROI_tubulin.csv")
processing_time['color_palette'] = processing_time['codec'].map(color_palette)

#%%
# dot plot the processing time
fig, ax = plt.subplots()
codec = 'av1'  # Define the value of codec
ax.scatter(data=processing_time, x='codec', y='compression_mean', color='color_palette', marker='o', s =250)  # Remove duplicate 'color' keyword argument
ax.errorbar(data=processing_time, x='codec', y='compression_mean', yerr='compression_std_dev', fmt='o', color='black', capsize=10)

ax.set_ylabel('Processing time (s)')
# ax.set_yticks([])
ax.set_title('Processing times SPARZ')
plt.show()


# %%
