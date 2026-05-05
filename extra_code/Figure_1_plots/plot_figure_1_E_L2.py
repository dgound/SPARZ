#%%
import os
import re
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#%%
# color pallete
# color_palette = {
#     'SPARZ': '#0E92EE',
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

order = ['SPARZ', 'h264', 'prores', 'av1', 'x265','SPARZ lossless' ,'ffv1', 'zstd']




path = '/Users/alioutas/HMS Dropbox/Antonios Lioutas/'


#%%
L2_diff = pd.read_csv('/Users/alioutas/HMS Dropbox/Antonios Lioutas/data_compression/data_compression_localizations/figure_1_data/synth_tubulin_data/density_difference_results_final.csv')
L2_diff['label'] = L2_diff['compression_method'].str.split('_').str[0]
L2_diff['label'] = L2_diff['label'].replace('AV1', 'av1')
L2_diff['label'] = L2_diff['label'].replace('sparz', 'SPARZ')
L2_diff['label'] = L2_diff['label'].replace('sparzffv1', 'SPARZ lossless')

# %%
# make a barplot of the L2 distance labeled Ralative spatial density error (vs. raw) 
plt.figure(figsize=(6, 3), dpi=300)
ax = sns.barplot(data=L2_diff, x='label', y='normalized_l2_density_difference', palette=color_palette, order=order)
plt.ylim(0, 0.08)
plt.ylabel('Relative spatial density error (vs. raw)')
plt.xlabel('Compression method')
# add value of each bar on top of it
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

plt.axvline(x=4.5, ymin=0, ymax=1, color='black', lw=2)
plt.text(5.5, 0.085, 'Lossless', ha='center', va='bottom', fontsize=12, color='black')
plt.text(1.5, 0.085, 'Near lossless', ha='center', va='bottom', fontsize=12, color='black')

plt.tight_layout()
plt.show()

# %%