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
############################################################################################
# ----------------------- NN distances ----------------------------------------------------#
############################################################################################


nn_distances = pd.read_csv('/Users/alioutas/HMS Dropbox/Antonios Lioutas/data_compression/data_compression_localizations/figure_1_data/synth_tubulin_data/output/synMT_vs_rawTP_20240501__distances.csv')
nn_distances['label'] = [x[0] for x in nn_distances['label'].str.split('_')]
# replace 'sparz' with 'SPARZ'
nn_distances['label'] = nn_distances['label'].replace('sparz', 'SPARZ')
nn_distances['label'] = nn_distances['label'].replace('sparzffv1', 'SPARZ lossless')
nn_distances['label'] = nn_distances['label'].replace('AV1', 'av1')
nn_distances['color_palette'] = nn_distances['label'].map(color_palette)

# count nulls


#%%
# boxplot the NN distances for each label and color it by the color palette column
# define the order of the labels
sns.boxplot(data=nn_distances, x='label', y='distance', palette=color_palette, flierprops = dict(marker='.', markerfacecolor='None', markersize=2,  markeredgecolor='black'), order=order)
# make y axis tart at -10
# plt.ylim(nn_distances.distance.min(), nn_distances.distance.max())
# some of the boxolots are invisible, so we need to increase by 10% the y axis
plt.ylim(-2, 40)
plt.axvline(x=4.5, ymin=0, ymax=1, color='black', lw=2)
# plt.yscale('log')
plt.ylabel('Distance (nm)')
plt.xlabel('')








############################################################################################
# ----------------------- Jaccard Index ---------------------------------------------------#
############################################################################################
# %%

jaccard = pd.read_csv(path + 'data_compression/data_compression_localizations/figure_1_data/synth_tubulin_data/output/synMT_vs_rawTP_20240501__metrics.csv')
jaccard = jaccard[jaccard['metric'] == 'Jaccard']
jaccard['label'] = [x[0] for x in jaccard['label'].str.split('_')]
jaccard['label'] = jaccard['label'].replace('sparz', 'SPARZ')
jaccard['label'] = jaccard['label'].replace('sparzffv1', 'SPARZ lossless')
jaccard['label'] = jaccard['label'].replace('AV1', 'av1')
jaccard['color_palette'] = jaccard['label'].map(color_palette)

#%%
# boxplot the NN distances for each label and color it by the color palette column
plt.figure(figsize=(6, 3))
ax = sns.stripplot(data=jaccard, x='label', y='value', palette=color_palette, size = 15, order = order)
plt.ylim(0, 1.09)
plt.ylabel('Jaccard index')
plt.axvline(x=4.5, ymin=0, ymax=1, color='black', lw=2)
# move legend outside the plot and show label names
# plt.legend(order, bbox_to_anchor=(1.05, 1), loc='upper left', title='Label')
plt.xlabel('')

# %%
