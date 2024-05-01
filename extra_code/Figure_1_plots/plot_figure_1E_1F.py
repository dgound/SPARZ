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
color_palette = {'AV1' : '#8ECAE6',
                'ffv1' : '#219EBC',
                'sparz': '#0E92EE',
                'h264' : '#023047',
                'prores' : '#817425',
                'x265' : '#FFB703',
                'zstd' : '#FB8500'
                }



############################################################################################
# ----------------------- NN distances ----------------------------------------------------#
############################################################################################


nn_distances = pd.read_csv('/Users/alioutas/Dropbox/Dropbox (HMS)/data_compression/data_compression_localizations/figure_1_data/synth_tubulin_data/output/synMT_vs_rawTP_20240501__distances.csv')
nn_distances['label'] = [x[0] for x in nn_distances['label'].str.split('_')]
nn_distances['color_palette'] = nn_distances['label'].map(color_palette)

#%%
# boxplot the NN distances for each label and color it by the color palette column
sns.boxplot(data=nn_distances, x='label', y='distance', palette=color_palette, flierprops = dict(marker='.', markerfacecolor='None', markersize=2,  markeredgecolor='black'))
plt.yscale('log')
plt.ylabel('Distance (log)')
plt.xlabel('')








############################################################################################
# ----------------------- Jaccard Index ---------------------------------------------------#
############################################################################################
# %%

jaccard = pd.read_csv('/Users/alioutas/Dropbox/Dropbox (HMS)/data_compression/data_compression_localizations/figure_1_data/synth_tubulin_data/output/synMT_vs_rawTP_20240501__metrics.csv')
jaccard = jaccard[jaccard['metric'] == 'Jaccard']
jaccard['label'] = [x[0] for x in jaccard['label'].str.split('_')]
jaccard['color_palette'] = jaccard['label'].map(color_palette)

#%%
# boxplot the NN distances for each label and color it by the color palette column
sns.stripplot(data=jaccard, x='label', y='value', palette=color_palette, size = 15)
plt.ylabel('Jaccard index')
plt.xlabel('')

# %%
