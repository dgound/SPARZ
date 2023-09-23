#%%
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/Users/alioutas/Library/CloudStorage/GoogleDrive-alioutas@gmail.com/My Drive/GitHub/SPARZ3/extra_code')
from h5r_functions import h5r_to_df
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import distance
from glob import glob
import seaborn as sns
import multiprocessing
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import entropy
from joblib import Parallel, delayed
from tqdm import tqdm
import glob


#####################################################################
# read in saved dataframes
#####################################################################
#%%
import pandas as pd

#VS_RAW
# df_out = pd.read_csv('/Volumes/T7/compression_data/data_compression_localizations/Nir_et_al/output/nir_etal_vs_raw_20230922_sparz_lev1_k11_rt55_distances.csv')
# df_stats_out = pd.read_csv('/Volumes/T7/compression_data/data_compression_localizations/Nir_et_al/output/nir_etal_vs_raw_20230922_sparz_lev1_k11_rt55_metrics.csv')

#VS_COMPRESSED
df_out = pd.read_csv('/Volumes/T7/compression_data/data_compression_localizations/Nir_et_al/output/nir_etal_vs_compressed_20230922_sparz_lev1_k11_rt55_distances.csv')
df_stats_out = pd.read_csv('/Volumes/T7/compression_data/data_compression_localizations/Nir_et_al/output/nir_etal_vs_compressed_20230922_sparz_lev1_k11_rt55_metrics.csv')

#%%
df_out.groupby('codec').describe()
#%%
df_out_new = df_out.copy()
df_out_new.distance = round(np.log(df_out_new.distance+1),10)

#%%
df_out_new.groupby('codec').describe()

# %%
sns.set_theme(style="whitegrid")
sns.kdeplot(data=df_out, hue="label", bw_adjust=.1, x="distance", color ="label" , fill=False, common_norm=True, alpha=1, linewidth=2, log_scale=False)
# sns.histplot(data=df_out, hue="label", x="distance", color ="label" , fill=False, common_norm=True, alpha=.4, linewidth=2, log_scale=False)
plt.axvline(x=40, color='r', linestyle='--', linewidth=2)
ax.legend(title='Codec')
# log y scale
plt.yscale('log')
plt.xscale('log')
# set x limits to 0-300
plt.xlim(0,300)
# the legend should be outside the plot at the bottom and aligned with the x axis
# plt.legend(bbox_to_anchor=(0., -0.3, 1., .102), loc='lower center', ncol=3, mode="expand", borderaxespad=0.)
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# save plot
# plt.savefig(os.path.join(os.getcwd()+ '/output/', data_name+ date +'_'+os.path.basename(os.path.dirname(files[0])) + '_kdeplot.png'), dpi=300)
# plt.savefig(os.path.join(path+ '/output/', data_name+ date +'_'+os.path.basename(os.path.dirname(files[0])) + '_kdeplot.png'), dpi=300)

# %%
fig, ax = plt.subplots()
for codec, group in df_out.groupby('label'):
    ax.hist(group['distance'], bins=100, alpha=0.6, label=codec)
ax.axvline(x=40, color='r', linestyle='--', linewidth=2)
ax.set_yscale('log')
ax.set_title('Nearest Neighbour Distance Distribution')
ax.set_xlabel('Nearest Neighbour Distance (log scale)')
ax.set_ylabel('Count (log scale)')
ax.legend(title='Codec')
plt.yscale('log')
plt.xscale('log')
fig.tight_layout()
# place legend outside the plot at the bottom and aligned with the x axis
plt.legend(bbox_to_anchor=(0., -0.4, 1., .102), loc='lower center', ncol=3, mode="expand", borderaxespad=0.)
plt.show()

# %%
# plot the amount of observations in df_out that the distances are above 40 for each label
df_out.groupby('label').apply(lambda x: (x['distance'] > 40).sum()).plot(kind='bar', title='Number of observations with distance > 40nm', color = 'k')


# %%
# plot the amount of observations in df_n_locs for the different labels
df_n_locs = df_out.groupby('label').count()
# rename distance to n_locs
df_n_locs.rename(columns={'distance': 'n_locs'}, inplace=True)
df_n_locs['n_locs'].plot(kind='bar', title='Number of localizations', color = 'k')

# %%
df_n_locs
# %%
plt.stem(df_n_locs.index, df_n_locs['n_locs'], use_line_collection=False)
# tilt x axis labels 45 degrees
plt.xticks(rotation=45)
plt.title('Total number of localizations')
plt.show()


# %%
# create dotplot of the metrics
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
ax = sns.stripplot(x="metric", y="value", hue="label", data=df_stats_out, dodge=True, size=15)

# Remove the legend
ax.get_legend().remove()

# Tilt the x-axis labels 45 degrees
plt.xticks(rotation=45, ha='right')

ax.set(yscale="log")
ax.set_ylabel('Value (log)')
ax.set_xlabel(' ')
plt.legend(bbox_to_anchor=(0., -.6, 1., .102), loc='lower center', ncol=3, mode="expand", borderaxespad=0.)


plt.show()
# %%
# facet grid plot
g = sns.FacetGrid(df_stats_out, col="metric", hue="label", sharey=False)
g.map_dataframe(sns.stripplot, x="label", y="value")
g.add_legend()

# set the y-axis scale for each facet
g.set(yscale="log")

# set the x-axis label for each facet
# g.set_xlabels('X Values')
g.set(xticklabels=[])
g.set(xlabel=None)
# sns.set_xlabel('')

# set the y-axis label for each facet
g.set_ylabels('Value (log)')

# set the title for each facet
g.set_titles("{col_name}")

# show the plot
plt.show()
# %%
