#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
# sys.path.append('/Users/alioutas/Library/CloudStorage/GoogleDrive-alioutas@gmail.com/My Drive/GitHub/SPARZ3/extra_code')
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


#####################################################################
# read in saved dataframes
#####################################################################
#%%
import pandas as pd

# Read in saved dataframes
# ********************* Uncomment either VS_RAW or VS_COMPRESSED depending on the data you want to use *********************

#VS_RAW
# df_out = pd.read_csv('/Volumes/T7/compression_data/data_compression_localizations/Nir_et_al/output/nir_etal_vs_raw_20230922_sparz_lev1_k11_rt55_distances.csv')
# df_stats_out = pd.read_csv('/Volumes/T7/compression_data/data_compression_localizations/Nir_et_al/output/nir_etal_vs_raw_20230922_sparz_lev1_k11_rt55_metrics.csv')

#VS_COMPRESSED
df_out = pd.read_csv('/Users/dimos/Dropbox (Lab at Large)/data_compression_localizations/Nir_et_al/output/nir_etal_vs_compressed_20230922_sparz_lev1_k11_rt55_distances.csv')
df_stats_out = pd.read_csv('/Users/dimos/Dropbox (Lab at Large)/data_compression_localizations/Nir_et_al/output/nir_etal_vs_compressed_20230922_sparz_lev1_k11_rt55_metrics.csv')


# %%
# KDE plot of distances with seaborn

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
# Histogram of distances (log scale)

# LOG SCALE
fig, ax = plt.subplots()
for codec, group in df_out.groupby('label'):
    ax.hist(group['distance'], bins=60, alpha=0.6, label=codec)
ax.axvline(x=40, color='r', linestyle='--', linewidth=2)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_title('Nearest Neighbour Distance Distribution')
ax.set_xlabel('Nearest Neighbour Distance (log scale)')
ax.set_ylabel('Count (log scale)')
ax.legend(title='Codec')
# plt.yscale('log')
# plt.xscale('log')
fig.tight_layout()
#limit x to 200
ax.set_xlim(0,200)
# place legend outside the plot at the bottom and aligned with the x axis
plt.legend(bbox_to_anchor=(0., -0.4, 1., .102), loc='lower center', ncol=3, mode="expand", borderaxespad=0.)
plt.show()

# %%
# Histogram of distances (no log scale)

# NO LOG SCALE
fig, ax = plt.subplots()
for codec, group in df_out.groupby('label'):
    ax.hist(group['distance'], bins=60, alpha=0.6, label=codec)
ax.axvline(x=40, color='r', linestyle='--', linewidth=2)
# ax.set_yscale('log')
# ax.set_xscale('log')
ax.set_title('Nearest Neighbour Distance Distribution')
ax.set_xlabel('Nearest Neighbour Distance')
ax.set_ylabel('Count')
ax.legend(title='Codec')
# plt.yscale('log')
# plt.xscale('log')
fig.tight_layout()
#limit x to 200
ax.set_xlim(0,200)
# place legend outside the plot at the bottom and aligned with the x axis
plt.legend(bbox_to_anchor=(0., -0.4, 1., .102), loc='lower center', ncol=3, mode="expand", borderaxespad=0.)
plt.show()

#%%
# Separate histograms for each label

fig, axs = plt.subplots(len(df_out['label'].unique()), 1, figsize=(8, 2*len(df_out['label'].unique())))
colors = plt.cm.tab20(np.linspace(0, 1, len(df_out['label'].unique())))
for i, (label, group) in enumerate(df_out.groupby('label')):
    axs[i].hist(group['distance'], bins=600, alpha=0.6, color=colors[i])
    axs[i].axvline(x=40, color='r', linestyle='--', linewidth=2)
    axs[i].set_title(label)
    axs[i].set_ylabel('Count')
    axs[i].set_xlim(0, 120)
    # comment the following line to make y independent for each plot
    axs[i].set_ylim(0, 30000)
    if i == len(df_out['label'].unique()) - 1:
        axs[i].set_xlabel('Nearest Neighbour Distance')
    else:
        axs[i].set_xticklabels([])
    fig.suptitle('Nearest Neighbour Distance Distribution', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#%%
# Histogram with seaborn

import seaborn as sns

sns.set_theme(style="whitegrid")
g = sns.histplot(data=df_out, x="distance", hue="label", bins=100, alpha=0.6)#, multiple="stack")
g.axvline(x=40, color='r', linestyle='--', linewidth=2)
g.set( title="Nearest Neighbour Distance Distribution", xlabel="Nearest Neighbour Distance (log scale)", ylabel="Count (log scale)") #xscale="log", yscale="log",
g.legend(title='Codec', bbox_to_anchor=(0., -0.4, 1., .102), loc='lower center', ncol=3, mode="expand", borderaxespad=0.)
sns.despine()
plt.show()

#%%
#%%
# Boxplot of distances for each label

sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="label", y="distance", data=df_out[(df_out.label!='ffv1') & (df_out.label!='x265') & (df_out.label!='h264')])
ax.set(yscale="log")
ax.set_ylabel('Distance (log)')
plt.axhline(y=40, color='r', linestyle='--', linewidth=2)
# the name of the codec should be in 45 degrees and aligned with the x axis
plt.xticks(rotation=45, ha='right')
plt.show()


# %%
# Plot the amount of observations in df_out that the distances are above 40 for each label
df_out.groupby('label').apply(lambda x: (x['distance'] > 40).sum()).plot(kind='bar', title='Number of observations with distance > 40nm', color = 'k')


# %%
# Plot the amount of observations in df_n_locs for the different labels
df_n_locs = df_out.groupby('label').count()
# rename distance to n_locs
df_n_locs.rename(columns={'distance': 'n_locs'}, inplace=True)
df_n_locs['n_locs'].plot(kind='bar', title='Number of localizations', color = 'k')

# # %%
# df_n_locs
# %%
# Stem plot of the total number of localizations

plt.stem(df_n_locs.index, df_n_locs['n_locs'], use_line_collection=False)
# tilt x axis labels 45 degrees
plt.xticks(rotation=45)
plt.title('Total number of localizations')
plt.show()


# %%
# Dotplot of the metrics
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
# Facet grid plot for the metrics
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
# Stem plot of the total number of localizations below 40nm

n_below = df_out[df_out['distance'] <= 40].groupby('label').count()
# make a stem plot where the x axis is the label and the y axis is the number of observations below 40
plt.stem(n_below.index, n_below['distance'], use_line_collection=False)
plt.xticks(rotation=45)
plt.title('Total number of localizations below 40nm')
plt.show()


# %%
# Bar plot of the fraction of localizations below 40nm
n_below = df_out[df_out['distance'] <= 40].groupby('label').count()
n_total = df_out.groupby('label').count()
n_below_fraction = n_below/n_total
plt.bar(n_below_fraction.index, n_below_fraction['distance'])
plt.xticks(rotation=45)
plt.title('Fraction of localizations below 40nm')
plt.show()
# %%
# Ordered barplot for the Jaccard index
import seaborn as sns
label_order = ['ffv1', 'h264', 'x265', 'sparz_lev0_k11_rt55', 'sparz_lev1_k11_rt55', 'sparz_lev2_k11_rt55', 'sparz_lev3_k11_rt55']
sns.barplot(x='label', y='value', hue='metric', data=df_stats_out[df_stats_out['metric'] == 'Jaccard'], order=label_order)
plt.xticks(rotation=45)
plt.title('Jaccard index')
plt.legend().set_visible(False)
plt.show()
# %%
