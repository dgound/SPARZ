#%%
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

#%%
from h5r_functions import h5r_to_df

#%%
#Import localizations from h5r file
filepath = '/Users/alioutas/Desktop/Desktop/sequence-MT0.N1.HD-BP_raw.h5r'
locs = h5r_to_df(filepath=filepath)

#%%
# filter localizations
# fitError_x0 between 0 and 30
# fitError_y0 between 0 and 30
# fitResults_A between 5 and 100000
locs = locs[(locs["fitError_x0"] > 0) & (locs["fitError_x0"] < 30) & (locs["fitError_y0"] > 0) & (locs["fitError_y0"] < 30) & (locs["fitResults_A"] > 5) & (locs["fitResults_A"] < 100000)]

#%%
min(locs.fitError_x0), max(locs.fitError_x0)

#%%
min(locs.fitError_y0), max(locs.fitError_y0)
#%%
min(locs.fitResults_A), max(locs.fitResults_A)

#%%
# import ground truth
#ground_truth = pd.read_csv("https://www.dropbox.com/s/alfg6rsq0wby1dp/activations_gt.csv?dl=1")
#ground_truth.columns = ["id", "frame", "fitResults_x0", "fitResults_y0", "fitResults_z0", "intensity..photon."]
# read xlsx file with header
ground_truth = pd.read_excel('/Users/alioutas/Desktop/Desktop/activations_gt.xlsx', header=0)
ground_truth.columns = ["id", "frame", "fitResults_x0", "fitResults_y0", "fitResults_z0", "intensity..photon."]
ground_truth
# %%
# calculate nearest neighbours between the localizations and the ground truth
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])
distances, indices = nbrs.kneighbors(locs[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])

#%%
# # plot localizations from function filtered
subsampled_locs = locs[(locs["fitResults_x0"] > 0) & (locs["fitResults_x0"] < 6500) & (locs["fitResults_y0"] > 0) & (locs["fitResults_y0"] < 6500)]
plt.scatter(subsampled_locs["fitResults_x0"], subsampled_locs["fitResults_y0"],s=1, c = subsampled_locs["fitResults_z0"])
plt.colorbar()
plt.show()

#%%
plt.scatter(locs["fitResults_x0"], locs["fitResults_y0"],s=1, c = locs["fitResults_z0"])
plt.colorbar()
plt.show()
#%%
# # plot localizations from function
plt.scatter(ground_truth["fitResults_x0"], ground_truth["fitResults_y0"], s=1,c=ground_truth["fitResults_z0"])
plt.colorbar()
plt.show()

#%%
# plot localizations from jupyter notebook
df_jupyter = pd.read_csv('/Users/alioutas/Desktop/random/output.csv')
plt.scatter(df_jupyter["fitResults_x0"], df_jupyter["fitResults_y0"], s=1,c = df_jupyter["fitResults_z0"])
plt.colorbar()
plt.show()

#%%
# plot filtered distances
locs['nn_distance'] = distances
subsampled_locs = locs[(locs["fitResults_x0"] > 0) & (locs["fitResults_x0"] < 6500) & (locs["fitResults_y0"] > 0) & (locs["fitResults_y0"] < 6500)]
plt.scatter(subsampled_locs["fitResults_x0"], subsampled_locs["fitResults_y0"], s=1, c = subsampled_locs["nn_distance"])
plt.colorbar()
plt.show()



# %%
locs[~((locs["fitResults_x0"] > 0) & (locs["fitResults_x0"] < 6500) & (locs["fitResults_y0"] > 0) & (locs["fitResults_y0"] < 6500))].index
# %%
locs.loc[locs[~((locs["fitResults_x0"] > 0) & (locs["fitResults_x0"] < 6500) & (locs["fitResults_y0"] > 0) & (locs["fitResults_y0"] < 6500))].index]
# %%
len(locs.loc[locs[~((locs["fitResults_x0"] > 0) & (locs["fitResults_x0"] < 6500) & (locs["fitResults_y0"] > 0) & (locs["fitResults_y0"] < 6500))].index])

# %%
# FFV1 vs GT
ground_truth = pd.read_excel('/Users/alioutas/Desktop/Desktop/activations_gt.xlsx', header=0)
ground_truth.columns = ["id", "frame", "fitResults_x0", "fitResults_y0", "fitResults_z0", "intensity..photon."]

filepath = '/Users/alioutas/Desktop/Desktop/sequence-MT0.N1.HD-BP_ffv1.h5r'
locs = h5r_to_df(filepath=filepath)

locs = locs[(locs["fitError_x0"] > 0) & (locs["fitError_x0"] < 30) & (locs["fitError_y0"] > 0) & (locs["fitError_y0"] < 30) & (locs["fitResults_A"] > 5) & (locs["fitResults_A"] < 100000)]

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])
distances, indices = nbrs.kneighbors(locs[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])

ffv1_dist_gt = distances

locs['nn_distance'] = distances
plt.scatter(locs["fitResults_x0"], locs["fitResults_y0"], s=1, c = locs["nn_distance"])
plt.colorbar()
plt.show()

# %%
# FFV1 vs Raw
ground_truth = h5r_to_df(filepath='/Users/alioutas/Desktop/Desktop/sequence-MT0.N1.HD-BP_raw.h5r')

filepath = '/Users/alioutas/Desktop/Desktop/sequence-MT0.N1.HD-BP_ffv1.h5r'
locs = h5r_to_df(filepath=filepath)

locs = locs[(locs["fitError_x0"] > 0) & (locs["fitError_x0"] < 30) & (locs["fitError_y0"] > 0) & (locs["fitError_y0"] < 30) & (locs["fitResults_A"] > 5) & (locs["fitResults_A"] < 100000)]

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])
distances, indices = nbrs.kneighbors(locs[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])

ffv1_dist_raw = distances

locs['nn_distance'] = distances
plt.scatter(locs["fitResults_x0"], locs["fitResults_y0"], s=1, c = locs["nn_distance"])
plt.colorbar()
plt.show()
# %%
# H264 vs GT
ground_truth = pd.read_excel('/Users/alioutas/Desktop/Desktop/activations_gt.xlsx', header=0)
ground_truth.columns = ["id", "frame", "fitResults_x0", "fitResults_y0", "fitResults_z0", "intensity..photon."]

filepath = '/Users/alioutas/Desktop/Desktop/sequence-MT0.N1.HD-BP_H264.h5r'
locs = h5r_to_df(filepath=filepath)

locs = locs[(locs["fitError_x0"] > 0) & (locs["fitError_x0"] < 30) & (locs["fitError_y0"] > 0) & (locs["fitError_y0"] < 30) & (locs["fitResults_A"] > 5) & (locs["fitResults_A"] < 100000)]

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])
distances, indices = nbrs.kneighbors(locs[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])

H264_dist_gt = distances

locs['nn_distance'] = distances
plt.scatter(locs["fitResults_x0"], locs["fitResults_y0"], s=1, c = locs["nn_distance"])
plt.colorbar()
plt.show()

# %%
# X265 vs Raw
ground_truth = h5r_to_df(filepath='/Users/alioutas/Desktop/Desktop/sequence-MT0.N1.HD-BP_raw.h5r')

filepath = '/Users/alioutas/Desktop/Desktop/sequence-MT0.N1.HD-BP_h265.h5r'
locs = h5r_to_df(filepath=filepath)

locs = locs[(locs["fitError_x0"] > 0) & (locs["fitError_x0"] < 30) & (locs["fitError_y0"] > 0) & (locs["fitError_y0"] < 30) & (locs["fitResults_A"] > 5) & (locs["fitResults_A"] < 100000)]

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])
distances, indices = nbrs.kneighbors(locs[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])

X265_dist_raw = distances

locs['nn_distance'] = distances
plt.scatter(locs["fitResults_x0"], locs["fitResults_y0"], s=1, c = locs["nn_distance"])
plt.colorbar()
plt.show()

# %%
# X265 vs GT
ground_truth = pd.read_excel('/Users/alioutas/Desktop/Desktop/activations_gt.xlsx', header=0)
ground_truth.columns = ["id", "frame", "fitResults_x0", "fitResults_y0", "fitResults_z0", "intensity..photon."]

filepath = '/Users/alioutas/Desktop/Desktop/sequence-MT0.N1.HD-BP_h265.h5r'
locs = h5r_to_df(filepath=filepath)

locs = locs[(locs["fitError_x0"] > 0) & (locs["fitError_x0"] < 30) & (locs["fitError_y0"] > 0) & (locs["fitError_y0"] < 30) & (locs["fitResults_A"] > 5) & (locs["fitResults_A"] < 100000)]

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])
distances, indices = nbrs.kneighbors(locs[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])

X265_dist_gt = distances

locs['nn_distance'] = distances
plt.scatter(locs["fitResults_x0"], locs["fitResults_y0"], s=1, c = locs["nn_distance"])
plt.colorbar()
plt.show()

# %%
# X265 vs Raw
ground_truth = h5r_to_df(filepath='/Users/alioutas/Desktop/Desktop/sequence-MT0.N1.HD-BP_raw.h5r')

filepath = '/Users/alioutas/Desktop/Desktop/sequence-MT0.N1.HD-BP_h265.h5r'
locs = h5r_to_df(filepath=filepath)

locs = locs[(locs["fitError_x0"] > 0) & (locs["fitError_x0"] < 30) & (locs["fitError_y0"] > 0) & (locs["fitError_y0"] < 30) & (locs["fitResults_A"] > 5) & (locs["fitResults_A"] < 100000)]

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])
distances, indices = nbrs.kneighbors(locs[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])

X265_dist_raw = distances

locs['nn_distance'] = distances
plt.scatter(locs["fitResults_x0"], locs["fitResults_y0"], s=1, c = locs["nn_distance"])
plt.colorbar()
plt.show()
# %%
# boxplot for the distances: ffv1_dist_raw, X265_dist_raw, H264_dist_raw
import matplotlib.pyplot as plt
import pandas as pd

#%%
ffv1_dist_raw = pd.DataFrame(ffv1_dist_raw)
ffv1_dist_raw['codec'] = 'ffv1'
X265_dist_raw = pd.DataFrame(X265_dist_raw)
X265_dist_raw['codec'] = 'X265'
H264_dist_raw = pd.DataFrame(H264_dist_raw)
H264_dist_raw['codec'] = 'H264'

df = pd.concat([ffv1_dist_raw, X265_dist_raw, H264_dist_raw])
df.columns = ['distances', 'codec']

#%%
# plot the boxplot of df
import seaborn as sns
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="codec", y="distances", data=df)
ax.set(yscale="log")
ax.set_ylabel('Distance (log)')

plt.show()

#%%
# combine ffv1_dist_raw, X265_dist_raw, H264_dist_raw in the most efficiant way to plot them in a boxplot
ffv1_dist_gt = pd.DataFrame(ffv1_dist_gt)
ffv1_dist_gt['codec'] = 'ffv1'
X265_dist_gt = pd.DataFrame(X265_dist_gt)
X265_dist_gt['codec'] = 'X265'
H264_dist_gt = pd.DataFrame(H264_dist_gt)
H264_dist_gt['codec'] = 'H264'

df = pd.concat([ffv1_dist_gt, X265_dist_gt, H264_dist_gt])
df.columns = ['distances', 'codec']

#%%
# plot the boxplot of df
import seaborn as sns
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="codec", y="distances", data=df)
#ax.set(yscale="log")
ax.set_ylabel('Distance from GT')

plt.show()


# %%
