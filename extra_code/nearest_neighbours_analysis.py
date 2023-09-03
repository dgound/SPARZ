#%%
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
from h5r_functions import h5r_to_df
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import distance

#%%
import multiprocessing
num_cores = multiprocessing.cpu_count()
num_cores = int(round(num_cores * 0.6, 0))

print(num_cores)

#%%
from joblib import Parallel, delayed

def compute_single_row(row, gt, n_neighbors):
    distances = np.array([distance.euclidean(row, gt_row) for gt_row in gt.values])
    sorted_indices = np.argsort(distances)
    return distances[sorted_indices[:n_neighbors]], sorted_indices[:n_neighbors]


def compute_nearest_neighbours_parallel(df, gt, n_neighbors=1, n_jobs=-1):
    # Initialize arrays to hold the results
    min_distances = np.zeros((df.shape[0], n_neighbors))
    min_indices = np.zeros((df.shape[0], n_neighbors), dtype=int)

    # Compute the minimum distances and indices for each row in df
    results = Parallel(n_jobs=n_jobs)(delayed(compute_single_row)(row, gt, n_neighbors) for _, row in df.iterrows())

    # Assign the results to the arrays
    for i, (distances, indices) in enumerate(results):
        min_distances[i] = distances
        min_indices[i] = indices

    return min_distances, min_indices





# %%
# Function to compute the n_neighbors nearest neighbours between two dataframes
def compute_nearest_neighbours(df, gt, n_neighbors=1):
    # Initialize arrays to hold the results
    min_distances = np.zeros((df.shape[0], n_neighbors))
    min_indices = np.zeros((df.shape[0], n_neighbors), dtype=int)
    
    # Compute the minimum distances and indices for each row in df
    for i, row in df.iterrows():
        distances = np.array([distance.euclidean(row, gt_row) for gt_row in gt.values])
        sorted_indices = np.argsort(distances)
        min_distances[i] = distances[sorted_indices[:n_neighbors]]
        min_indices[i] = sorted_indices[:n_neighbors]
    
    return min_distances, min_indices

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
# nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])
# distances, indices = nbrs.kneighbors(locs[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])

#%%
col_select = ['fitResults_x0',	'fitResults_y0', 'fitResults_z0']
df_query = locs.reset_index()[col_select]
gt = ground_truth.reset_index()[col_select]
n_neighbors = 1
# distances, indices = compute_nearest_neighbours(df_query, gt, n_neighbors)
distances, indices = compute_nearest_neighbours_parallel(df_query, gt, n_neighbors=1, n_jobs=num_cores)

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

# nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])
# distances, indices = nbrs.kneighbors(locs[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])
col_select = ['fitResults_x0',	'fitResults_y0', 'fitResults_z0']
df_query = locs.reset_index()[col_select]
gt = ground_truth.reset_index()[col_select]
n_neighbors = 1
# distances, indices = compute_nearest_neighbours(df_query, gt, n_neighbors)
distances, indices = compute_nearest_neighbours_parallel(df_query, gt, n_neighbors=1, n_jobs=num_cores)
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

# nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])
# distances, indices = nbrs.kneighbors(locs[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])

col_select = ['fitResults_x0',	'fitResults_y0', 'fitResults_z0']
df_query = locs.reset_index()[col_select]
gt = ground_truth.reset_index()[col_select]
n_neighbors = 1
# distances, indices = compute_nearest_neighbours(df_query, gt, n_neighbors)
distances, indices = compute_nearest_neighbours_parallel(df_query, gt, n_neighbors=1, n_jobs=num_cores)

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

# nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])
# distances, indices = nbrs.kneighbors(locs[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])

col_select = ['fitResults_x0',	'fitResults_y0', 'fitResults_z0']
df_query = locs.reset_index()[col_select]
gt = ground_truth.reset_index()[col_select]
n_neighbors = 1
# distances, indices = compute_nearest_neighbours(df_query, gt, n_neighbors)
distances, indices = compute_nearest_neighbours_parallel(df_query, gt, n_neighbors=1, n_jobs=num_cores)

H264_dist_gt = distances

locs['nn_distance'] = distances
plt.scatter(locs["fitResults_x0"], locs["fitResults_y0"], s=1, c = locs["nn_distance"])
plt.colorbar()
plt.show()

# %%
# X264 vs Raw
ground_truth = h5r_to_df(filepath='/Users/alioutas/Desktop/Desktop/sequence-MT0.N1.HD-BP_raw.h5r')

filepath = '/Users/alioutas/Desktop/Desktop/sequence-MT0.N1.HD-BP_h264.h5r'
locs = h5r_to_df(filepath=filepath)

locs = locs[(locs["fitError_x0"] > 0) & (locs["fitError_x0"] < 30) & (locs["fitError_y0"] > 0) & (locs["fitError_y0"] < 30) & (locs["fitResults_A"] > 5) & (locs["fitResults_A"] < 100000)]

# nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])
# distances, indices = nbrs.kneighbors(locs[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])

col_select = ['fitResults_x0',	'fitResults_y0', 'fitResults_z0']
df_query = locs.reset_index()[col_select]
gt = ground_truth.reset_index()[col_select]
n_neighbors = 1
# distances, indices = compute_nearest_neighbours(df_query, gt, n_neighbors)
distances, indices = compute_nearest_neighbours_parallel(df_query, gt, n_neighbors=1, n_jobs=num_cores)

H264_dist_raw = distances

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

# nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])
# distances, indices = nbrs.kneighbors(locs[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])

col_select = ['fitResults_x0',	'fitResults_y0', 'fitResults_z0']
df_query = locs.reset_index()[col_select]
gt = ground_truth.reset_index()[col_select]
n_neighbors = 1
# distances, indices = compute_nearest_neighbours(df_query, gt, n_neighbors)
distances, indices = compute_nearest_neighbours_parallel(df_query, gt, n_neighbors=1, n_jobs=num_cores)

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

# nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])
# distances, indices = nbrs.kneighbors(locs[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])

col_select = ['fitResults_x0',	'fitResults_y0', 'fitResults_z0']
df_query = locs.reset_index()[col_select]
gt = ground_truth.reset_index()[col_select]
n_neighbors = 1
# distances, indices = compute_nearest_neighbours(df_query, gt, n_neighbors)
distances, indices = compute_nearest_neighbours_parallel(df_query, gt, n_neighbors=1, n_jobs=num_cores)

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

# nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])
# distances, indices = nbrs.kneighbors(locs[["fitResults_x0", "fitResults_y0", "fitResults_z0"]])

col_select = ['fitResults_x0',	'fitResults_y0', 'fitResults_z0']
df_query = locs.reset_index()[col_select]
gt = ground_truth.reset_index()[col_select]
n_neighbors = 1
# distances, indices = compute_nearest_neighbours(df_query, gt, n_neighbors)
distances, indices = compute_nearest_neighbours_parallel(df_query, gt, n_neighbors=1, n_jobs=num_cores)

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

df_raw = pd.concat([ffv1_dist_raw, X265_dist_raw, H264_dist_raw])
df_raw.columns = ['distances', 'codec']

#%%
# plot the boxplot of df
import seaborn as sns
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="codec", y="distances", data=df_raw)
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

df_gt = pd.concat([ffv1_dist_gt, X265_dist_gt, H264_dist_gt])
df_gt.columns = ['distances', 'codec']

#%%
# plot histograms of df
import seaborn as sns
sns.set_theme(style="whitegrid")


#%%
# plot the bocplots of df
import seaborn as sns
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="codec", y="distances", data=df_gt)
#ax.set(yscale="log")
ax.set_ylabel('Distance from GT')

plt.show()


# %%
# Plot histograms of the distances
import seaborn as sns
sns.kdeplot(data=df_raw, hue="codec", bw_adjust=.1, x="distances", color ="codec" , fill=False, common_norm=True, alpha=.4, linewidth=2, log_scale=True)


# %%

plt.hist(min_distances, bins=100, log=True)
plt.show()
# %%
sns.kdeplot(data=df, hue = 'codec', bw_adjust=.1, fill=False, common_norm=True, alpha=.8, linewidth=2, log_scale=True)

# %%
data = []