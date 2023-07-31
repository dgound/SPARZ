#%%
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

#%%
from h5r_functions import h5r_to_df

#%%
#Import localizations from h5r file
filepath = '/Users/alioutas/Desktop/random/sequence-MT0.N1.HD-BP.h5r'
locs = h5r_to_df(filepath=filepath)

#%%
# import ground truth
ground_truth = pd.read_csv("https://www.dropbox.com/s/alfg6rsq0wby1dp/activations_gt.csv?dl=1")
ground_truth.columns = ["id", "frame", "fitResults_x0", "fitResults_y0", "fitResults_z0", "intensity..photon."]

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
