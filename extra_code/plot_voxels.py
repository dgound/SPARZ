#%%
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
def pointcloud_to_voxels(data, voxel_size):
    """
    Convert point cloud data to a set of occupied voxels.
    
    Parameters:
    - data: DataFrame or numpy array containing x, y, z coordinates.
    - voxel_size: The size of the voxel used for binning the points.
    
    Returns:
    - A set of tuples representing the occupied voxel indices.
    """
    # Compute voxel indices for each point
    voxel_indices = (data / voxel_size).astype(int)
    
    # Convert to set for unique voxel indices
    occupied_voxels = set([tuple(row) for row in voxel_indices])
    
    return occupied_voxels


def visualize_voxels(voxel_indices_set, voxel_size=0.5, color = 'black', alpha = 0.5):
    """
    Visualize voxels in a 3D plot.
    
    Parameters:
    - voxel_indices_set: Set of tuples containing voxel indices.
    - voxel_size: The size of each voxel (cube).
    
    Returns:
    - A 3D plot showing the occupied voxels.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    corner_coordinates = np.array(list(voxel_indices_set)) * voxel_size

    ax.bar3d(corner_coordinates[:, 0], corner_coordinates[:, 1], corner_coordinates[:, 2],
             voxel_size, voxel_size, voxel_size,
             color=color,
             alpha=alpha,
             shade=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    fig.canvas.draw()
    return ax

#%%
file_raw = '/Users/dimos/Nir_et_al_raw16b.h5r'
file_lev0 = '/Users/dimos/Nir_et_al_lev0.h5r'
# %%
with h5py.File(file_raw, 'r') as f:
    dataset = f["FitResults"]
    data_raw = dataset[:]
# %%
data_raw
# %%
raw_locs = np.column_stack((data_raw['fitResults']['x0'], data_raw['fitResults']['y0'], data_raw['fitResults']['z0']))
# %%
with h5py.File(file_lev0, 'r') as f:
    dataset = f["FitResults"]
    data_lev0 = dataset[:]
# %%
lev0_locs = np.column_stack((data_lev0['fitResults']['x0'], data_lev0['fitResults']['y0'], data_lev0['fitResults']['z0']))

# %%
p_raw = pointcloud_to_voxels(raw_locs, 0.5)
# %%
p_lev0 = pointcloud_to_voxels(lev0_locs, 0.5)
# %%

# %%
import plotly.graph_objects as go

def visualize_voxels_plotly(voxel_indices_set, voxel_size=0.5, color='rgba(0, 0, 0, 0.5)'):
    """
    Visualize voxels in a 3D plot using Plotly.
    
    Parameters:
    - voxel_indices_set: Set of tuples containing voxel indices.
    - voxel_size: The size of each voxel (cube).
    
    Returns:
    - A 3D plot showing the occupied voxels.
    """

    corner_coordinates = np.array(list(voxel_indices_set)) * voxel_size

    x, y, z = corner_coordinates.T

    # create a Mesh3D instance for each voxel
    fig = go.Figure(data=
                    go.Mesh3d(
        # 8 vertices, 12 face definition for a cube
        x=np.append(x, x + voxel_size),
        y=np.append(y, y + voxel_size),
        z=np.append(z, z + voxel_size),
        color='rgba(0, 0, 0, 0.5)', # matches color parameter in initial function
        opacity=0.5))

    fig.update_layout(scene=dict(xaxis_title='X',
                                yaxis_title='Y',
                                zaxis_title='Z'))
    fig.show()
# %%
visualize_voxels_plotly(p_raw)
# %%
from mpl_toolkits.mplot3d import Axes3D

def plot_voxels(occupied_voxels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
    
    x, y, z = zip(*occupied_voxels) # extract all x, y and z coordinates
    
    ax.scatter(x, y, z, c='b', marker='o') # Plot all points as a scatter plot

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()
# %%
plot_voxels(np.array(list(p_raw)))
# %%
np.array(list(p_raw)).shape
# %%
def visualize_voxels(voxel_indices_set, voxel_size=0.5, color = 'black', alpha = 0.5):
    """
    Visualize voxels in a 3D plot.
    
    Parameters:
    - voxel_indices_set: Set of tuples containing voxel indices.
    - voxel_size: The size of each voxel (cube).
    
    Returns:
    - A 3D plot showing the occupied voxels.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    for voxel in voxel_indices_set:
        # Extract the corner of the voxel based on index and voxel size
        corner = np.array(voxel) * voxel_size
        # Plot a cube at the corner position with given voxel size
        ax.bar3d(corner[0], corner[1], corner[2], 
                 voxel_size, voxel_size, voxel_size, 
                 shade=True,
                 color = color,
                 alpha=alpha)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()
# %%
visualize_voxels(p_raw)
# %%
p_raw[:10]
# %%

# %%
visualize_voxels_plotly(p_raw)
# %%
def visualize_voxels_vec(voxel_indices_set, voxel_size=0.5, color = 'black', alpha = 0.1):
    """
    Visualize voxels in a 3D plot.
    
    Parameters:
    - voxel_indices_set: Set of tuples containing voxel indices.
    - voxel_size: The size of each voxel (cube).
    
    Returns:
    - A 3D plot showing the occupied voxels.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    corner_coordinates = np.array(list(voxel_indices_set)) * voxel_size
    print ('corner_coordinates', corner_coordinates.shape)
    ax.bar3d(corner_coordinates[:, 0], corner_coordinates[:, 1], corner_coordinates[:, 2],
             voxel_size, voxel_size, voxel_size,
             color=color,
             alpha=alpha,
             shade=True)

    # for voxel in voxel_indices_set:
    #     # Extract the corner of the voxel based on index and voxel size
    #     corner = np.array(voxel) * voxel_size
    #     # Plot a cube at the corner position with given voxel size
    #     ax.bar3d(corner[0], corner[1], corner[2], 
    #              voxel_size, voxel_size, voxel_size, 
    #              shade=True,
    #              color = color,
    #              alpha=alpha)
    
    max_range = np.array([corner_coordinates[:, i].max() - corner_coordinates[:, i].min() for i in range(3)]).max()
    x_mid = (corner_coordinates[:, 0].max() + corner_coordinates[:, 0].min()) * 0.5
    y_mid = (corner_coordinates[:, 1].max() + corner_coordinates[:, 1].min()) * 0.5
    z_mid = (corner_coordinates[:, 2].max() + corner_coordinates[:, 2].min()) * 0.5

    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()
# %%
p_raw = pointcloud_to_voxels(raw_locs, 100)
# %%
p_lev0 = pointcloud_to_voxels(lev0_locs, 250)
visualize_voxels_vec(p_raw,voxel_size=10)
# %%
df = pd.read_csv('/Users/dimos/20230917_raw_ssim.csv')
# %%
gc.collect()
#%%
df.codec.value_counts()
# %%
import pyvista as pv
import numpy as np

# Set the random seed for reproducibility
np.random.seed(0)

# Creating ten voxels randomly placed on the unit cube
voxels = np.random.rand(10, 3)

# creating the plotter
p = pv.Plotter()

# Loop through and plot each voxel
for voxel in voxels:
    # Create each voxel as a small cube
    cube = pv.Cube(center=voxel, x_length=0.1, y_length=0.1, z_length=0.1)
    p.add_mesh(cube,color='blue')

# Show the plotter
p.show_grid()
p.show()
# %%
# Set the random seed for reproducibility
np.random.seed(0)

# Creating ten blue voxels randomly placed on the unit cube
voxels_blue = np.random.rand(50, 3)

# Offset for red voxels
offset = 0.25  # You can adjust this value to change overlap degree

# Creating ten red voxels, placed relative to the blue ones
voxels_red = voxels_blue + offset

# Create the plotter
p = pv.Plotter()


# Loop through and plot each blue voxel
# for voxel in voxels_blue:
#     # Create each voxel as a small cube
#     cube = pv.Cube(center=voxel, x_length=0.1, y_length=0.1, z_length=0.1)
#     p.add_mesh(cube, color='blue', opacity=0.5)

# Loop through and plot each red voxel
for voxel in voxels_red:
    # Create each voxel as a small cube
    cube = pv.Cube(center=voxel, x_length=0.1,     y_length=0.1, z_length=0.1)
    p.add_mesh(cube, color='red', opacity=0.5)

labels = dict(zlabel='', xlabel='', ylabel='',ticks=None)
_ = p.add_axes(line_width=5, labels_off=True)

p.show_grid(**labels)
# p.add_axes(**labels)
p.show()
p.screenshot("voxels_red.png", window_size=[1500, 1500])
# %%

# %%
np.random.seed(0)

# Creating ten blue points randomly placed on the unit cube
points_blue = np.random.rand(50, 3)

# Offset for red points
offset = 0.02  # You can adjust this value to change overlap degree

# Creating ten red points, placed relative to the blue ones
points_red = points_blue + offset

# Create the plotter
p = pv.Plotter()

# Create and plot each point cloud
cloud_blue = pv.PolyData(points_blue)
cloud_red = pv.PolyData(points_red)

# p.add_mesh(cloud_blue, color='blue', render_points_as_spheres=True, point_size=15)
p.add_mesh(cloud_red, color='red', render_points_as_spheres=True, point_size=15)

# Hide the axes labels and ticks
p.show_axes = False

# Show the plotter
labels = dict(zlabel='', xlabel='', ylabel='',ticks=None)
_ = p.add_axes(line_width=5, labels_off=True)

p.show_grid(**labels)
# p.show_grid()
p.show()
p.screenshot("points_red.png", window_size=[1500, 1500])
# %%
