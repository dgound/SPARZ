# Note book of the SPARZ pipeline that loads data, runs the compression and then decompresses the data.

#%%
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
# from skimage import imread_collection
import glob
import time
import sparse
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
import dask.array as da
import dask_image.imread
from dask import delayed
from skvideo.io import FFmpegWriter
from reader import imread as vimread
import dask
from dask import compute
import gc
from tqdm import tqdm
from SPARZ import SPARZIP, SPARUNZIP






#%%
#path1 = '/Users/dimos/raw_image_compression/tubulin_biplane/COS-7_Tubulin_SOFI_Flip565_biplane_reflected.tiff'
#path2 = '/Users/dimos/raw_image_compression/tubulin_biplane/COS-7_Tubulin_SOFI_Flip565_biplane_transmitted.tiff'

# path2 = '/Users/alioutas/Dropbox (HMS)/data_compression/dimos/compression/nir_et_al/raw_bp2/img_01_01*.tiff'
# path1 = '/Users/alioutas/Dropbox (HMS)/data_compression/dimos/compression/nir_et_al/raw_bp1/img_01_01*.tiff'

# path2 = '/Users/alioutas/Dropbox (HMS)/data_compression/data/synth_MT/sequence-as-stack-MT0.N1.HD-BP-250.tif'
# path1 = '/Users/alioutas/Dropbox (HMS)/data_compression/data/synth_MT/sequence-as-stack-MT0.N1.HD-BP+250.tif'


path2 = '/Users/alioutas/Dropbox (HMS)/raw_data_tiff/img_*_bp2.tiff'
path1 = '/Users/alioutas/Dropbox (HMS)/raw_data_tiff/img_*_bp1.tiff'



#%%
kernel_size = 9     
rel_thresh = 0.45

#%%
#z = SPARZIP(path1, stem='guy',output_path='/Users/alioutas/Desktop/sparz_nir/', path_image_files2=path2,rel_threshold = rel_thresh, kernel_size=kernel_size,reflect_bp2=True,align_planes=True)

#z = SPARZIP(path1, stem='guy_smaller',output_path='/Users/alioutas/Desktop/sparz_nir/', path_image_files2=path2,rel_threshold = rel_thresh, kernel_size=kernel_size,reflect_bp2=False,align_planes=False)
z = SPARZIP(path1, stem='guy_smaller',output_path='/Users/alioutas/Desktop/sparz_nir/', path_image_files2=path2,reflect_bp2=False,align_planes=False)

#%%
z.deflate_encode()
# %%
