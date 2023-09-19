#%%
import imageio as imageio
from scipy.stats import wasserstein_distance as wd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm
import pandas as pd
import numpy as np

#%%
def element_wise_diff(arr1, arr2):
    # Calculate the element-wise difference
    diff = np.subtract(arr1, arr2)

    # Find the positions where array2 is greater than array1
    greater_indices = np.where(arr2 > arr1)

    # Subtract the larger value from the smaller one at those positions
    result = np.copy(diff)
    result[greater_indices] = -result[greater_indices]
    return result

#%%
# find all available raw images in the path with glob
path_raw = '/Volumes/T7/compression_data/data_compression_localizations/input_data/nir_etal_raw_16bit/raw_16bit'
files_raw = natsorted(glob.glob(path_raw + '**/*.tiff'))


#%%
# find all folders but not files in the path with glob
path = '/Volumes/T7/compression_data/data_compression_localizations/input_data/nir_etal_ROI/'
folders = natsorted(glob.glob(path + '/*/'))
codec = [folder.split('/')[-2] for folder in folders]
codec_name = [folder.split('/')[-2] for folder in folders]

#%%
# create lists of all the files in each folder
for i in range(len(folders)):
    # files_compressed.append()
    codec[i] = natsorted(glob.glob(folders[i] + '*.tiff'))

#%%
# Lists to store intermediate results
wd_list = []
ssim_list = []
mse_list = []
mean_abs_list = []

for i in tqdm(range(len(files_raw))):
    raw = imageio.mimread(files_raw[i])

    #print('processing raw file:', i)

    for j in range(len(codec)):
        comp = imageio.mimread(codec[j][i])
        basename = os.path.basename(codec[j][i])

        for k in range(len(raw)):
            
            a = raw[k]
            c = comp[k]

            wd_value = wd(a.flatten(), c.flatten())
            wd_list.append({'value': wd_value, 'codec': codec_name[j], 'file': basename})

            ssim_value, _ = compare_ssim(a, c, full=True)
            ssim_list.append({'value': ssim_value, 'codec': codec_name[j], 'file': basename})

            diff = element_wise_diff(a, c)
            mse_value = np.mean(diff ** 2)
            mse_list.append({'value': mse_value, 'codec': codec_name[j], 'file': basename})

            mean_abs = np.mean(diff)
            mean_abs_list.append({'value': mean_abs, 'codec': codec_name[j], 'file': basename})

# Convert lists to DataFrames
raw_wd = pd.DataFrame(wd_list)
raw_ssim = pd.DataFrame(ssim_list)
raw_mse = pd.DataFrame(mse_list)
raw_mean_abs = pd.DataFrame(mean_abs_list)

#%%
raw_wd.to_csv('/Users/alioutas/Dropbox (HMS)/data_compression/analysis/20230917_image_analysis/20230917_raw_wd.csv')
raw_ssim.to_csv('/Users/alioutas/Dropbox (HMS)/data_compression/analysis/20230917_image_analysis/20230917_raw_ssim.csv')
raw_mse.to_csv('/Users/alioutas/Dropbox (HMS)/data_compression/analysis/20230917_image_analysis/20230917_raw_mse.csv')
raw_mean_abs.to_csv('/Users/alioutas/Dropbox (HMS)/data_compression/analysis/20230917_image_analysis/20230917_raw_mean_abs.csv')


#%%
# read the csv files
raw_wd = pd.read_csv('/Users/alioutas/Dropbox (HMS)/data_compression/analysis/20230917_image_analysis/20230917_raw_wd.csv')
raw_ssim = pd.read_csv('/Users/alioutas/Dropbox (HMS)/data_compression/analysis/20230917_image_analysis/20230917_raw_ssim.csv')
raw_mse = pd.read_csv('/Users/alioutas/Dropbox (HMS)/data_compression/analysis/20230917_image_analysis/20230917_raw_mse.csv')
raw_mean_abs = pd.read_csv('/Users/alioutas/Dropbox (HMS)/data_compression/analysis/20230917_image_analysis/20230917_raw_mean_abs.csv')

#%%
# kde plot of the wasserstein distances
sns.kdeplot(data=raw_wd.rename(columns={'value': 'wd'}).melt(id_vars='codec', value_vars='wd'), x='value', hue='codec')
plt.yscale('log')
plt.xscale('log')
plt.title('Wasserstein distance')
# plt.hist(raw_wd, bins=1000, color='codec')
plt.show()

#%%
# kde plot of the ssim distances
# sns.kdeplot(raw_ssim)
# plt.hist(raw_ssim, bins=1000)
sns.kdeplot(data=raw_ssim.rename(columns={'value': 'wd'}).melt(id_vars='codec', value_vars='wd'), x='value', hue='codec')
plt.yscale('log')
plt.xscale('log')
plt.title('Structural similarity')
#log
plt.show()

#%%
# kde plot of the mse distances
sns.kdeplot(data=raw_mse.rename(columns={'value': 'wd'}).melt(id_vars='codec', value_vars='wd'), x='value', hue='codec')
plt.xscale('log')
plt.yscale('log')
plt.title('Mean squared error')
# plt.hist(raw_mse, bins=1000)
plt.show()
# %%
# kde plot of the mse distances
# sns.kdeplot(raw_mean_abs)
sns.kdeplot(data=raw_mean_abs.rename(columns={'value': 'wd'}).melt(id_vars='codec', value_vars='wd'), x='value', hue='codec')
plt.xscale('log')
plt.yscale('log')
plt.title('Mean error')
# place the legend outside the figure/plot on the right
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()








####################################### LEFTOVERS #########################################

#%%

# raw_wd = pd.DataFrame()
# raw_ssim = pd.DataFrame()
# raw_mse =  pd.DataFrame()
# raw_mean_abs = pd.DataFrame()

# for i in tqdm(range(len(files_raw))):
#     raw = imageio.mimread(files_raw[i])

#     print('processing raw file:', i)

#     for j in range(len(codec)):
#         comp = imageio.mimread(codec[j][i])

#         for k in range(len(raw)):
            
#             a = raw[k]
#             # a = (((a - a.min())/ (a.max() - a.min()))*65535).astype('uint16')
#             c = comp[k]
#             # dif = cv2.subtract(a,c)
#             raw_wd = pd.concat([raw_wd, pd.DataFrame({'value': wd(a.flatten(),c.flatten()), 'codec': codec_name[j]}, index=[0])])

#             # raw_wd.append(wd(a.flatten(),c.flatten()))
#             # raw_wd['codec'] = codec_name[j]

#             ssim_value, _ = compare_ssim(a, c, full=True)
#             raw_ssim = pd.concat([raw_ssim, pd.DataFrame({'value': ssim_value, 'codec': codec_name[j]}, index=[0])])
#             # raw_ssim.append(ssim_value)
#             # raw_ssim['codec'] = codec_name[j]

#             mse_value = np.mean(element_wise_diff(a,c) ** 2)
#             raw_mse = pd.concat([raw_mse, pd.DataFrame({'value': mse_value, 'codec': codec_name[j]}, index=[0])])
#             # raw_mse.append(mse_value)
#             # raw_mse['codec'] = codec_name[j]

#             mean_abs = np.mean(element_wise_diff(a,c))
#             raw_mean_abs = pd.concat([raw_mean_abs, pd.DataFrame({'value': mean_abs, 'codec': codec_name[j]}, index=[0])])
#             # raw_mean_abs.append(mean_abs)
#             # raw_mean_abs['codec'] = codec_name[j]
