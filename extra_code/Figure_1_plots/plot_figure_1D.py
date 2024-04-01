#%%
import os
import re
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import pandas as pd

#%%
# color pallete
color_pallete = {'av1' : '#8ECAE6',
                'ffv1' : '#219EBC',
                'h264' : '#023047',
                'prores' : '#817425',
                'x265' : '#FFB703',
                'zstd' : '#FB8500'
                }


#%%
# Toy processing time values
fake_processing_time = pd.DataFrame({'codec': ['av1', 'ffv1', 'h264', 'prores', 'x265', 'zstd'],
                                    'processing_time': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                                    'color_pallete' : color_pallete.values()})

#%%
# dot plot the processing time
fig, ax = plt.subplots()
codec = 'av1'  # Define the value of codec
ax.scatter(data=fake_processing_time, x='codec', y='processing_time', color='color_pallete', marker='o', s =250)  # Remove duplicate 'color' keyword argument

ax.set_xlabel('Processing time (s)')
ax.set_yticks([])
plt.show()

# %%
