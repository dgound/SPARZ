# SPARZ 


## Installation
Download the repo and navigate to inside the downloaded folder (there should be a file named setup.py in there) and type:
<code> pip install . </code>

## Dependencies 


## Usage
Minimum working example

<code>
from SPARZ import SPARZIP, SPARUNZIP

z=SPARZIP(path_image_files1="/PATH/TO/PLANE1_DATA",
                 stem=“NAME”,
                 output_path=“OUTPUT_PATH",
                 path_image_files2:str = "/PATH/TO/PLANE2_DATA”)

z.run()
</code>
To unzip:

<code>
u = SPARUNZIP(path_sparse_bp1=‘PATH_TO_BP1_NPZ_FILE', 
                 path_encoded_bp1=‘PATH_TO_BP1_MP4_ FILE’,
                 stem=’NAME', 
                 output_path=‘OUTPUT_PATH', 
                 path_sparse_bp2:str=‘PATH_TO_BP2_NPZ_FILE',, 
                 path_encoded_bp2:str=‘PATH_TO_BP2_MP4_ FILE’)

u.run()
</code>

### Test dataset
