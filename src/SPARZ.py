#%%
import numpy as np
import tifffile
from tifffile import TiffFile
from skimage.feature import peak_local_max
import glob
import time
import sparse
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
import dask.array as da
import dask_image.imread
from dask import delayed
from reader import imread as vimread
import dask
from dask import compute
import gc
from tqdm import tqdm
import os
import pandas as pd
import json
from dask import delayed
import ffmpeg
from dask.diagnostics import ProgressBar
# from concurrent.futures import ThreadPoolExecutor, as_completed
import zstandard as zstd
from statsmodels.stats.power import TTestIndPower
import av
import concurrent.futures
# import joblib

#%%
class SPARZIP:
    def __init__(self, path_image_files1:str,  
                 stem:str, 
                 output_path:str,
                 path_image_files2:str = None,
                 relative_threshold:float = 0.45, 
                 epsilon:int = 12, 
                 kernel_size:int = 9, 
                #  compression_level:int=0,
                 batch_size:int=100, 
                 stack_size:int=250,
                 reflect_bp2:bool = False, 
                 find_peaks:bool = True,
                 align_planes:bool=False,
                 num_workers:int=4,
                 num_dask_workers:int=2
                 ):
      
        """
        Class for compressing the data into a sparse matrix and into a mp4 file.

        Parameters
        ----------
        path_image_files1 : str
            path to the first image file
        path_image_files2 : str
            path to the second image file
        stem : str
            stem of the output file
        relative_threshold : float, optional
            relative threshold for peak finding, by default 0.5
        epsilon : int, optional
            epsilon for peak finding, by default 12
        kernel_size : int, optional
            kernel size for peak finding, by default 3
        comp_level : int, optional
            compression level for the output file, by default 0
        reflect_bp2 : bool, optional
            whether to reflect the second image, by default False
        align_planes : bool, optional
            whether to align the planes, by default False

        """
        # self.codec = codec
        self.single_plane = False
        if path_image_files2 is None:
            self.single_plane = True
        self.path_image_files1 = sorted(glob.glob(path_image_files1))
        if self.single_plane==False:
            self.path_image_files2 = sorted(glob.glob(path_image_files2))
        self.bp1, self.bp2 = self.load_images(path_image_files1, path_image_files2)
        # if self.bp1.dtype == 'float32':
        try:
            if any(arr.dtype == np.float32 for arr in self.bp1):
                self.bp1, self.bp2 = self.to_16bit()
        except TypeError:
            dtype_first_array = self.bp1[0].compute().dtype
            if dtype_first_array == np.float32:
                self.bp1, self.bp2 = self.to_16bit()
        self.stem = stem
        if output_path[-1] != '/':
            self.output_path = output_path+"/"
        else:
            self.output_path = output_path
        self.rel_threshold = relative_threshold
        self.epsilon = epsilon
        self.kernel_size = kernel_size
        # self.compression_level = compression_level
        self.batch_size = batch_size
        self.stack_size = stack_size
        self.find_roi = find_peaks
        self.num_workers = num_workers
        self.num_dask_workers = num_dask_workers
        if reflect_bp2:
            if self.single_plane:
                print('Skipping reflection on single plane data.')
            else:
                self.bp2 = [np.flip(x, 2) for x in self.bp2]
        if align_planes:
            if self.single_plane:
                print('Cannot align single plane data. Skipping.')
            else:
                self.bp2 = self.align_planes(self.bp1, self.bp2)

        self.processed_bp1,self.processed_bp2 = self.process_images()

    @delayed
    def load_dat_file(self,dat_file,dimX, dimY):
        raw_data = np.memmap(dat_file, dtype='uint16')
        img_nums = len(raw_data) // (dimX * dimY)
        return np.reshape(raw_data, (img_nums, dimY, dimX))

    def load_images(self, path_image_files1:str, path_image_files2:str):
        print ('Lazily loading images...')
        # files1 = sorted(glob.glob(path_image_files1))
        files1=self.path_image_files1
        if not files1:
            raise ValueError(f"No files found at {path_image_files1}")
        multipage = False
        ext = os.path.splitext(files1[0])[1]
        if not ext:
            raise ValueError(f"No extension found for file {files1[0]}")
        if ext == '.tiff' or ext == '.tif':
            with TiffFile(files1[0]) as tif:
                multipage = len(tif.pages) > 1
            if multipage:
                p1 = [dask_image.imread.imread(f) for f in files1]
            else:
                p1 = []
                for i in range(0, len(files1), self.stack_size):
                    images = [dask_image.imread.imread(f) for f in files1[i:i+self.stack_size]]
                    p1.append(da.concatenate(images,axis=0))
        elif ext == '.dat':
            dir1=os.path.dirname(files1[0])
            json_path = os.path.join(dir1, 'data.json')
            if not os.path.exists(json_path):
                raise ValueError(f"data.json not found in directory: {dir1}")
            with open(json_path, 'r') as f:
                data_tmp = pd.DataFrame.from_dict(json.load(f))
            dimX = data_tmp.loc['Image'].value['RecordDimX']
            dimY = data_tmp.loc['Image'].value['RecordDimY']
            p1 = []
            for dat_file in files1:
                raw_data = np.memmap(dat_file, dtype='uint16')
                img_nums = len(raw_data) // (dimX * dimY)
                p1.append(da.from_delayed(self.load_dat_file(dat_file, dimX, dimY), shape=(img_nums, dimY, dimX), dtype='uint16').rechunk((1, dimY, dimX)))
        else:
            raise ValueError(f'Unsupported file extension: {ext}.Supported extensions are .tiff, .tif and SRX .dat')
        if self.single_plane:
            print ('Working with single plane data.')
            return p1, None
        print('Working with biplane data.')
        # files2 = sorted(glob.glob(path_image_files2))
        files2 = self.path_image_files2
        if not files2:
            raise ValueError(f"No files found at {path_image_files2}")
        ext = os.path.splitext(files2[0])[1]
        if not ext:
            raise ValueError(f"No extension found for file {files2[0]}")
        if ext == '.tiff' or ext == '.tif':
            if multipage:
                p2 = [dask_image.imread.imread(f) for f in files2]
            else:
                p2 = []
                for i in range(0, len(files2), self.stack_size):
                    images = [dask_image.imread.imread(f) for f in files2[i:i+self.stack_size]]
                    p2.append(da.concatenate(images,axis=0))
        elif ext == '.dat':
            dir1=os.path.dirname(files1[0])
            json_path = os.path.join(dir1, 'data.json')
            if not os.path.exists(json_path):
                raise ValueError(f"data.json not found in directory: {dir1}")
            p2 = []
            for dat_file in files2:
                raw_data = np.memmap(dat_file, dtype='uint16')
                img_nums = len(raw_data) // (dimX * dimY)
                p2.append(da.from_array(np.reshape(raw_data, (img_nums, dimY, dimX)), chunks=(1, dimY, dimX)) )
        else:
            raise ValueError(f'Unsupported file extension: {ext}')
        return p1, p2

    def to_16bit(self):
        print('Float32 data detected.')
        print('Converting to 16 bit...')
        # z1 = self.bp1.map_blocks(lambda x:((x - x.min())/ (x.max() - x.min())*65535).astype('uint16'), dtype='uint16')
        z1 = [blck.map_blocks(lambda x:((x - x.min())/ (x.max() - x.min())*65535).astype('uint16'), dtype='uint16') for blck in self.bp1]
        if self.single_plane:
            return z1, None
        # z2 = self.bp2.map_blocks(lambda x:(( x - x.min())/ (x.max() - x.min()) * 65535).astype('uint16'), dtype='uint16')
        z2 = [blck.map_blocks(lambda x:((x - x.min())/ (x.max() - x.min())*65535).astype('uint16'), dtype='uint16') for blck in self.bp2]
        return z1, z2 
    # @njit
    def find_peaks(self, image:np.ndarray, kernel_size:int, min_distance:int = 1):

        #find the local maxima in the image
        coordinates = peak_local_max(image, 
                                    threshold_rel= self.rel_threshold, 
                                    min_distance=min_distance, 
                                    footprint=np.ones((kernel_size, kernel_size))
                                    )

        return coordinates
    # @njit
    def intersection(self, peaks1:np.ndarray, peaks2:np.ndarray, bp1:np.ndarray, epsilon:int):
        # Compute element-wise differences between peaks
        diffs = np.abs(peaks1[:, None, :] - peaks2)
        
        # Find indices of peaks in peaks1 that are within epsilon of any peak in peaks2
        common_inds_1 = np.where(np.any(np.all(np.abs(diffs) <= epsilon, axis=2), axis=1))[0]
        
        # Find indices of peaks in peaks2 that are within epsilon of any peak in peaks1
        # common_inds_2 = np.where(np.any(np.all(np.abs(diffs) <= epsilon, axis=2), axis=0))[0]
        
        # Create binary matrix where common peaks are set to 1 and all other elements are 0
        common_peaks = np.zeros(bp1.shape)
        common_peaks[peaks1[common_inds_1, 0], peaks1[common_inds_1, 1]] = 1
        
        return common_peaks
    # @njit
    def union(self, peaks1:np.ndarray, peaks2:np.ndarray, bp1):
        merged = np.vstack([peaks1, peaks2])

        merged_peaks = np.zeros(bp1)
        merged_peaks[merged[:, 0], merged[:, 1]] = 1    
        return merged_peaks
    # @njit
    def add_kernel(self, image:np.ndarray,kernel_size:int):

        kernel = np.ones((kernel_size, kernel_size))

        pad_size = kernel.shape[0]//2
        arr = np.pad(image, pad_size, mode='constant')

        # Find the indices of the ones in the input array
        ones_indices = np.argwhere(arr == 1)

        # Initialize a new array with the same shape as the input array
        new_arr = np.zeros(arr.shape)

        # Loop through the indices of the ones and center the kernel on each one
        for i, j in ones_indices:
            start_i = i - kernel.shape[0]//2
            end_i = i + kernel.shape[0]//2 + 1
            start_j = j - kernel.shape[1]//2
            end_j = j + kernel.shape[1]//2 + 1

            # Center the kernel on the current one
            new_arr[start_i:end_i, start_j:end_j] = kernel

        # Remove the padding from the new array
        return new_arr[pad_size:-pad_size, pad_size:-pad_size]
    

    def align_planes(self,plane1:np.ndarray, plane2:np.ndarray):
        print('Aligning planes, please wait...')
        # frames = plane1.shape[0]
        # # print (frames)
        # if frames < 50:
        #     shifts, _, _ = phase_cross_correlation(plane1, plane2,
        #                                                 upsample_factor=100)

        
        # else:
            # sample = np.random.randint(int(frames*0.48), int(frames*0.52),150)
        plane1_c = da.concatenate(plane1, axis=0)
        plane2_c = da.concatenate(plane2, axis=0)
        b1 = plane1_c[int(plane1_c.shape[0]*0.48): int(plane1_c.shape[0]*0.52)]
        b2 = plane2_c[int(plane2_c.shape[0]*0.48): int(plane2_c.shape[0]*0.52)]
        shifts, _, _ = phase_cross_correlation(b1, b2,
                                                    upsample_factor=100)
        #We don't want to shift frame numbers, so we set the first value to 0
        shifts[0] = 0
        print(f'Applying mean shift {shifts} to all frames.')
        del plane1_c, plane2_c
        gc.collect()
            
        return [p.map_blocks(lambda x: shift(x, shifts, mode='constant')) for p in plane2]


    def get_raw_frame(self,index:int=None,start_frame:int=None,end_frame:int=None,plane:str='bp1'):
        if index is None:
            print ('No index provided. Using index 0.')
            index = 0
        if start_frame is None:
            # raise ValueError('Please provide a start frame.')
            print ('No starting frame provided. Using frame 0.')
            start_frame = 0
        if end_frame is None:
            print('No end frame provided. Using starting_frame+1.')
            end_frame = start_frame+1
        if plane == 'bp1':
            return self.bp1[index].blocks[start_frame:end_frame,0].compute()
        else:
            return self.bp2[index].blocks[start_frame:end_frame,0].compute()

    def get_processed_frame(self,index:int=None,start_frame:int=None,end_frame:int=None,plane:str='bp1'):
        if index is None:
            print ('No index provided. Using index 0.')
            index = 0
        if start_frame is None:
            # raise ValueError('Please provide a start frame.')
            print ('No starting frame provided. Using frame 0.')
            start_frame = 0
        if end_frame is None:
            print('No end frame provided. Using starting_frame+1.')
            end_frame = start_frame+1
        if plane == 'bp1':
            return self.processed_bp1[index].blocks[start_frame:end_frame,0].compute().todense()
        else:
            return self.processed_bp2[index].blocks[start_frame:end_frame,0].compute().todense()
        
    def process_images(self):
        print('Processing images...')
        # map1 = self.bp1.map_blocks(lambda x: self.find_peaks(x[0,:,:],self.kernel_size,min_distance=1), dtype='int16')
        map1 = [blck.map_blocks(lambda x: self.find_peaks(x[0,:,:],self.kernel_size,min_distance=1), dtype='int16') for blck in self.bp1]
        if self.single_plane == False:
            assert len(self.bp1) == len(self.bp2), 'Error: Both biplanes must have the same number of images.'
            # map2 = self.bp2.map_blocks(lambda x: self.find_peaks(x[0,:,:],self.kernel_size,min_distance=1), dtype='int16')
            map2 = [blck.map_blocks(lambda x: self.find_peaks(x[0,:,:],self.kernel_size,min_distance=1), dtype='int16') for blck in self.bp2]
            # map_union = da.map_blocks(self.union, map1, map2, self.bp1[0,:,:].shape, dtype='int16')
            map_union = [da.map_blocks(self.union, map1[i], map2[i], self.bp1[0][0,:,:].shape, dtype='int16') for i in range(len(map1))]
            # map_kernel = map_union.map_blocks(self.add_kernel, self.kernel_size, dtype='int16')
            map_kernel = [blck.map_blocks(self.add_kernel, self.kernel_size, dtype='int16') for blck in map_union]
            # sp1 = da.where(map_kernel, self.bp1, 0)
            try:
                sp1 = [da.where(map_kernel[i], self.bp1[i], 0) for i in range(len(map_kernel))]
                # sp2 = da.where(map_kernel, self.bp2, 0)
                sp2 = [da.where(map_kernel[i], self.bp2[i], 0) for i in range(len(map_kernel))]
            except TypeError:
                def apply_where(kernel, img):
                    return da.where(kernel, img, 0)
                sp1 = [da.map_blocks(apply_where, map_kernel[i], self.bp1[i], dtype='int16') for i in range(len(map_kernel))]
                sp2 = [da.map_blocks(apply_where, map_kernel[i], self.bp2[i], dtype='int16') for i in range(len(map_kernel))]
            
            print('Done.')
            # return sp1.map_blocks(sparse.COO, dtype='int16'), sp2.map_blocks(sparse.COO, dtype='int16')
            return [sp1[i].map_blocks(sparse.COO, dtype='int16') for i in range(len(sp1))], [sp2[i].map_blocks(sparse.COO, dtype='int16') for i in range(len(sp2))]
        
        def add_mask(peaks:np.ndarray):
            tmp = np.zeros(self.bp1[0][0,:,:].shape)
            tmp[peaks[:, 0], peaks[:, 1]] = 1 
            return tmp
        map_mask = [b.map_blocks(lambda x: add_mask(x), dtype='int16') for b in map1]
        map_kernel = [k.map_blocks(self.add_kernel, self.kernel_size, dtype='int16') for k in map_mask]
        try:
            sp1 = [da.where(map_kernel[i], self.bp1[i], 0) for i in range(len(map_kernel))]
        except TypeError:
            def apply_where(kernel, img):
                return da.where(kernel, img, 0)
            sp1 = [da.map_blocks(apply_where, map_kernel[i], self.bp1[i], dtype='int16') for i in range(len(map_kernel))]
        print('Done.')
        return [sp1[i].map_blocks(sparse.COO, dtype='int16') for i in range(len(sp1))], None
    
    # def compress_joblib(self,path,mat):
        # joblib.dump(mat, path, compress=('lzma', 9))


    def deflate(self):
        print('Deflating images...')
        show_progress_bar = False
        try:
            get_ipython()
            show_progress_bar = True
        except NameError:
            show_progress_bar = False
        with dask.config.set(scheduler='threads'):
            # Prepare a list to store delayed operations
            saves = []
            for i in range(len(self.processed_bp1)):
                # Directly append delayed save_npz operations to the list
                flnm1 = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files1[i]))[1])[0]
                saves.append(delayed(sparse.save_npz)(self.output_path+self.stem+'_peaks_bp1_part_'+str(i)+'.npz',self.processed_bp1[i]))
                # saves.append(delayed(self.compress_joblib)(self.output_path+flnm1+'.sparz',self.processed_bp1[i]))
                if self.single_plane == False:
                    flnm2 = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files2[i]))[1])[0]
                    # saves.append(delayed(self.compress_joblib)(self.output_path+flnm2+'.sparz',self.processed_bp2[i]))
                    saves.append(delayed(sparse.save_npz)(self.output_path+self.stem+'_peaks_bp2_part_'+str(i)+'.npz',self.processed_bp2[i]))
            # Perform the save_npz operations
            if show_progress_bar:
                progress_bar = tqdm(total=len(saves), desc="Creating sparse matrices", position=0, leave=True)
            
            for i in range(0, len(saves), self.batch_size):
                batch = saves[i:i+self.batch_size]
                dask.compute(*batch, scheduler='threads',num_workers=self.num_workers)
                if show_progress_bar:
                    progress_bar.update(self.batch_size)
            if show_progress_bar:
                progress_bar.close()
        end = time.time()
        



    def determine_ctu_size(self, image_width, image_height):
        # Example logic for determining CTU size
        min_dimension = min(image_width, image_height)
        if min_dimension <= 64:
            return 16  # Smallest CTU size for very small images
        elif min_dimension <= 128:
            return 32
        else:
            return 64  # Default CTU size for larger images
        

    def encode(self, codec:str='x265', compression_lvl:int=0, custom_dict:dict=None, custom_file_extension:str=None):
        if codec =='x265':
            w, h = self.bp1[0].shape[1], self.bp1[0].shape[2]
            ctu_size = self.determine_ctu_size(w, h)
            compression_levels = {0: {
                                                'vcodec': 'libx265',
                                                'pix_fmt': 'gray12le',
                                                'x265-params': f'lossless=1:ctu={ctu_size}',
                                            },
                                1:{
                                                'vcodec': 'libx265',
                                                'crf': '0', 
                                                'pix_fmt': 'gray12le',
                                                'x265-params': f'ctu={ctu_size}',
                                            },
                                2:{
                                                'vcodec': 'libx265',
                                                'crf': '5',
                                                'pix_fmt': 'gray12le',
                                                'x265-params': f'ctu={ctu_size}',
                                            },

                                3:{
                                                'vcodec': 'libx265',
                                                'crf': '15',
                                                'pix_fmt': 'gray12le',
                                                'x265-params': f'ctu={ctu_size}',
                                            }                                      
                                            
                                            
                            }
        elif codec == 'av1':
            compression_levels = {0: {
                                                'vcodec': 'libaom-av1',
                                                'pix_fmt': 'gray16le', 
                                                'strict': 'experimental',
                                                'crf': '0',
                                                'fps_mode': 'vfr'
                                }
,
                                1:{
                                                'vcodec': 'libaom-av1',
                                                'crf': '5',
                                                'pix_fmt': 'gray16le',
                                                'fps_mode': 'vfr'
                                            },
                                2:{
                                                'vcodec': 'libaom-av1',
                                                'crf': '15',
                                                'pix_fmt': 'gray16le',
                                                'fps_mode': 'vfr'
                                            },

                                3:{
                                                'vcodec': 'libaom-av1',
                                                'crf': '25',
                                                'pix_fmt': 'gray16le',
                                                'fps_mode': 'vfr'
                                            }                                      
                                            
                                            
                            }
        elif codec == 'x264':
            compression_levels = {0: {
                                                'vcodec': 'libx264',
                                                'pix_fmt': 'gray16le',
                                                'crf': '0'
                                },
                                1:{
                                                'vcodec': 'libx264',
                                                'crf': '5',
                                                'pix_fmt': 'gray16le'
                                },
                                2:{
                                                'vcodec': 'libx264',
                                                'crf': '15',
                                                'pix_fmt': 'gray16le'
                                },
                                3:{
                                                'vcodec': 'libx264',
                                                'crf': '25',
                                                'pix_fmt': 'gray16le'
                                }
                }

        elif codec =='ffv1':
            compression_levels = {0: {
                                                'vcodec': 'ffv1',
                                                'pix_fmt': 'gray16le',
                                },
                                1:{
                                                'vcodec': 'ffv1',
                                                'pix_fmt': 'gray16le',
                                                'level': '3'
                                },
                                2:{
                                                'vcodec': 'ffv1',
                                                'pix_fmt': 'gray16le',
                                                'level': '1'
                                },
                                3:{
                                                'vcodec': 'ffv1',
                                                'pix_fmt': 'gray16le',
                                                'level': '0'
                                }
    
                }
        elif codec == 'prores':
            compression_levels = {
            0: {
                'vcodec': 'prores_ks',
                'pix_fmt': 'yuv444p10le',
                'profile:v': '4444',
            },
            1: {
                'vcodec': 'prores_ks',
                'pix_fmt': 'yuv422p10le',
                'profile:v': 'hq',
            },
            2: {
                'vcodec': 'prores_ks',
                'pix_fmt': 'yuv422p10le',
                'profile:v': 'standard',
            },
            3: {
                'vcodec': 'prores_ks',
                'pix_fmt': 'yuv422p10le',
                'profile:v': 'lt',
            }
        }
        
        elif codec =='user':
            assert custom_dict is not None, 'Error: Custom dictionary not provided.'
            assert custom_file_extension is not None, 'Error: Custom file extension not provided.'
            assert isinstance(custom_dict, dict), 'Error: Custom dictionary must be a dictionary.'
            assert 'vcodec' in custom_dict.keys(), 'Error: Custom dictionary must contain a vcodec key.'
            print ('Applying user defined settings.')

        else:
            raise ValueError(f'Unsupported codec: {codec}. Supported codecs are x265 and av1.')
        print('Compressing video...')
        show_progress_bar = False
        try:
            get_ipython()
            show_progress_bar = True
        except NameError:
            show_progress_bar = False
        # Process videos using delayed
        writes = []
        for k in range(len(self.processed_bp1)):
            input_file_name1 = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files1[k]))[1])[0]

            if codec =='prores':
                video_name1 = f'{self.output_path}{input_file_name1}_compression_level_{compression_lvl}.mov'
            elif codec =='ffv1':
                video_name1 = f'{self.output_path}{input_file_name1}_compression_level_{compression_lvl}.avi'
            elif codec=='user':
                video_name1 = f'{self.output_path}{input_file_name1}_compression_level_{compression_lvl}.{custom_file_extension}'
            else:
                video_name1 = f'{self.output_path}{input_file_name1}_compression_level_{compression_lvl}.mp4'
                

            if codec!='user':
                writer_args = compression_levels[compression_lvl]
            else:
                writer_args = custom_dict

            if self.single_plane:
                writes.append(delayed(self.write_frames_to_video)(
                    self.bp1[k], video_name1, writer_args
                ))
            else:
                writes.append(delayed(self.write_frames_to_video)(
                    self.bp1[k], video_name1, writer_args
                ))
                input_file_name2 = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files2[k]))[1])[0]
                
                if codec=='prores':
                    video_name2 = f'{self.output_path}{input_file_name2}_compression_level_{compression_lvl}.mov'
                elif codec=='ffv1':
                    video_name2 = f'{self.output_path}{input_file_name2}_compression_level_{compression_lvl}.avi'
                elif codec=='user':
                    video_name2 = f'{self.output_path}{input_file_name2}_compression_level_{compression_lvl}.{custom_file_extension}'
                else:
                    video_name2 = f'{self.output_path}{input_file_name2}_compression_level_{compression_lvl}.mp4'
                # video_name2 = f'{self.output_path}{self.stem}_bp2_compression_level_{compression_lvl}_part_{k}.mp4'
                writes.append(delayed(self.write_frames_to_video)(
                    self.bp2[k], video_name2, writer_args
                ))

        # Execute the delayed writes
        if show_progress_bar:
            with ProgressBar():
                compute(*writes, scheduler='threads',num_workers=self.num_workers)
        else:
            compute(*writes, scheduler='threads',num_workers=self.num_workers)

    def write_frames_to_video(self, bp1_block, video_name, writer_args):
        # Convert Dask array slices to NumPy arrays
        bp1_frames = bp1_block
        # bp2_frames = bp2_block if bp2_block is not None else None

        # Prepare output directories
        os.makedirs(os.path.dirname(video_name), exist_ok=True)

        input_dict = {
            'format': 'rawvideo',
            'pix_fmt': 'gray16le',
            's': f'{bp1_frames.shape[2]}x{bp1_frames.shape[1]}',
            'y': None,
            # 's': '{}x{}'.format(*input_frames.shape[1:3][[::-1]])
        }

        ffmpeg_input = ffmpeg.input('pipe:', **input_dict)
        # Create a ffmpeg output with the writer arguments
        writer_args['s'] = input_dict['s']
        # with open(os.devnull, "w") as devnull:
        ffmpeg_output = ffmpeg.output(ffmpeg_input, video_name, loglevel="quiet", **writer_args)

        # Run the ffmpeg command
        # ffmpeg.run(ffmpeg_output, input=input_frames.tobytes())
        ffmpeg.run(ffmpeg_output, input=bp1_frames.tobytes(), capture_stdout=True, capture_stderr=False)

    def zstd_compress(self, compression_level, effect_size, power, compute_dict):
        # Perform power analysis to find sample size
        analysis = TTestIndPower()
        sample_size = int(analysis.solve_power(effect_size=effect_size, power=power, alpha=0.05))

        # Initialize the ZstdCompressor with the desired compression level
        cctx = zstd.ZstdCompressor(level=compression_level)

        # Sample and compress frames from bp1 and bp2
        for k in range(len(self.bp1)):
            input_file_name1 = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files1[k]))[1])[0]
            input_file_name2 = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files2[k]))[1])[0]
            # Assume self.bp1[k] and self.bp2[k] can be iterated chunk-wise
            if compute_dict:
            # Compute sample size based frames
                bp1_samples = np.random.choice(self.bp1[k].shape[0], sample_size, replace=False)
                bp1_samples = self.bp1[k][bp1_samples].flatten().compute()
                bp1_sample_bytes = [sample.tobytes() for sample in bp1_samples]
                bp2_samples = np.random.choice(self.bp2[k].shape[0], sample_size, replace=False)
                bp2_samples = self.bp2[k][bp2_samples].flatten().compute()
                bp2_sample_bytes = [sample.tobytes() for sample in bp2_samples]
            
                print(f'Training dictionaries with {sample_size} samples...')
                bp1_dict = zstd.ZstdCompressionDict(bp1_samples,dict_type=zstd.DICT_TYPE_RAWCONTENT)
                bp2_dict = zstd.ZstdCompressionDict(bp2_samples,dict_type=zstd.DICT_TYPE_RAWCONTENT)
                
                bp1_dict = zstd.train_dictionary(dict_size=131072,samples=bp1_sample_bytes)
                bp2_dict = zstd.train_dictionary(dict_size=131072,samples=bp2_sample_bytes)
                        

                print(f'Compressing data with compression level {compression_level}...')

            else:
                print(f'Compressing data with compression level {compression_level} without dictionary...')



            output_file1 = f'{self.output_path}{input_file_name1}_level_{compression_level}.zst'
            output_file2 = f'{self.output_path}{input_file_name2}_level_{compression_level}.zst'

            try:
                with open(output_file1, 'wb') as f1:
                    if compute_dict:
                        cctx = zstd.ZstdCompressor(dict_data=bp1_dict, level=compression_level)
                    else:
                        cctx = zstd.ZstdCompressor(level=compression_level)

                    zdict_present = 1 if compute_dict else 0
                    compressor1 = cctx.stream_writer(f1, write_size=32768)
                    
                    # Write dictionary presence flag
                    f1.write(np.array([zdict_present], dtype=np.uint8).tobytes())
                    
                    if compute_dict:
                        # Write dictionary length and content if present
                        f1.write(np.array([len(bp1_dict)], dtype=np.int32).tobytes())
                        f1.write(bp1_dict.as_bytes())
                    
                    # Serialize and compress array shape and dtype
                    shape_bytes = np.array(self.bp1[k].shape, dtype=np.int32).tobytes()
                    f1.write(shape_bytes)
                    
                    dtype_bytes = np.array(str(self.bp1[k].dtype), dtype='S20').tobytes()
                    f1.write(dtype_bytes)
                    
                    # Compress data chunks
                    for chunk in self.bp1[k]:
                        compressor1.write(chunk.compute().tobytes())
                        
                    compressor1.flush(zstd.FLUSH_FRAME)
            except Exception as e:
                print(f"Error during compression: {e}")

            try:
                with open(output_file2, 'wb') as f1:
                    if compute_dict:
                        # If dictionary computation is desired, initialize compressor with the dictionary
                        cctx = zstd.ZstdCompressor(dict_data=bp2_dict, level=compression_level)
                    else:
                        cctx = zstd.ZstdCompressor(level=compression_level)

                    zdict_present = 1 if compute_dict else 0
                    compressor1 = cctx.stream_writer(f1, write_size=32768)
                    
                    # Write dictionary presence flag
                    f1.write(np.array([zdict_present], dtype=np.uint8).tobytes())
                    
                    if compute_dict:
                        # Write dictionary length and content if present
                        f1.write(np.array([len(bp2_dict)], dtype=np.int32).tobytes())
                        f1.write(bp2_dict.as_bytes())
                    
                    # Serialize and compress array shape and dtype
                    shape_bytes = np.array(self.bp2[k].shape, dtype=np.int32).tobytes()
                    f1.write(shape_bytes)
                    
                    dtype_bytes = np.array(str(self.bp2[k].dtype), dtype='S20').tobytes()
                    f1.write(dtype_bytes)
                    
                    # Compress data chunks
                    for chunk in self.bp2[k]:
                        compressor1.write(chunk.compute().tobytes())
                        
                    compressor1.flush(zstd.FLUSH_FRAME)
            except Exception as e:
                print(f"Error during compression: {e}")


    def run(self,codec:str='x265', compression_level:int=0,custom_dict:dict=None,custom_file_extension:str=None,compute_zstd_dict:bool=False):#,find_peaks:bool=True):
        if codec!='zstd' and compute_zstd_dict:
            print ('Warning: Dictionary computation is only supported for Zstandard compression. Ignoring compute_zstd_dict flag.')
            compute_zstd_dict = False
        
        if (self.find_roi) and (codec =='zstd'):
            print ('Warning: ROI detection is not supported for Zstandard compression. Ignoring find_roi flag.')
            self.find_roi = False
            
        if (self.find_roi):
            self.deflate()
            gc.collect()
        if codec in ['x265', 'av1', 'x264', 'ffv1', 'prores', 'user']:
            assert compression_level in [0,1,2,3], 'Error: Compression level for ffmpeg based compression must be 0, 1, 2 or 3.'
            self.encode(codec=codec, compression_lvl=compression_level,custom_dict=custom_dict,custom_file_extension=custom_file_extension)
        elif codec=='zstd':
            assert compression_level <= 22, 'Error: Valid compression levels for Zstandard compression are all negative integers through 22.'
            self.zstd_compress(compression_level=compression_level, effect_size=0.5, power=0.95, compute_dict=compute_zstd_dict)
        else:
            raise ValueError(f'Unsupported codec: {codec}.')


class SPARUNZIP:
    def __init__(self, path_sparse_bp1:str, 
                 path_encoded_bp1:str, 
                 stem:str, 
                 output_path:str, 
                 path_sparse_bp2:str=None, 
                 path_encoded_bp2:str=None, 
                 use_roi:bool=True, 
                 chunk_size:int=10, 
                 num_workers:int=4, 
                 num_dask_workers:int=2):
        
        # self.path_encoded_bp1, self.path_encoded_bp2 = path_encoded_bp1, path_encoded_bp2
        self.path_encoded_bp1 = sorted(glob.glob(path_encoded_bp1))
        self.path_encoded_bp2 = sorted(glob.glob(path_encoded_bp2)) if path_encoded_bp2 is not None else None
        self.use_roi = use_roi

        # if path_encoded_bp2 is not None:
        #     self.path_encoded_bp2 = sorted(glob.glob(path_encoded_bp2))
        # else:
        #     self.path_encoded_bp2 = None
        # self.encoded_bp1_files, self.encoded_bp2_files = sorted(glob.glob(path_encoded_bp1)), None
        # self.use_zstd_dict = use_zstd_dict)
        if os.path.splitext(self.path_encoded_bp1[0])[1] == '.zst':
            if self.use_roi:
                print('ROI detection is not supported for Zstandard compressed files. Ignoring use_roi flag.')
                self.use_roi = False
            self.encoded_bp1, self.encoded_bp2 = self.decode_zst(path_encoded_bp1, path_encoded_bp2)
        else:
            self.encoded_bp1, self.encoded_bp2 = self.decode(path_encoded_bp1, path_encoded_bp2)
        self.shapes = [x.shape[:3] for x in self.encoded_bp1]
        if self.use_roi:
            self.sparse_bp1, self.sparse_bp2 = self.load_sparse(path_sparse_bp1, path_sparse_bp2,self.shapes)
            # print ('Loaded sparse matrices.')
            # print (self.sparse_bp1)
            # print (self.sparse_bp1[0].shape)
            self.processed_bp1, self.processed_bp2 = self.process_frames()
            # print ('Processed frames.')
            # print (self.processed_bp1)
            # print (self.processed_bp1[0].shape)
        self.stem = stem
        self.num_workers = num_workers
        self.num_dask_workers = num_dask_workers
        if output_path[-1] != '/':
            self.output_path = output_path+"/"
        else:
            self.output_path = output_path
        self.chunk_size = chunk_size

    # def load_sparse(self, path_sparse_bp1:str, path_sparse_bp2:str):
    #     print ('Loading sparse matrices...')
    #     files_bp1 = sorted(glob.glob(path_sparse_bp1))
    #     if path_sparse_bp2 is not None:
    #         files_bp2 = sorted(glob.glob(path_sparse_bp2))
    #         assert len(files_bp1) == len(files_bp2), 'Error: Both biplanes must have the same number of images.'
    #         bp1,bp2 = [],[]
    #         for i in range(len(files_bp1)):
    #             try:
    #                 bp1.append(joblib.load(files_bp1[i]))
    #                 bp2.append(joblib.load(files_bp2[i]))
    #             except Exception as e:
    #                 print(f'Error loading sparse matrices: {e}')
    #                 print(f'File: {files_bp1[i]}')
    #         return bp1, bp2
    #     bp1 =[]
    #     for i in range(len(files_bp1)):
    #         bp1.append(joblib.load(files_bp1[i]))
    #     return bp1, None

    def load_sparse(self,sparse_bp1:str, sparse_bp2:str, shapes:tuple):
        print ('Loading sparse matrices...')
        files_bp1 = sorted(glob.glob(sparse_bp1))
        if sparse_bp2 is not None:
            files_bp2 = sorted(glob.glob(sparse_bp2))
            assert len(files_bp1) == len(files_bp2), 'Error: Both biplanes must have the same number of images.'
            bp1,bp2 = [],[]
            for i in range(len(files_bp1)):
                bp1.append(da.from_array(sparse.load_npz(files_bp1[i]), chunks=(1,shapes[i][1],shapes[i][2])))
                bp2.append(da.from_array(sparse.load_npz(files_bp2[i]), chunks=(1,shapes[i][1],shapes[i][2])))
            return bp1, bp2
        bp1 =[]
        for i in range(len(files_bp1)):
            bp1.append(da.from_array(sparse.load_npz(files_bp1[i]), chunks=(1,shapes[i][1],shapes[i][2])))
        return bp1, None


    def load_mp4(self, file_path):
        # Open the video file
        container = av.open(file_path)
        video_stream = container.streams.video[0]
        frame_count = video_stream.frames

        # Read all frames into a list of numpy arrays
        @delayed
        def read_frames():
            frames = []
            dtype = None
            for frame in container.decode(video_stream):
                np_frame = frame.to_ndarray(format='gray16le')
                if dtype is None:
                    dtype = np_frame.dtype  # Set dtype on first frame
                frames.append(np_frame)
            return np.array(frames), dtype
    
        @delayed
        def read_frames_h264():
            frames = []
            for frame in container.decode(video_stream):
                # Assuming conversion directly to 'gray16le' is handled elsewhere or not necessary
                y_plane = frame.planes[0]
                y_data = np.frombuffer(y_plane, np.uint16)
                y_data = y_data.reshape((frame.height, frame.width))
                y_data_16bit = np.left_shift(y_data, 6)
                frames.append(y_data_16bit)
            return np.array(frames), y_data_16bit.dtype
            # return np.stack(frames, axis=0) 


        # Get delayed frames and dtype
        if video_stream.codec.name == 'h264':
            print('H264 codec detected.')
            frames_dtype = read_frames()
        else:
            print(f'{video_stream.codec.name} codec detected.')
            frames_dtype = read_frames()

        # Calculate the shape and dtype of the frames for creating a Dask array
        # Since the dtype and frames are in a single tuple, we need to compute them to extract properly
        frames, dtype = frames_dtype.compute()

        # Create a Dask array from the numpy array of frames
        dask_frames = da.from_array(frames, chunks=(1, *frames[0].shape))
        return dask_frames

    def decode(self, path_bp1, path_bp2=None):
        print('Decoding images...')
        files_bp1 = sorted(glob.glob(path_bp1))
        bp1 = [self.load_mp4(file) for file in files_bp1]
        # print ('bp1',bp1[0])

        bp2 = None
        if path_bp2:
            files_bp2 = sorted(glob.glob(path_bp2))
            assert len(files_bp1) == len(files_bp2), 'Error: Both biplanes must have the same number of images.'
            bp2 = [self.load_mp4(file) for file in files_bp2]

        return bp1, bp2


    # def load_mp4(self, file_path):
    #     dtype = None
    #     frame_count = 0
    #     first_frame = None

    #     # Read the first frame to infer dtype and get frame count
    #     with av.open(file_path) as container:
    #         frame_count = container.streams.video[0].frames
    #         for packet in container.demux():
    #             for frame in packet.decode():
    #                 first_frame = frame.to_ndarray(format='gray16le')
    #                 dtype = first_frame.dtype
    #                 break
    #             if dtype is not None:
    #                 break

    #     def frame_generator(file_path):
    #         with av.open(file_path) as container:
    #             video_stream = container.streams.video[0]
    #             for frame_index, frame in enumerate(container.decode(video_stream)):
    #                 yield frame_index, frame

    #     def read_frame(i):
    #         for index,frame in frame_generator(file_path):
    #             if index == i:
    #                 return frame.to_ndarray(format='gray16le')

    #     frames = [delayed(read_frame)(i) for i in range(frame_count)]

    #     # Create Dask arrays for each frame
    #     frame_arrays = [da.from_delayed(frame, shape=first_frame.shape, dtype=dtype) for frame in frames]

    #     # Concatenate frame arrays into a single Dask array
    #     video_array = da.stack(frame_arrays, axis=0)

    #     return video_array

    
    # def decode(self, path_bp1: str, path_bp2: str):
    #     print('Decoding images...')
    #     # files_bp1 = sorted(glob.glob(path_bp1))
    #     files_bp1 = self.path_encoded_bp1
    #     if self.path_encoded_bp2 is None:
    #         with concurrent.futures.ThreadPoolExecutor() as executor:
    #             bp1 = list(executor.map(lambda file: self.load_mp4(file), files_bp1))
    #         return bp1, None
    #     # files_bp2 = sorted(glob.glob(path_bp2))
    #     files_bp2 = self.path_encoded_bp2
    #     assert len(files_bp1) == len(files_bp2), 'Error: Both biplanes must have the same number of images.'
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         bp1 = list(executor.map(lambda file: self.load_mp4(file), files_bp1))
    #         bp2 = list(executor.map(lambda file: self.load_mp4(file), files_bp2))
    #     return bp1, bp2

    def decode_fallback(self, path_bp1:str, path_bp2:str):
        print('Decoding images...')
        files_bp1 = sorted(glob.glob(path_bp1))
        if path_bp2 is not None:
            files_bp2 = sorted(glob.glob(path_bp2))
            # self.encoded_bp2_files = files_bp2
            assert len(files_bp1) == len(files_bp2), 'Error: Both biplanes must have the same number of images.'
            bp1, bp2 = [], []
            for i in range(len(files_bp1)):
                bp1.append(vimread(files_bp1[i], dtypes='uint16'))
                bp2.append(vimread(files_bp2[i], dtypes='uint16'))
            return bp1, bp2
        bp1 = []
        for i in range(len(files_bp1)):
            bp1.append(vimread(files_bp1[i], dtypes='uint16'))
        return bp1, None
        # return vimread(path_bp1, dtypes='uint16'), vimread(path_bp2, dtypes='uint16')

    def decode_zst(self, path_bp1: str, path_bp2: str = None):
        print('Decoding images...')
        files_bp1 = sorted(glob.glob(path_bp1))

        bp1 = []
        for file_path in files_bp1:
            with open(file_path, 'rb') as f:
                zdict_present = np.frombuffer(f.read(1), dtype=np.uint8)[0]
                if zdict_present:
                    # Read the length of the zdict
                    zdict_length = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    # Read the zdict itself
                    zdict = f.read(zdict_length)
                    zstd_dict_bp1 = zstd.ZstdCompressionDict(zdict)
                    dctx_bp1 = zstd.ZstdDecompressor(dict_data=zstd_dict_bp1)
                    # Adjust the start of the actual data
                    # start_of_data = f.tell()  # Adjusted to current file position
                else:
                    # If no zdict, the actual data starts after the flag
                    # start_of_data = 1
                    dctx_bp1 = zstd.ZstdDecompressor()
                
                data_shape = np.frombuffer(f.read(12), dtype=np.int32)
                dtype_str = np.frombuffer(f.read(20), dtype='S20').tobytes().decode('utf-8').rstrip('\x00')
                # Here's the key change: Seek to the start of the actual compressed data
                start_of_data = f.tell() 
                f.seek(start_of_data)
                with dctx_bp1.stream_reader(f) as reader:
                    decompressed_data = reader.read()

                # Assuming the metadata (shape and dtype) is at the beginning of the decompressed data
                
                data_dtype = np.dtype(dtype_str)
                data_array = np.frombuffer(decompressed_data, dtype=data_dtype).reshape(data_shape)
                bp1.append(da.from_array(data_array))

        # Similar adjustments would be needed for `files_bp2` handling

        if path_bp2:
            files_bp2 = sorted(glob.glob(path_bp2))
            self.path_encoded_bp2 = files_bp2
            assert len(files_bp1) == len(files_bp2), 'Error: Both biplanes must have the same number of images.'
            bp2 = []
            for file_path in files_bp2:
                with open(file_path, 'rb') as f:
                    zdict_present = np.frombuffer(f.read(1), dtype=np.uint8)[0]
                    if zdict_present:
                        # Read the length of the zdict
                        zdict_length = np.frombuffer(f.read(4), dtype=np.int32)[0]
                        # Read the zdict itself
                        zdict = f.read(zdict_length)
                        zstd_dict_bp2 = zstd.ZstdCompressionDict(zdict)
                        dctx_bp2 = zstd.ZstdDecompressor(dict_data=zstd_dict_bp2)
                        # Adjust the start of the actual data
                        # start_of_data = f.tell()  # Adjusted to current file position
                    else:
                        # If no zdict, the actual data starts after the flag
                        # start_of_data = 1
                        dctx_bp2 = zstd.ZstdDecompressor()

                    data_shape = np.frombuffer(f.read(12), dtype=np.int32)
                    dtype_str = np.frombuffer(f.read(20), dtype='S20').tobytes().decode('utf-8').rstrip('\x00')
                    start_of_data = f.tell() 
                    # Here's the key change: Seek to the start of the actual compressed data
                    f.seek(start_of_data)
                    with dctx_bp2.stream_reader(f) as reader:
                        decompressed_data = reader.read()

                    # Assuming the metadata (shape and dtype) is at the beginning of the decompressed data

                    data_dtype = np.dtype(dtype_str)
                    data_array = np.frombuffer(decompressed_data, dtype=data_dtype).reshape(data_shape)
                    bp2.append(da.from_array(data_array))

        return bp1, None if path_bp2 is None else bp2
        

    def process_frames(self):
        print('Lazily Processing images...')
        if self.encoded_bp2 is None:
            return [da.where(self.sparse_bp1[i]!=0,self.sparse_bp1[i],self.encoded_bp1[i]) for i in range(len(self.encoded_bp1))], None
        return [da.where(self.sparse_bp1[i]!=0,self.sparse_bp1[i],self.encoded_bp1[i]) for i in range(len(self.encoded_bp1))], [da.where(self.sparse_bp2[j]!=0,self.sparse_bp2[j],self.encoded_bp2[j])for j in range(len(self.encoded_bp2))]



    def run(self):
        print('Inflating images...')
        show_progress_bar = False
        try:
            get_ipython()
            show_progress_bar = True
        except NameError:
            show_progress_bar = False
        if self.use_roi:
            print('Patching in ROI...')
            if show_progress_bar:
                    progress_bar1 = tqdm(total=len(self.processed_bp1), desc="Extracting frames from plane 1", position=0, leave=True)
                    if self.encoded_bp2 is not None:
                        progress_bar2 = tqdm(total=len(self.processed_bp2), desc="Extracting frames from plane 1", position=0, leave=True)
            for k in range(len(self.processed_bp1)):
                num_frames = self.processed_bp1[k].shape[0]

                input_file_name1 = os.path.splitext(os.path.split(os.path.normpath(self.path_encoded_bp1[k]))[1])[0]
                filename1 = f'{self.output_path}{self.stem}_{input_file_name1}.tiff'
                # filename1 = f'{self.output_path}{self.stem}_bp1_part_{k}.tiff'

                with tifffile.TiffWriter(filename1, bigtiff=True) as tif:

                    for i in range(0, num_frames, self.chunk_size):
                        chunk = self.processed_bp1[k][i:i+self.chunk_size].compute()  # Compute a chunk of frames
                        for frame in chunk:
                            tif.write(frame, photometric='minisblack')
                        if show_progress_bar:
                            progress_bar1.update(self.chunk_size)
                if show_progress_bar:
                    progress_bar1.close()
                if self.encoded_bp2 is not None:
                    input_file_name2 = os.path.splitext(os.path.split(os.path.normpath(self.path_encoded_bp2[k]))[1])[0]
                    filename2 = f'{self.output_path}{self.stem}_{input_file_name2}.tiff'
                    # filename2 = f'{self.output_path}{self.stem}_bp2_part_{k}.tiff'
                    with tifffile.TiffWriter(filename2, bigtiff=True) as tif:
                        if show_progress_bar:
                            progress_bar2 = tqdm(total=len(self.processed_bp2), desc="Extracting frames from plane 2", position=0, leave=True)
                        for i in range(0, num_frames, self.chunk_size):
                            chunk = self.processed_bp2[k][i:i+self.chunk_size].compute()  # Compute a chunk of frames
                            for frame in chunk:
                                tif.write(frame, photometric='minisblack')
                            if show_progress_bar:
                                progress_bar2.update(self.chunk_size)
                    if show_progress_bar:
                        progress_bar2.close()

                gc.collect()
                
            
        else:
            print('Extracting background only...')
            
            for k in range(len(self.encoded_bp1)):
                num_frames = self.encoded_bp1[k].shape[0]
                # if type(self.path_encoded_bp1) == list:
                input_file_name1 = os.path.splitext(os.path.split(os.path.normpath(self.path_encoded_bp1[k]))[1])[0]
                # else:
                    # input_file_name1 = os.path.splitext(os.path.split(os.path.normpath(self.path_encoded_bp1))[1])[0]
                
                filename1 = f'{self.output_path}{self.stem}_{input_file_name1}.tiff'
                # filename1 = f'{self.output_path}{self.stem}_bp1_part_{k}.tiff'

                with tifffile.TiffWriter(filename1, bigtiff=True) as tif:
                    if show_progress_bar:
                        progress_bar = tqdm(total=len(self.encoded_bp1), desc="Extracting frames from plane 1", position=0, leave=True)
                    for i in range(0, num_frames, self.chunk_size):
                        chunk = self.encoded_bp1[k][i:i+self.chunk_size].compute()  # Compute a chunk of frames
                        for frame in chunk:
                            tif.write(frame, photometric='minisblack')
                        if show_progress_bar:
                            progress_bar.update(self.chunk_size)
                if show_progress_bar:
                    progress_bar.close()
                if self.encoded_bp2 is not None:
                    # if type(self.path_encoded_bp2) == list:
                    # print ('encoded_bp2:',self.path_encoded_bp2)
                    input_file_name2 = os.path.splitext(os.path.split(os.path.normpath(self.path_encoded_bp2[k]))[1])[0]
                    # else:
                        # input_file_name2 = os.path.splitext(os.path.split(os.path.normpath(self.path_encoded_bp2))[1])[0]
                    filename2 = f'{self.output_path}{self.stem}_{input_file_name2}.tiff'
                    # filename2 = f'{self.output_path}{self.stem}_bp2_part_{k}.tiff'
                    with tifffile.TiffWriter(filename2, bigtiff=True) as tif:
                        if show_progress_bar:
                            progress_bar = tqdm(total=len(self.encoded_bp1), desc="Extracting frames from plane 2", position=0, leave=True)
                        for i in range(0, num_frames, self.chunk_size):
                            chunk = self.encoded_bp2[k][i:i+self.chunk_size].compute()  # Compute a chunk of frames
                            for frame in chunk:
                                tif.write(frame, photometric='minisblack')
                            if show_progress_bar:
                                progress_bar.update(self.chunk_size)
                    if show_progress_bar:
                        progress_bar.close()
                gc.collect()
        print('Done.')

# %%
# bp1='/Users/dimos/raw_image_compression/nir_et_al/img_*_bp1.tiff'
# bp2='/Users/dimos/raw_image_compression/nir_et_al/img_*_bp2.tiff'
# %%

# bp1='/Users/dimos/SPARZ_fig1/sequence-as-stack-MT0.N1.HD-BP-250.tif'
# bp2='/Users/dimos/SPARZ_fig1/sequence-as-stack-MT0.N1.HD-BP+250.tif'

# codec='prores'
# extension='mov' if codec=='prores' else ('zst' if codec=='zstd' else ('avi' if codec=='ffv1' else 'mp4'))
# print('extension:',extension)
# ROI = 'with_ROI'
# out=f'/Users/dimos/SPARZ_fig1/{ROI}/{codec}/data'
# sparsze = True if ROI =='with_ROI' else False
# sparse_bp1 = f'/Users/dimos/SPARZ_fig1/{ROI}/{codec}/data/*-250*.npz' if sparsze else None
# sparse_bp2 = f'/Users/dimos/SPARZ_fig1/{ROI}/{codec}/data/*+250*.npz' if sparsze else None
# # sparse_bp1 = f'/Users/dimos/SPARZ_fig1/{ROI}/{codec}/data/img*bp1.npz' if sparsze else None
# # sparse_bp2 = f'/Users/dimos/SPARZ_fig1/{ROI}/{codec}/data/img*bp2.npz' if sparsze else None


# z=SPARZIP(path_image_files1=bp1,
#            path_image_files2=bp2,
#            output_path=out,
#            stem='test',
#            find_peaks=sparsze)
# #%%
# # z.run(codec=codec,compute_zstd_dict=False,compression_level=0)
# # %%
# u=SPARUNZIP(path_sparse_bp1=sparse_bp1,
#             path_encoded_bp1=f'/Users/dimos/SPARZ_fig1/{ROI}/{codec}/data/*-250*.{extension}',
#             stem='zstd_test',
#             output_path=out,
#             path_sparse_bp2=sparse_bp2,
#             path_encoded_bp2=f'/Users/dimos/SPARZ_fig1/{ROI}/{codec}/data/*+250*.{extension}' if sparse else None,
#             use_roi=sparsze)
# # %%
# u.run()

# # # # %%

# # %%
# u.decode(u.path_encoded_bp1,u.path_encoded_bp2)
# # %%
# m1=u.encoded_bp1[0].compute()
# # %%
# import matplotlib.pyplot as plt
# plt.imshow(m1[0,:,:])
# # %%
# m = u.decode(u.path_encoded_bp1[0],None)
# # %%
# u.path_encoded_bp1
# # %%
# m[0][0].compute()

# # %%
# m1=m[0][0].compute()
# # %%
# import matplotlib.pyplot as plt
# plt.imshow(m1[1,:,:])
# # %%
# files_bp1 = sorted(glob.glob(f'/Users/dimos/SPARZ_fig1/{ROI}/{codec}/data/*-250*.{extension}'))
# # bp1 = [self.load_mp4(file) for file in files_bp1]
# # print ('bp1',bp1[0])
# # %%
# import matplotlib.pyplot as plt
# plt.imshow(u.load_mp4(files_bp1[0])[100,:,:].compute())

# # %%
# u.load_mp4(files_bp1[0])[100,:,:].compute()
# #%%
# container = av.open(files_bp1[0])
# #%%
# video_stream = container.streams.video[0]
# frame_count = video_stream.frames

# for frame in container.decode(video_stream):
#     print (frame)
#     np_frame = frame.reformat(format='gray16le')
#     np_frame = np.frombuffer(np_frame.planes[0], dtype=np.uint16)
#     print(np_frame)
#     break

# #%%
# def load_mp4(file_path):
#     try:
#         container = av.open(file_path)
#     except av.AVError as e:
#         print(f"Failed to open file {file_path}: {e}")
#         return None

#     video_stream = container.streams.video[0]
#     frames = []

#     for packet in container.demux(video_stream):
#         for frame in packet.decode():
#             try:
#                 if 'yuv' in frame.format.name:
#                     # Extract the Y plane (luminance)
#                     y_plane = frame.to_ndarray()  # Extract Y component which is index 0
#                     if '10le' in frame.format.name:
#                         # Scale 10-bit values to 16-bit
#                         y_plane_16bit = np.left_shift(y_plane.astype(np.uint16), 6)
#                     else:
#                         # Assuming it's 8-bit, scale to 16-bit
#                         y_plane_16bit = np.left_shift(y_plane.astype(np.uint16), 8)
#                     frames.append(y_plane_16bit)
#                 else:
#                     print(f"Unsupported frame format encountered: {frame.format.name}")
#             except Exception as e:
#                 print(f"Error processing frame: {e}")

#     if not frames:
#         print("No frames were processed. Check video format and stream contents.")
#         return None

#     # Stack frames into a single 3D numpy array
#     video_data = np.stack(frames, axis=0)
#     return video_data
# #%%
# ff=glob.glob(f'/Users/dimos/SPARZ_fig1/{ROI}/{codec}/data/*-250*.{extension}')[0]
# ff

# #%%
# # Usage
# video_frames = load_mp4(ff)
# if video_frames is not None:
#     print("Video loaded and processed successfully.")
# else:
#     print("Failed to load or process video.")

# # %%
# # %%
# def load_mp4(file_path):
#     try:
#         container = av.open(file_path)
#     except av.AVError as e:
#         print(f"Failed to open file {file_path}: {e}")
#         return None

#     video_stream = container.streams.video[0]
#     frames = []

#     for packet in container.demux(video_stream):
#         for frame in packet.decode():
#             if frame.format.name == 'yuv420p10le':
#                 # Extract the Y plane data, which is the first plane.
#                 y_plane = frame.planes[0]
#                 # Convert the plane's buffer to a 1D numpy array of uint16 (safe cast since we're dealing with 10-bit data)
#                 y_data = np.frombuffer(y_plane, np.uint16)
                
#                 # Reshape the data to match the height and width of the frame.
#                 y_data = y_data.reshape((frame.height, frame.width))

#                 # Scale 10-bit values to 16-bit
#                 y_data_16bit = np.left_shift(y_data, 6)
                
#                 frames.append(y_data_16bit)
#             else:
#                 print(f"Unsupported frame format encountered: {frame.format.name}")

#     if not frames:
#         print("No frames were processed. Check video format and stream contents.")
#         return None

#     # Stack frames into a single 3D numpy array
#     video_data = np.stack(frames, axis=0)
#     return video_data

# # Usage
# video_frames = load_mp4(ff)
# if video_frames is not None:
#     print("Video loaded and processed successfully.")
# else:
#     print("Failed to load or process video.")
# # %%
# plt.imshow(video_frames[0,:,:])
# # %%

# %%
