#%%
import numpy as np
import tifffile
from tifffile import TiffFile
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
# from skvideo.io import FFmpegWriter
from reader import imread as vimread
import dask
from dask import compute
import gc
from tqdm import tqdm
import os
import pandas as pd
import json
from dask import delayed
# import threading
import ffmpeg
from dask.diagnostics import ProgressBar
import concurrent.futures
import av

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
                 reflect_bp2:bool = False, 
                 find_peaks:bool = True,
                 align_planes:bool=False):
      
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
        self.find_roi = find_peaks
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
                for i in range(0, len(files1), 250):
                    images = [dask_image.imread.imread(f) for f in files1[i:i+250]]
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
                for i in range(0, len(files2), 250):
                    images = [dask_image.imread.imread(f) for f in files2[i:i+250]]
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
        print('Mapping functions to images...')
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
    


    def deflate(self):
        print('Deflating images...')
        with dask.config.set(scheduler='threads'):
            # Prepare a list to store delayed operations
            saves = []
            for i in range(len(self.processed_bp1)):
                # Directly append delayed save_npz operations to the list
                flnm1 = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files1[i]))[1])[0]
                # saves.append(delayed(sparse.save_npz)(self.output_path+self.stem+'_peaks_bp1_part_'+str(i)+'.npz',self.processed_bp1[i]))
                saves.append(delayed(sparse.save_npz)(self.output_path+flnm1+'.npz',self.processed_bp1[i]))
                if self.single_plane == False:
                    flnm2 = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files2[i]))[1])[0]
                    saves.append(delayed(sparse.save_npz)(self.output_path+flnm2+'.npz',self.processed_bp2[i]))
                    # saves.append(delayed(sparse.save_npz)(self.output_path+self.stem+'_peaks_bp2_part_'+str(i)+'.npz',self.processed_bp2[i]))
            # Perform the save_npz operations
            progress_bar = tqdm(total=len(saves), desc="Creating sparse matrices", position=0, leave=True)
            
            for i in range(0, len(saves), self.batch_size):
                batch = saves[i:i+self.batch_size]
                dask.compute(*batch, scheduler='threads',num_workers=int(os.cpu_count() * 0.75))

                progress_bar.update(self.batch_size)
            progress_bar.close()
        end = time.time()


    def encode(self,compression_lvl:int=0):
        compression_levels = {0: {
                                                'vcodec': 'libx265',
                                                'pix_fmt': 'yuv444p16le',
                                                'channels': '1',
                                                'x265-params': 'lossless=1',
                                         },
                            1:{
                                             'vcodec': 'libx265',
                                             'crf': '0',
                                             'pix_fmt': 'yuv444p16le',
                                             'channels': '1'
                                         },
                            2:{
                                             'vcodec': 'libx265',
                                             'crf': '10',
                                             'pix_fmt': 'yuv444p16le',
                                             'channels': '1'
                                         },

                            3:{
                                             'vcodec': 'libx265',
                                             'crf': '15',
                                             'pix_fmt': 'yuv444p16le',
                                             'channels': '1'
                                         }                                      
                                         
                                         
                        }
        print('Compressing video...')
        
        # Process videos using delayed
        writes = []
        for k in range(len(self.processed_bp1)):
            input_file_name = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files1[k]))[1])[0]
            video_name = f'{self.output_path}{self.stem}_{input_file_name}_compression_level_{compression_lvl}_part_{k}.mp4'
            writer_args = compression_levels[compression_lvl]

            if self.single_plane:
                writes.append(delayed(self.write_frames_to_video)(
                    self.bp1[k], video_name, writer_args
                ))
            else:
                input_file_name1 = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files1[k]))[1])[0]
                video_name1 = f'{self.output_path}{input_file_name1}_compression_level_{compression_lvl}.mp4'
                # video_name1 = f'{self.output_path}{self.stem}_bp1_compression_level_{compression_lvl}_part_{k}.mp4'
                writes.append(delayed(self.write_frames_to_video)(
                    self.bp1[k], video_name1, writer_args
                ))
                input_file_name2 = os.path.splitext(os.path.split(os.path.normpath(self.path_image_files2[k]))[1])[0]
                video_name2 = f'{self.output_path}{input_file_name2}_compression_level_{compression_lvl}.mp4'
                # video_name2 = f'{self.output_path}{self.stem}_bp2_compression_level_{compression_lvl}_part_{k}.mp4'
                writes.append(delayed(self.write_frames_to_video)(
                    self.bp2[k], video_name2, writer_args
                ))

        # Execute the delayed writes
        with ProgressBar():
            compute(*writes, scheduler='threads',num_workers=int(os.cpu_count() * 0.75))

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
            'y': None
            # 's': '{}x{}'.format(*input_frames.shape[1:3][[::-1]])
        }

        ffmpeg_input = ffmpeg.input('pipe:', **input_dict)
        # Create a ffmpeg output with the writer arguments
        writer_args['s'] = input_dict['s']
        # with open(os.devnull, "w") as devnull:
        ffmpeg_output = ffmpeg.output(ffmpeg_input, video_name, **writer_args)

        # Run the ffmpeg command
        # ffmpeg.run(ffmpeg_output, input=input_frames.tobytes())
        ffmpeg.run(ffmpeg_output, input=bp1_frames.tobytes(), capture_stdout=True, capture_stderr=False)

    def run(self,compression_level:int=0):#,find_peaks:bool=True):
        if self.find_roi:
            self.deflate()
            gc.collect()
        self.encode(compression_level)


class SPARUNZIP:
    def __init__(self, path_sparse_bp1:str, path_encoded_bp1:str, stem:str, output_path:str, path_sparse_bp2:str=None, path_encoded_bp2:str=None, use_roi:bool=True, chunk_size:int=10):
        self.path_encoded_bp1, self.path_encoded_bp2 = path_encoded_bp1, path_encoded_bp2
        self.encoded_bp1, self.encoded_bp2 = self.decode(path_encoded_bp1, path_encoded_bp2)
        self.shapes = [x.shape[:3] for x in self.encoded_bp1]
        self.sparse_bp1, self.sparse_bp2 = self.load_sparse(path_sparse_bp1, path_sparse_bp2, self.shapes)
        self.processed_bp1, self.processed_bp2 = self.process_frames()
        self.stem = stem
        if output_path[-1] != '/':
            self.output_path = output_path+"/"
        else:
            self.output_path = output_path
        self.use_roi = use_roi
        self.chunk_size = chunk_size

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
                # bp1 = da.from_array(sparse.load_npz(files_bp1[i]), chunks=(1,shapes[1],shapes[2]))
                # bp2 = da.from_array(sparse.load_npz(files_bp2[i]), chunks=(1,shapes[1],shapes[2]))
            return bp1, bp2
        # return bp1.map_blocks(lambda x: x.todense(), dtype='int16'), bp2.map_blocks(lambda x: x.todense(), dtype='int16')
        bp1 =[]
        for i in range(len(files_bp1)):
            bp1.append(da.from_array(sparse.load_npz(files_bp1[i]), chunks=(1,shapes[i][1],shapes[i][2])))
        return bp1, None
    
    def decode(self, path_bp1:str, path_bp2:str):
        print('Decoding images...')
        files_bp1 = sorted(glob.glob(path_bp1))
        if path_bp2 is not None:
            files_bp2 = sorted(glob.glob(path_bp2))
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
    
    def process_frames(self):
        print('Lazily Processing images...')
        if self.encoded_bp2 is None:
            return [da.where(self.sparse_bp1[i]!=0,self.sparse_bp1[i],self.encoded_bp1[i]) for i in range(len(self.encoded_bp1))], None
        return [da.where(self.sparse_bp1[i]!=0,self.sparse_bp1[i],self.encoded_bp1[i]) for i in range(len(self.encoded_bp1))], [da.where(self.sparse_bp2[j]!=0,self.sparse_bp2[j],self.encoded_bp2[j])for j in range(len(self.encoded_bp2))]

    # def save_file(self,arr, block_info=None):
    #     """ Save file to foo-x-y.tif, where x and y are block locations """
    #     filename = self.output_path+"decoded_bp1" + "-".join(map(str, block_info[0]["chunk-location"])) + ".tiff"
    #     tifffile.imwrite(filename, arr, photometric='minisblack')
    #     return arr

    def run(self):
        print('Inflating images...')
        if self.use_roi:
            print('Patching in ROI...')
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
                        progress_bar1.update(self.chunk_size)
                progress_bar1.close()
                if self.encoded_bp2 is not None:
                    input_file_name2 = os.path.splitext(os.path.split(os.path.normpath(self.path_encoded_bp2[k]))[1])[0]
                    filename2 = f'{self.output_path}{self.stem}_{input_file_name2}.tiff'
                    # filename2 = f'{self.output_path}{self.stem}_bp2_part_{k}.tiff'
                    with tifffile.TiffWriter(filename2, bigtiff=True) as tif:
                        progress_bar2 = tqdm(total=len(self.processed_bp2), desc="Extracting frames from plane 2", position=0, leave=True)
                        for i in range(0, num_frames, self.chunk_size):
                            chunk = self.processed_bp2[k][i:i+self.chunk_size].compute()  # Compute a chunk of frames
                            for frame in chunk:
                                tif.write(frame, photometric='minisblack')
                            progress_bar2.update(self.chunk_size)
                    progress_bar2.close()
                
            
        else:
            print('Extracting background only...')
            
            for k in range(len(self.encoded_bp1)):
                num_frames = self.encoded_bp1[k].shape[0]

                input_file_name1 = os.path.splitext(os.path.split(os.path.normpath(self.path_encoded_bp1[k]))[1])[0]
                filename1 = f'{self.output_path}{self.stem}_{input_file_name1}.tiff'
                # filename1 = f'{self.output_path}{self.stem}_bp1_part_{k}.tiff'

                with tifffile.TiffWriter(filename1, bigtiff=True) as tif:
                    progress_bar = tqdm(total=len(self.encoded_bp1), desc="Extracting frames from plane 1", position=0, leave=True)
                    for i in range(0, num_frames, self.chunk_size):
                        chunk = self.encoded_bp1[k][i:i+self.chunk_size].compute()  # Compute a chunk of frames
                        for frame in chunk:
                            tif.write(frame, photometric='minisblack')
                        progress_bar.update(self.chunk_size)
                progress_bar.close()
                if self.encoded_bp2 is not None:
                    input_file_name2 = os.path.splitext(os.path.split(os.path.normpath(self.path_encoded_bp2[k]))[1])[0]
                    filename2 = f'{self.output_path}{self.stem}_{input_file_name2}.tiff'
                    # filename2 = f'{self.output_path}{self.stem}_bp2_part_{k}.tiff'
                    with tifffile.TiffWriter(filename2, bigtiff=True) as tif:
                        progress_bar = tqdm(total=len(self.encoded_bp1), desc="Extracting frames from plane 2", position=0, leave=True)
                        for i in range(0, num_frames, self.chunk_size):
                            chunk = self.encoded_bp2[k][i:i+self.chunk_size].compute()  # Compute a chunk of frames
                            for frame in chunk:
                                tif.write(frame, photometric='minisblack')
                            progress_bar.update(self.chunk_size)
                    progress_bar.close()
        print('Done.')