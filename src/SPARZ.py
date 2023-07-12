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
# from scipy.ndimage import fourier_shift
from scipy.ndimage import shift
# from concurrent import futures
# import multiprocessing
import dask.array as da
import dask_image.imread
from dask import delayed
from skvideo.io import FFmpegWriter
from reader import imread as vimread
import dask
# dask.config.set(scheduler='threads')
from concurrent.futures import ThreadPoolExecutor
from dask import compute


#%%
class SPARZIP:
    def __init__(self, path_image_files1:str,  
                 stem:str, 
                 output_path:str,
                 path_image_files2:str = None,
                 rel_threshold:float = 0.5, 
                 epsilon:int = 12, 
                 kernel_size:int = 3, 
                 comp_level:int=0, 
                 reflect_bp2:bool = False, 
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
        rel_threshold : float, optional
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
        self.bp1, self.bp2 = self.load_images(path_image_files1, path_image_files2)
        if self.bp1.dtype == 'float32':
            self.bp1, self.bp2 = self.to_16bit()
        self.stem = stem
        if output_path[-1] != '/':
            self.output_path = output_path+"/"
        else:
            self.output_path = output_path
        self.rel_threshold = rel_threshold
        self.epsilon = epsilon
        self.kernel_size = kernel_size
        self.compression_level = comp_level
        if reflect_bp2:
            if self.single_plane:
                print('Skipping reflection on single plane data.')
            else:
                self.bp2 = np.flip(self.bp2, 2)
        if align_planes:
            if self.single_plane:
                print('Cannot align single plane data. Skipping.')
            else:
                self.bp2 = self.align_planes(self.bp1, self.bp2)

        self.processed_bp1,self.processed_bp2 = self.process_images()

    def load_images(self, path_image_files1:str, path_image_files2:str):
        print ('Lazily loading images...')
        if len(glob.glob(path_image_files1)) == 1:
            #glob the files in the folder
            return dask_image.imread.imread(path_image_files1), dask_image.imread.imread(path_image_files2)
        p1 = [dask_image.imread.imread(f) for f in sorted(glob.glob(path_image_files1))]
        if self.single_plane:
            print ('Working with single plane data.')
            return da.concatenate(p1, axis=0), None
        print('Working with biplane data.')
        p2 = [dask_image.imread.imread(f) for f in sorted(glob.glob(path_image_files2))]
        return da.concatenate(p1, axis=0), da.concatenate(p2, axis=0)


    def to_16bit(self):
        print('Float32 data detected.')
        print('Converting to 16 bit...')
        z1 = self.bp1.map_blocks(lambda x:((x - x.min())/ (x.max() - x.min())*65535).astype('uint16'), dtype='uint16')
        if self.single_plane:
            return z1, None
        z2 = self.bp2.map_blocks(lambda x:(( x - x.min())/ (x.max() - x.min()) * 65535).astype('uint16'), dtype='uint16')
        return z1, z2 
    
    def find_peaks(self, image:np.ndarray, kernel_size:int, min_distance:int = 1):

        #find the local maxima in the image
        coordinates = peak_local_max(image, 
                                    threshold_rel= self.rel_threshold, 
                                    min_distance=min_distance, 
                                    footprint=np.ones((kernel_size, kernel_size))
                                    )

        return coordinates

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

    def union(self, peaks1:np.ndarray, peaks2:np.ndarray, bp1):
        merged = np.vstack([peaks1, peaks2])

        merged_peaks = np.zeros(bp1)
        merged_peaks[merged[:, 0], merged[:, 1]] = 1    
        return merged_peaks

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
        frames = plane1.shape[0]
        # print (frames)
        if frames < 50:
            shifts, _, _ = phase_cross_correlation(plane1, plane2,
                                                        upsample_factor=100)

        
        else:
            # sample = np.random.randint(int(frames*0.48), int(frames*0.52),150)
            b1 = plane1[int(plane1.shape[0]*0.48): int(plane1.shape[0]*0.52)]
            b2 = plane2[int(plane2.shape[0]*0.48): int(plane2.shape[0]*0.52)]
            shifts, _, _ = phase_cross_correlation(b1, b2,
                                                        upsample_factor=100)
            
            #We don't want to shift frame numbers, so we set the first value to 0
            shifts[0] = 0
            print(f'Applying mean shift {shifts} to all frames.')
            
        return plane2.map_blocks(lambda x: shift(x, shifts, mode='constant'))

            

    # def align_planes(self, plane1:np.ndarray, plane2:np.ndarray):
        
    #     print('Aligning planes, please wait...')
    #     frames = plane1.shape[0]
    #     # print (frames)

    #     if frames < 50:
    #         print('all')
    #         shifts, _, _ = phase_cross_correlation(plane1, plane2,
    #                                                     upsample_factor=100)
    #         # offset_image = fourier_shift(np.fft.fftn(plane2), shifts)
    #         print(f'shift: {shifts}')
    #         return plane2.map_blocks(lambda x: shift(x, shifts, mode='constant'))
        
    #     sample = np.random.randint(int(frames*0.4), int(frames*0.6),150)
        
    #     shifts_list =[]
    #     for s in sample:
    #         shifts, _, _ = phase_cross_correlation(plane1.blocks[s].compute(), plane2.blocks[s].compute(),
    #                                                     upsample_factor=100)
    #         shifts_list.append(shifts)
    #     tmp_shift = np.mean(shifts_list, axis=0)
    #     mean_shift = np.zeros(3)
    #     mean_shift[1:] = tmp_shift
    #     print(f'Applying mean shift {mean_shift[1:]} to all frames.')

    #     return plane2.map_blocks(lambda x: shift(x, mean_shift, mode='constant'))
    def get_raw_frame(self,start_frame:int=None,end_frame:int=None,plane:str='bp1'):
        if start_frame is None:
            raise ValueError('Please provide a start frame.')
        if end_frame is None:
            end_frame = start_frame+1
        if plane == 'bp1':
            return self.bp1.blocks[start_frame:end_frame,0].compute()
        else:
            return self.bp2.blocks[start_frame:end_frame,0].compute()

    def get_processed_frame(self,start_frame:int=None,end_frame:int=None,plane:str='bp1'):
        if start_frame is None:
            raise ValueError('Please provide a start frame.')
        if end_frame is None:
            end_frame = start_frame+1
        if plane == 'bp1':
            return self.processed_bp1.blocks[start_frame:end_frame,0].compute().todense()
        else:
            return self.processed_bp2.blocks[start_frame:end_frame,0].compute().todense()
        
    def process_images(self):
        print('Mapping functions to images...')

        map1 = self.bp1.map_blocks(lambda x: self.find_peaks(x[0,:,:],self.kernel_size,min_distance=1), dtype='int16')
        if self.single_plane == False:
            map2 = self.bp2.map_blocks(lambda x: self.find_peaks(x[0,:,:],self.kernel_size,min_distance=1), dtype='int16')
            map_union = da.map_blocks(self.union, map1, map2, self.bp1[0,:,:].shape, dtype='int16')
            map_kernel = map_union.map_blocks(self.add_kernel, self.kernel_size, dtype='int16')
            sp1 = da.where(map_kernel, self.bp1, 0)
            sp2 = da.where(map_kernel, self.bp2, 0)
            print('Done.')
            return sp1.map_blocks(sparse.COO, dtype='int16'), sp2.map_blocks(sparse.COO, dtype='int16')
        
        def add_mask(peaks:np.ndarray):
            tmp = np.zeros(self.bp1[0,:,:].shape)
            tmp[peaks[:, 0], peaks[:, 1]] = 1 
            return tmp
        map_mask = map1.map_blocks(lambda x: add_mask(x), dtype='int16')
        map_kernel = map_mask.map_blocks(self.add_kernel, self.kernel_size, dtype='int16')
        sp1 = da.where(map_kernel, self.bp1, 0)
        print('Done.')
        return sp1.map_blocks(sparse.COO, dtype='int16'), None
    

    def deflate(self):
        start = time.time()
        print('Deflating images...')
        with dask.config.set(pool=ThreadPoolExecutor(8)):
            sparse.save_npz(self.output_path+self.output_path+self.stem+'_peaks_bp1.npz',self.processed_bp1.compute())
            if self.single_plane == False:
                sparse.save_npz(self.output_path+self.output_path+self.stem+'_peaks_bp2.npz',self.processed_bp2.compute())
        end = time.time()
        print('Deflate completed in ', (end-start)/60, ' minutes')


    def encode(self):
        compression_levels = {0: {
                                                '-vcodec': 'libx265',
                                                '-pix_fmt': 'yuv444p12le',
                                                '-channels': '1',
                                                '-x265-params': 'lossless=1',
                                         },
                            1:{
                                             '-vcodec': 'libx265',
                                             '-crf': 0,
                                             '-pix_fmt': 'yuv444p12le',
                                             '-channels': '1'
                                         },
                            2:{
                                             '-vcodec': 'libx265',
                                             '-crf': 10,
                                             '-pix_fmt': 'yuv444p12le',
                                             '-channels': '1'
                                         },

                            3:{
                                             '-vcodec': 'libx265',
                                             '-crf': 15,
                                             '-pix_fmt': 'yuv444p12le',
                                             '-channels': '1'
                                         }                                      
                                         
                                         
                        }

        print('Compressing video...')

        writer1 = FFmpegWriter(f'{self.output_path}{self.stem}_bp1_compression_level_{self.compression_level}.mp4',
                                        outputdict=compression_levels[self.compression_level]
                                        )                                         

        writer2 = FFmpegWriter(f'{self.output_path}{self.stem}_bp2_compression_level_{self.compression_level}.mp4',
                                        outputdict=compression_levels[self.compression_level]
                                        )

        # Rechunk the array to have many small chunks
        # bp1 = self.bp1.rechunk((-1, 'auto', 'auto'))
        # bp2 = self.bp2.rechunk((-1, 'auto', 'auto')) if not self.single_plane else None

        writes = []
        for i in range(self.bp1.numblocks[0]):
            write1 = delayed(writer1.writeFrame)(self.bp1.blocks[i,0])
            writes.append(write1)

            if not self.single_plane:
                write2 = delayed(writer2.writeFrame)(self.bp2.blocks[i,0])
                writes.append(write2)
        try:
            with dask.config.set(scheduler='threads'):
                compute(*writes)
        
        except AttributeError:
            print('WARNING: Parallel writing failed. Writing frames sequentially.')
            for write in writes:
                write.compute()

        writer1.close()
        if not self.single_plane:
            writer2.close()
    
    # def encode(self):
        
    #     compression_levels = {0: {
    #                                             '-vcodec': 'libx265',
    #                                             '-pix_fmt': 'yuv444p12le',
    #                                             '-channels': '1',
    #                                             '-x265-params': 'lossless=1',
    #                                      },
    #                         1:{
    #                                          '-vcodec': 'libx265',
    #                                          '-crf': 0,
    #                                          '-pix_fmt': 'yuv444p12le',
    #                                          '-channels': '1'
    #                                      },
    #                         2:{
    #                                          '-vcodec': 'libx265',
    #                                          '-crf': 10,
    #                                          '-pix_fmt': 'yuv444p12le',
    #                                          '-channels': '1'
    #                                      },

    #                         3:{
    #                                          '-vcodec': 'libx265',
    #                                          '-crf': 15,
    #                                          '-pix_fmt': 'yuv444p12le',
    #                                          '-channels': '1'
    #                                      }                                      
                                         
                                         
    #                     }

    #     print('Compressing video...')
    #     writer1 = FFmpegWriter(f'{self.output_path}{self.stem}_bp1_compression_level_{self.compression_level}.mp4',
    #                                      outputdict=compression_levels[self.compression_level]
    #                                     )                                         

    #     writer2 = FFmpegWriter(f'{self.output_path}{self.stem}_bp2_compression_level_{self.compression_level}.mp4',
    #                                      outputdict=compression_levels[self.compression_level]
    #                                      )
        
    #     for i in range(self.bp1.numblocks[0]):
    #         writer1.writeFrame(self.bp1.blocks[i,0].compute())
    #         if self.single_plane == False:
    #             writer2.writeFrame(self.bp2.blocks[i,0].compute())
    #     writer1.close()
    #     if self.single_plane == False:
    #         writer2.close()
    
    def deflate_encode(self):
        start = time.time()
        print('Deflating images...')
        with dask.config.set(pool=ThreadPoolExecutor(4)):
            sparse.save_npz(self.output_path+self.stem+'_peaks_bp1.npz',self.processed_bp1.compute())
            if self.single_plane == False:
                sparse.save_npz(self.output_path+self.stem+'_peaks_bp2.npz',self.processed_bp2.compute())
        self.encode()
        end = time.time()
        print('Deflate completed in ', (end-start)/60, ' minutes')


class SPARUNZIP:
    def __init__(self, path_sparse_bp1:str, path_sparse_bp2:str, path_encoded_bp1:str, path_encoded_bp2:str, stem:str, output_path:str):
        self.encoded_bp1, self.encoded_bp2 = self.decode(path_encoded_bp1, path_encoded_bp2)
        self.shapes = self.encoded_bp1.shape[:3]
        self.sparse_bp1, self.sparse_bp2 = self.load_sparse(path_sparse_bp1, path_sparse_bp2, self.shapes)
        self.processed_bp1, self.processed_bp2 = self.process_frames()
        self.stem = stem
        if output_path[-1] != '/':
            self.output_path = output_path+"/"
        else:
            self.output_path = output_path

    def load_sparse(self,sparse_bp1:str, sparse_bp2:str, shapes:tuple):
        print ('Loading sparse matrices...')
        bp1 = da.from_array(sparse.load_npz(sparse_bp1), chunks=(1,shapes[1],shapes[2]))
        bp2 = da.from_array(sparse.load_npz(sparse_bp2), chunks=(1,shapes[1],shapes[2]))
        return bp1.map_blocks(lambda x: x.todense(), dtype='int16'), bp2.map_blocks(lambda x: x.todense(), dtype='int16')
    
    def decode(self, path_bp1:str, path_bp2:str):
        print('Decoding images...')
        return vimread(path_bp1, dtypes='uint16'), vimread(path_bp2, dtypes='uint16')
    
    def process_frames(self):
        print('Lazily Processing images...')
        return da.where(self.sparse_bp1!=0,self.sparse_bp1,self.encoded_bp1), da.where(self.sparse_bp2!=0,self.sparse_bp2,self.encoded_bp2)

    # def save_file(self,arr, block_info=None):
    #     """ Save file to foo-x-y.tif, where x and y are block locations """
    #     filename = self.output_path+"decoded_bp1" + "-".join(map(str, block_info[0]["chunk-location"])) + ".tiff"
    #     tifffile.imwrite(filename, arr, photometric='minisblack')
    #     return arr
    
    def inflate(self):
        print('Inflating images...')
        num_frames = self.encoded_bp1.shape[0]
        print (num_frames)
        for i in range(num_frames):
            tifffile.imwrite(f'{self.output_path}{self.stem}_bp1_{i:0{len(str(num_frames))}}.tiff', self.encoded_bp1[i].compute(), photometric='minisblack')
            tifffile.imwrite(f'{self.output_path}{self.stem}_bp2_{i:0{len(str(num_frames))}}.tiff', self.encoded_bp2[i].compute(), photometric='minisblack')

        # self.encoded_bp1.map_blocks(self.save_file, dtype=self.encoded_bp1.dtype).compute()
        
  
        


        
   


#%%
path1 = '/Users/dimos/raw_image_compression/tubulin_biplane/COS-7_Tubulin_SOFI_Flip565_biplane_reflected.tiff'
path2 = '/Users/dimos/raw_image_compression/tubulin_biplane/COS-7_Tubulin_SOFI_Flip565_biplane_transmitted.tiff'
# path1 = '/Users/dimos/raw_image_compression/nir_et_al/img_*_bp1.tiff'
# path2 = '/Users/dimos/raw_image_compression/nir_et_al/img_*_bp2.tiff'
# # # beads_path = '/Users/dimos/raw_image_compression/tubulin_biplane/Biplane_beads_calibration.tif'
kernel_size = 9     
rel_thresh = 0.45

#%%
z = SPARZIP(path1, stem='tub',output_path='/Users/dimos/Desktop/', path_image_files2=path2,rel_threshold = rel_thresh, kernel_size=kernel_size,reflect_bp2=True,align_planes=True)
# # #%%
# #%%
# plt.imshow(z.get_processed_frame(start_frame=10)[0,:,:])
#%%
z.deflate_encode()
# # #%% 
# # start = time.time()
# # u=SPARUNZIP('/Users/dimos/Desktop/tub_peaks_bp1.npz','/Users/dimos/Desktop/tub_peaks_bp2.npz','/Users/dimos/Desktop/tub_bp1_compression_level_0.mp4','/Users/dimos/Desktop/tub_bp2_compression_level_0.mp4',output_path="/Users/dimos/Desktop/test/",stem='nir')
# # end = time.time()
# # print (end-start)

# # # # %%
# # u.inflate()
# # %%
# z.bp1.blocks[0:1,0].compute()
# %%
