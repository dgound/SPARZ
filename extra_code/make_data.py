#%%
from SPARZ import SPARZIP, SPARUNZIP
import timeit,time
import gc
import statistics
import pandas as pd

bp1='/mnt/altnas/work/dimos/raw_data_tiff/*bp1.tiff'
bp2='/mnt/altnas/work/dimos/raw_data_tiff/*bp2.tiff'
# codec='prores'
# out=f'/mnt/altnas/work/dimos/SPARZ_fig1/no_ROI/{codec}/data'

# bp1='/Users/dimos/SPARZ_fig1/sequence-as-stack-MT0.N1.HD-BP-250.tif'
# bp2='/Users/dimos/SPARZ_fig1/sequence-as-stack-MT0.N1.HD-BP+250.tif'



ROI = 'with_ROI'
comp_level = 0
# codecs=['zstd','prores']
# codecs = ['av1','ffv1','prores','x264','x265']
codecs  =['av1','ffv1','prores','x264','x265','zstd']
tries=10
all_the_times = []
# run_times=[]
#%%
for codec in codecs:
   codec_times = {
         'codec':codec,
         # 'overhead':{},
         'compression_overhead_mean':0,
         'compression_overhead_std_dev':0,
         # 'times':[],
         'compression_mean':0,
         'compression_std_dev':0,
         # 'overhead':{},
         'decompression_overhead_mean':0,
         'decompression_overhead_std_dev':0,
         # 'times':[],
         'decompression_mean':0,
         'decompression_std_dev':0
   }
   extension='mov' if codec=='prores' else ('avi' if codec=='ffv1' else ('zst' if codec=='zstd' else 'mp4'))
   # out=f'/Users/dimos/SPARZ_fig1/{ROI}/{codec}/data'
   out=f'/mnt/altnas/work/dimos/SPARZ_fig1/{ROI}/{codec}/data'
   sparse = True if ROI=='with_ROI' else False
   # sparse_bp1 = f'/Users/dimos/SPARZ_fig1/{ROI}/{codec}/data/*-250*.npz' if sparse else None
   # sparse_bp2 = f'/Users/dimos/SPARZ_fig1/{ROI}/{codec}/data/*+250*.npz' if sparse else None
   sparse_bp1 = f'/mnt/altnas/work/dimos/SPARZ_fig1/{ROI}/{codec}/data/*bp1*.npz' if sparse else None
   sparse_bp2 = f'/mnt/altnas/work/dimos/SPARZ_fig1/{ROI}/{codec}/data/*bp2*.npz' if sparse else None


   z=SPARZIP(path_image_files1=bp1,
            path_image_files2=bp2,
            output_path=out,
            stem='fig1',
            find_peaks=sparse)

   def overhead():
        try:
            z=SPARZIP(path_image_files1=bp1,
            path_image_files2=bp2,
            output_path=out,
            stem='fig1',
            find_peaks=sparse)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Data loading failed: {e}")
        return z

   def compress_images(z):
      try:
         z.run(codec=codec,compute_zstd_dict=True,compression_level=comp_level)
         print("Compression completed successfully.")
      except Exception as e:
            print(f"Compression failed: {e}")

#    execution_time = timeit.repeat(compress_images, number=10)
#    print(f"Compression {codec} execution time: {execution_time} seconds")

   times = []
   overhead_times = []

   for _ in range(tries):
      start_time = time.perf_counter()
      z=overhead()
      end_time = time.perf_counter()
      overhead_times.append(end_time - start_time)
      start_time = time.perf_counter()
      compress_images(z)
      end_time = time.perf_counter()
      times.append(end_time - start_time)

   overhead_mean_time = statistics.mean(overhead_times)
   overhead_std_dev_time = statistics.stdev(overhead_times)

   mean_time = statistics.mean(times)
   std_dev_time = statistics.stdev(times)

#    print(f"{codec} Overhead Times: {overhead_times}")
#    print(f"{codec} Overhead Mean: {overhead_mean_time}")
#    print(f"{codec} Overhead Standard Deviation: {overhead_std_dev_time}")

#    print(f"{codec} Compress Times: {times}")
#    print(f"{codec} Mean: {mean_time}")
#    print(f"{codec} Standard Deviation: {std_dev_time}")

   # codec_times['compression'] = {'codec':codec,
   #    #   'overhead': overhead_times,
   #      'overhead_mean': mean_time,
   #      'overhead_std_dev': std_dev_time,
   #    #   'times': times,
   #      'mean': mean_time,
   #      'std_dev': std_dev_time
   # }
   codec_times['compression_overhead_mean']=overhead_mean_time
   codec_times['compression_overhead_std_dev']=overhead_std_dev_time
   codec_times['compression_mean']=mean_time
   codec_times['compression_std_dev']=std_dev_time

   gc.collect()

   def decomp_overhead():
        try:
            u=SPARUNZIP(#path_sparse_bp1=f'/mnt/altnas/work/dimos/SPARZ_fig1/with_ROI/{codec}/data/*bp1*.npz',
               path_sparse_bp1 = sparse_bp1,
               # path_encoded_bp1=f'/Users/dimos/SPARZ_fig1/{ROI}/{codec}/data/*-250_*.{extension}',
               path_encoded_bp1=f'/mnt/altnas/work/dimos/SPARZ_fig1/{ROI}/{codec}/data/*bp1*.{extension}',
               stem=f'{codec}_fig1',
               output_path=f'/Users/dimos/SPARZ_fig1/{ROI}/{codec}/images',
               #path_sparse_bp2=f'/mnt/altnas/work/dimos/SPARZ_fig1/with_ROI/{codec}/data/*bp2*.npz',
               path_sparse_bp2 = sparse_bp2,
               # path_encoded_bp2=f'/Users/dimos/SPARZ_fig1/{ROI}/{codec}/data/*+250_*.{extension}',
               path_encoded_bp2=f'/mnt/altnas/work/dimos/SPARZ_fig1/{ROI}/{codec}/data/*bp2*.{extension}',
               use_roi=sparse
               )
            print("Data loaded successfully.")
            return u
        except Exception as e:
            print(f"Data loading failed: {e}")
        

#    u=SPARUNZIP(#path_sparse_bp1=f'/mnt/altnas/work/dimos/SPARZ_fig1/with_ROI/{codec}/data/*bp1*.npz',
#                path_sparse_bp1 = sparse_bp1,
#                path_encoded_bp1=f'/mnt/altnas/work/dimos/SPARZ_tubulin/{ROI}/{codec}/data/*-250_*.{extension}',
#                stem=f'{codec}_fig1',
#                output_path=f'/mnt/altnas/work/dimos/SPARZ_tubulin/{ROI}/{codec}/images',
#                #path_sparse_bp2=f'/mnt/altnas/work/dimos/SPARZ_fig1/with_ROI/{codec}/data/*bp2*.npz',
#                path_sparse_bp2 = sparse_bp2,
#                path_encoded_bp2=f'/mnt/altnas/work/dimos/SPARZ_tubulin/{ROI}/{codec}/data/*+250_*.{extension}',
#                use_roi=sparse,
#                max_workers=4,
#                chunk_size=100)

   def decompress_images(u):
      try:
         u.run()
         print(f"Decompression using {codec} completed successfully.")
      except Exception as e:
         print(f"Compression failed: {e}")


   times = []
   overhead_times = []

   for _ in range(tries):
      start_time = time.perf_counter()
      u=decomp_overhead() 
      end_time = time.perf_counter()
      overhead_times.append(end_time - start_time)
      start_time = time.perf_counter()
      decompress_images(u)
      end_time = time.perf_counter()
      times.append(end_time - start_time)

   overhead_mean_time = statistics.mean(overhead_times)
   overhead_std_dev_time = statistics.stdev(overhead_times)

   mean_time = statistics.mean(times)
   std_dev_time = statistics.stdev(times)

#    print (f"{codec} Overhead Times: {overhead_times}")
#    print(f"{codec} Overhead Mean: {overhead_mean_time}")
#    print(f"{codec} Overhead Standard Deviation: {overhead_std_dev_time}")
   
#    print(f"{codec} Decompress Times: {times}")
#    print(f"{codec} Mean:, {mean_time}")
#    print(f"{codec} Standard Deviation:, {std_dev_time}")

   # codec_times['decompression'] = {'codec':codec,
   #      'overhead': overhead_times,
   #      'overhead_mean': mean_time,
   #      'overhead_std_dev': std_dev_time,
   #      'times': times,
   #      'mean': mean_time,
   #      'std_dev': std_dev_time
   #  }
   codec_times['decompression_overhead_mean']=overhead_mean_time
   codec_times['decompression_overhead_std_dev']=overhead_std_dev_time
   codec_times['decompression_mean']=mean_time
   codec_times['decompression_std_dev']=std_dev_time

   all_the_times.append(codec_times)
   ##execution_time = timeit.repeat(decompress_images, number=10)
   ##print(f"Decompression {codec} execution time: {execution_time} seconds")
# %%
all_the_compressions = pd.DataFrame(all_the_times)
# all_the_decompressions = pd.DataFrame([x['decompression']for x in all_the_times])

# all_the_compressions.to_csv(f'/Users/dimos/SPARZ_fig1/{ROI}/run_times.csv')
all_the_compressions.to_csv(f'/mnt/altnas/work/dimos/SPARZ_fig1/{ROI}/run_times.csv')
# all_the_decompressions.to_csv(f'/Users/dimos/SPARZ_fig1/{ROI}/decompression_times.csv')
# %%
# all_the_compressions
# %%
# import glob
# glob.glob(f'/Users/dimos/SPARZ_fig1/{ROI}/{codec}/data/*-250_*.{extension}')
# %%
s