import numbers
from tifffile import natural_sorted
import warnings 
import pims   
import numpy as np
import glob
import dask.array as da

def imread(fname, nframes=1, *, arraytype="numpy",dtypes='uint16'):

    """
    This is just the imread function from dask_image.imread with a few modifications to facilitate lazy loading mpeg videos.
    
    First, it is modified to take dtypes as an argument. This is because the default dtype for mpeg videos is uint8, which is not appropriate for our data.
    
    Second, the function read_frames is modified to use the pims.ImageIOReader to read mpeg videos with dtype specified by the user.
    
    Finally, one of the main reason for using dask is to process each frame on the fly without loading the entire video into memory.
    
    Original code can be found here: https://image.dask.org/en/latest/_modules/dask_image/imread.html#imread
    """

    sfname = str(fname)
    if not isinstance(nframes, numbers.Integral):
        raise ValueError("`nframes` must be an integer.")
    if (nframes != -1) and not (nframes > 0):
        raise ValueError("`nframes` must be greater than zero.")

    if arraytype == "numpy":
        arrayfunc = np.asanyarray
    elif arraytype == "cupy":   # pragma: no cover
        import cupy
        arrayfunc = cupy.asanyarray

    with pims.open(sfname) as imgs:
        shape = (len(imgs),) + imgs.frame_shape
        # dtype = np.dtype(imgs.pixel_type)

    if nframes == -1:
        nframes = shape[0]

    if nframes > shape[0]:
        warnings.warn(
            "`nframes` larger than number of frames in file."
            " Will truncate to number of frames in file.",
            RuntimeWarning
        )
    elif shape[0] % nframes != 0:
        warnings.warn(
            "`nframes` does not nicely divide number of frames in file."
            " Last chunk will contain the remainder.",
            RuntimeWarning
        )

    # place source filenames into dask array after sorting
    filenames = natural_sorted(glob.glob(sfname))
    if len(filenames) > 1:
        ar = da.from_array(filenames, chunks=(nframes,))
        multiple_files = True
    else:
        ar = da.from_array(filenames * shape[0], chunks=(nframes,))
        multiple_files = False

    # read in data using encoded filenames
    a = ar.map_blocks(
        _map_read_frame,
        chunks=da.core.normalize_chunks(
            (nframes,) + shape[1:], shape),
        multiple_files=multiple_files,
        new_axis=list(range(1, len(shape))),
        arrayfunc=arrayfunc,
        meta=arrayfunc([]).astype(dtypes),  # meta overwrites `dtype` argument
    )
    return a if len(a.shape)==3 else a[:,:,:,0]

def _map_read_frame(x, multiple_files, block_info=None, **kwargs):

    fn = x[0]  # get filename from input chunk

    if multiple_files:
        i, j = 0, 1
    else:
        i, j = block_info[None]['array-location'][0]

    return _read_frame(fn=fn, i=slice(i, j), **kwargs)


def _read_frame(fn, i, *, dtypes='uint16', arrayfunc=np.asanyarray):
    with pims.ImageIOReader(fn, dtype=dtypes) as imgs:
        return arrayfunc(imgs[i])