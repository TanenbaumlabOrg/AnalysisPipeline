import numpy as np

def ffdf_series(img, ff, df):
    # all input is numpy arrays
    # first dimension = time
    img_ff = np.zeros_like(img, dtype=np.uint16)
    if img.ndim == 3:
        for t in range(img.shape[0]):
            img_ff[t] = ffdf(img[t], ff, df, format_out = np.uint16)
    else:
        print("input, not a series!")
        img_ff = np.zeros((2,1))
    return img_ff

def ffdf(img, ff, df, format_out = np.uint16, verbose: bool = False):
    '''
    Perform flatfield and darkfield illumination correction on a single 16-bit image.
    ! Input images must be of same shape.

    Input
        img [np]: Y,X numpy array of raw image data, input cannot be dask array!
        ff [np]: Y,X numpy array of flatfield image
        df [np]: Y,X numpy array of darkfield image
    Return
        img_ff [np]: Y,X numpy array of illumation corrected data
    '''

    # test for equal shape of input images
    if img.shape == ff.shape == df.shape:
        img_df = apply_darkfield(img, df)
        img_ff = apply_flatfield(img_df, ff)

        if format_out == np.uint16:
            # set negative values to 0
            img_ff = np.nan_to_num(img_ff)
            img_ff = np.clip(img_ff, 0, 65535)
            img_ff = np.round(img_ff).astype(np.uint16)
    else:
        print("img, ff, df diverging shapes")
        img_ff = np.zeros((2,1))

    if verbose:
        print(img.shape, ff.shape, df.shape)
    return img_ff

def apply_flatfield(img, ff):
    '''
    Divide img by ff
    '''
    img_out = np.divide(img, ff)
    return img_out

def apply_darkfield(img, df):
    '''
    Substract df from img
    '''
    # subtract into 32bit to allow negative values
    img_out = np.subtract(img, df, dtype = np.int32)
    # clip negative values to 0
    img_out = np.clip(img_out, 0, 65535)
    # set data type back to uint16
    img_out = img_out.astype(np.uint16)
    return img_out

def calculate_ffdf(img_ff, img_df):
    '''
    img_ff [np]: stack of images stacked on axis 0. ndim = 3
    img_df [np]: stack of images stacked on axis 0. ndim = 3
    
    return single image
    '''

    img_ff_med = np.median(img_ff, axis = 0)
    img_df_med = np.median(img_df, axis = 0)
    ff = img_ff_med-img_df_med
    return ff, img_df_med

def stack_project_median(img, **kwargs):
    # Determine number of dimensions
    num_dims = img.ndim

    if num_dims < 2:
        raise ValueError("Array must have at least 2 dimensions.")

    # Compute median along all axes except the last two
    axes_to_reduce = tuple(range(num_dims - 2))
    return np.median(img, axis = axes_to_reduce)

def stack_project_median_uint16(img):
    # Determine number of dimensions
    num_dims = img.ndim

    if num_dims < 2:
        raise ValueError("Array must have at least 2 dimensions.")

    # Compute median along all axes except the last two
    axes_to_reduce = tuple(range(num_dims - 2))

    return np.round(np.median(img, axis = axes_to_reduce)).astype(np.uint16)

def stack_calculate_median(img, dtype = None):
    array = np.median(img, tuple(range(img.ndim))[-2:])

    if dtype == 'uint16':
        array = np.round(array).astype(np.uint16)
    return array

def combine_axes(arr):
    """
    Combines all axes except for the last two into a single axis.
    For example:
      - A (5, 4, 3) array → (20, 3)
      - A (6, 5, 4, 3) array → (30, 4, 3)
      - A (7, 6, 5, 4, 3) array → (210, 4, 3)
    """
    num_dims = arr.ndim
    
    if num_dims < 3:
        raise ValueError("Array must have at least 3 dimensions to combine leading axes.")

    # Compute the new first dimension size by multiplying all but the last two dims
    new_first_dim = np.prod(arr.shape[:-2])
    
    # Reshape while keeping the last two dimensions intact
    return arr.reshape(new_first_dim, *arr.shape[-2:])

def stack_outlier_reject(img):
    '''
    A. Reject image that differ in median intensity by >2x SC.
    B. Check if min intensity (excluding not illuminated area) rises above noise floor.
    '''
    img_slice_medians = np.median(img, [-2, -1])

def stack_normalize(img):
    '''
    Subtract darkfield
    Normalize stack
    '''
    img_slice_medians = np.median(img, [-2, -1])

