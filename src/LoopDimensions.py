import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from tqdm.contrib.itertools import product
from skimage import measure

def loop_tileregion_np(img: list, func, tr_init: list, channels_oi: list, sizes: dict, volumetric: bool = False,  **kwargs):
    '''
    Loop over all dimensions of a dataset post stitching.
    Suggested task: segmenation, 1 or 2 channels, no Z-awareness currently.
    Output n-dim numpy array per tileregion. 1 result per position/time/slice.

    Input
        img [list of np]: stitched image dataset. format: [TR][T,Z,C,Y,X]
        func: function to run on individual images, like segmentation.
        tr_init [list]: starting positions of tile regions.
        channels_oi [list]: on which channel to perform operation.
            if 2 channels supplied, assumes ['cell', 'nucleus'].
        sizes [dict]: nd2 metadata collection dimensions.
        volumetric [bool]: whether to perform volumetric z-slice aware processing, instead of per slice.
        **kwargs: arguments supplied to func.
    Return
        result of function, appended list. [TR][T,Z,Y,X]
    '''
    results = []

    # Loop over tile regions
    for tr in tqdm(range(len(tr_init)),
                   total = len(tr_init),
                   desc='stitching tiles'):
        
        # Loop all but XYC axis
        img_tr = img[tr]
        img_tr_shape = img_tr.shape

        result = np.zeros(img_tr_shape[:2] + img_tr_shape[-2:], dtype = np.uint16)
        if sizes.get('Z', 1) == 1 and not volumetric:
            for t, z in tqdm(np.ndindex(*img_tr_shape[:2]),
                             total=np.prod(img_tr_shape[:2])):
    #           print(indices + tuple(channels_oi))
                if len(channels_oi) <= 2:
                    img_current = img_tr[t, z, channels_oi].squeeze()
                else:
                    print("Too many channels supplied")
                # function execution
                result[t, z] = func(img = img_current, **kwargs)
        else:
            # 3D segmentation?
            print('Z-slice aware processing not yet build')
        results.append(result)
    return results

def loop_tileregion_pd(img: list, func, tr_init: list, channels_oi: list, switch_channels: bool, **kwargs):
    '''
    Loop over all dimensions of a dataset post stitching.
    Suggested task: feature extraction or spot detection.
    Output pandas with axis indeces per tileregion.

    Input
        img [list of np]: stitched image dataset. format: [TR][T,Z,C,Y,X]
        func: function to run on individual images, like segmentation.
        tr_init [list]: starting positions of tile regions.
        channels_oi [list]: on which channel to perform operation.
        sizes [dict]: nd2 metadata collection dimensions.
        switch_channels [bool]: whether to transpose channel axis to last position, typical for skimage::regionprops.
        **kwargs: arguments supplied to func.
    Return
        result of function, appended list.
    '''
    results = []

    # Loop over tile regions
    for tr in tqdm(range(len(tr_init))):
        # Loop all but XYC axis
        img_tr = img[tr]
        img_tr_shape = img_tr.shape
        result = []
        for t, z in tqdm(np.ndindex(*img_tr_shape[:2])):
            img_current = img_tr[t, z, channels_oi].squeeze()
            if len(channels_oi) > 1 and switch_channels:
                # transpose channel to last axis?
                img_current = np.transpose(img_current, axes = (1, 2, 0))
            # function execution
            result_temp = func(img = img_current, **kwargs)
            if isinstance(result_temp, dict):
                result_temp = pd.DataFrame(result_temp)
            result_temp[['T', 'Z']] = t, z
            result.append(result_temp)
        results.append(result)
    results = pd.concat(results)
    return results

def loop_tiles_pd(data_img, sizes, func, model,
                  channels_oi: list, binning: int = 1,
                  dict_flatfield: dict = {'flatfield': False}, **kwargs):
    '''
    Loop over all images in a multidimensional numpy/dask array.

    kwargs could contain a flatfield_package in a dictionary, which includes:
    - flatfield: bool
    - flatfield_func
    - img_ff
    - img_df
    '''

    results = []
    for indices in product(*map(range, data_img.blocks.shape)):
        # 1 image, could be multichannel [(C), Y, X]
        chunk = data_img.blocks[indices].compute().squeeze()

        # multichannel image
        for i_c in np.where(channels_oi)[0]:
            if chunk.ndim == 3:
                img = chunk[i_c]
                
                # store dimensional indexes (excluding YX)
                ind = np.concatenate((indices[:-3], [channels_oi[i_c]]))
                ind_names = list(sizes.keys())[:-2]
            elif chunk.ndim == 2:
                img = chunk

                # store dimensional indexes (excluding YX)
                ind = np.concatenate((indices[:-2], [channels_oi[i_c]]))
                if 'C' in sizes:
                    ind_names = list(sizes.keys())[:-2]
                else:
                    ind_names = np.concatenate((sizes.keys()[:-2], ['C']))
            else:
                raise ValueError('Wrong chunk dimensions')

            # flatfield
            if dict_flatfield['flatfield']:
                flatfield_func = dict_flatfield['flatfield_func']
                img = flatfield_func(img,
                                     dict_flatfield['img_ff'][i_c].squeeze(),
                                     dict_flatfield['img_df'])

            # binning
            if binning > 1:
                img = measure.block_reduce(img, block_size = binning)

            # run the function
            result = func(img = img, model = model)
            # append columns image indexes
            result[ind_names] = ind

            results.append(result)

    # bind chunk data to a single dataframe
    results = pd.concat(results)

    if binning > 1:
        results[['Y', 'X']] = results[['Y', 'X']] * binning

    return results
