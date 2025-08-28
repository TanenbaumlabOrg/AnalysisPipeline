import time, os
import re
import numpy as np
from tqdm.notebook import tqdm
from tifffile import imread, imwrite
from cellpose import models

from cellpose import version as cellpose_version
if int(cellpose_version[0]) >= 3:
    from cellpose import denoise

def load_model_cellpose(name: str = 'nuclei', external: bool = False, denoise_model: bool = False):
    '''
    Load cellpose model, either internal (specify name) or external (specify path).
    Input
        name [str]: name of internal model, or path of external model.
        external [bool]: whether to use external model.
    Return
        loaded cellpose model
    '''

    if external:
        if os.path.exists(name):
            if denoise_model:
                cellpose_model = denoise.CellposeDenoiseModel(
                    gpu = True,
                    pretrained_model = name,
                    restore_type = "denoise_nuclei")
            else:
                if int(cellpose_version[0]) == 2:
                    model = models.CellposeModel(pretrained_model = name, net_avg = False, gpu=True)
                if int(cellpose_version[0]) == 3:
                    model = models.CellposeModel(pretrained_model = name, gpu=True)
        else:
            print("Model path not valid")
            cellpose_model = ''
    else:
        if name == 'nuclei_denoise':
            cellpose_model = denoise.CellposeDenoiseModel(
                gpu = True,
                model_type = "nuclei",
                restore_type = "denoise_nuclei")
        elif name == 'cyto3_denoise':
            cellpose_model = denoise.CellposeDenoiseModel(
                gpu = True,
                model_type = "cyto3",
                restore_type = "denoise_cyto3")
        else:
            cellpose_model = models.Cellpose(model_type = name, gpu=True)
        
    return cellpose_model

def create_masks(path_img, path_mask, diam = 80, used_model = 'nuclei', swap_channels = False, resample = False):

    """
    Nuclear masks are generated from segmentation channel as a numpy array.
    
    path_img = input folder.
    path_mask = mask output 
    folder.
    diam = Expected object diameter can be adjusted.
    swap_channels = true if nuclear channel is last in stack.
    resample = can be set for smoother mask edges.

    """
    # ! MAKE EAT BOTH TIF AND TIFF

    for file in tqdm([ f for f in os.listdir(path_img) if (str(f))[-3:] == "tif"]):
        helper = []
        try:
            img = imread(path_img + file)
            print(f" loading {file}")
            if swap_channels == True:
                img_flipped = np.flip(img, axis=1)
            else:
                img_flipped = img
            model = models.Cellpose(model_type = used_model)
            lis_img = [np.squeeze(img_flipped[i, 0]) for i in range(len(img_flipped))]

            channels = [0,0]

            for i in tqdm(lis_img):
                start = time.time()
                masks, flows, styles, diams = model.eval(i, diameter = diam, channels = channels, resample = resample)
                finish = time.time()
                print(f"nuclei for 2D image predicted in {finish-start} seconds")
                helper.append(masks[np.newaxis, :, :])
            maskss = np.concatenate(helper, axis=0)
            imwrite(path_mask + file.split(".")[0] + "_mask.tif", np.asarray(maskss))
            del img
            
        except ValueError:
            print(f"encountered value error in file {file}. Perhaps found no masks")
            continue

def create_mask_timeseries(img, filename, path_mask, diam = 80, model_type = 'included', model_name = 'nuclei', resample = False):

    """
    Nuclear masks are generated from segmentation channel.
    
    img = image data as 3d array TYX (?lazy)
    path_mask = mask output folder
    model_type = included or custom model
    model_name = existing or custom model name (?path)
    diam = Expected object diameter can be adjusted.
    swap_channels = true if nuclear channel is last in stack.
    resample = can be set for smoother mask edges.
    """
    mask_array = []
    if model_type == 'included':
        model = models.Cellpose(model_type = model_name)
    else:
        if int(cellpose_version[0]) == 2:
            model = models.CellposeModel(pretrained_model = model_name, net_avg = False)
        if int(cellpose_version[0]) == 3:
            model = models.CellposeModel(pretrained_model = model_name)

    if str(type(img)) == "<class 'resource_backed_dask_array.ResourceBackedDaskArray'>":
        img = img.compute()
    
    lis_img = [np.squeeze(img[i, :, :]) for i in range(len(img))]
    channels = [0,0]
    for i in tqdm(lis_img):
        start = time.time()
        if model_type == 'included':
            masks, flows, styles, diams = model.eval(i, diameter = diam, channels = channels, resample = resample)
        else:
            masks, flows, styles = model.eval(i, diameter = diam, channels = channels, resample = resample)
        finish = time.time()
        mask_array.append(masks[np.newaxis, :, :])
    maskss = np.concatenate(mask_array, axis=0)
    imwrite(path_mask + filename + "_mask.tif", np.asarray(maskss))
    del lis_img
    del mask_array

def create_mask_timeseries_cell(img, filename, path_mask, mask_suffix = "_mask_cell", channels = [1,2], diam = 80, model_type = 'included', model_name = 'cyto', resample = False):
    """
    Nuclear masks are generated from segmentation channel.
    
    img = image data as 3d array TYX (?lazy)
    path_mask = mask output folder
    model_type = included or custom model
    model_name = existing or custom model name (?path)
    diam = Expected object diameter can be adjusted.
    swap_channels = true if nuclear channel is last in stack.
    resample = can be set for smoother mask edges.
    """
    mask_array = []
    if model_type == 'included':
        model = models.Cellpose(model_type = model_name)
    else:
        if int(cellpose_version[0]) == 2:
            model = models.CellposeModel(pretrained_model = model_name, net_avg = False)
        if int(cellpose_version[0]) == 3:
            model = models.CellposeModel(pretrained_model = model_name)

    if str(type(img)) == "<class 'resource_backed_dask_array.ResourceBackedDaskArray'>":
        img = img.compute()
    
    lis_img = [np.squeeze(img[i, :, :, :]) for i in range(len(img))]
    for i in tqdm(lis_img):
        start = time.time()
        if model_type == 'included':
            masks, flows, styles, diams = model.eval(i, diameter = diam, channels = channels, resample = resample)
        else:
            if int(cellpose_version[0]) == 3:
                masks, flows, styles, diams = model.eval(i, diameter = diam, channels = channels, resample = resample)
            else:
                masks, flows, styles = model.eval(i, diameter = diam, channels = channels, resample = resample)
        finish = time.time()
        mask_array.append(masks[np.newaxis, :, :])
    maskss = np.concatenate(mask_array, axis=0)
    imwrite(path_mask + filename + mask_suffix + ".tif", np.asarray(maskss))
    del lis_img
    del mask_array

def create_mask(img, cellpose_model, diam = 80, resample = True, channels = [0,0]):
    """
    Nuclear masks are generated from segmentation channel.
    
    Inputs:
    - img = image data as 2d array YX (?lazy)
    - cellpose_model = loaded model to be used
    - diam = estimated average object diameter

    Return:
    - numpy array YX, each object unique intensity
    """
    
    if str(type(img)) == "<class 'resource_backed_dask_array.ResourceBackedDaskArray'>":
        img = img.compute()
    
    pattern = r"^(<)cellpose\.models\.Cellpose(?!Model)"
    if re.match(pattern, str(cellpose_model)):
        masks, flows, styles, diams = cellpose_model.eval(img, diameter = diam, channels = channels, resample = resample)
    else:
        if int(cellpose_version[0]) == 3:
            masks, flows, styles, diams = cellpose_model.eval(img, diameter = diam, channels = channels, resample = resample)
        else:
            masks, flows, styles = cellpose_model.eval(img, diameter = diam, channels = channels, resample = resample)
    
    return masks
