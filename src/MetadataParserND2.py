import os
import sys
import nd2
import re
import numpy as np
import pandas as pd

sys.path.insert(1, os.path.dirname(os.getcwd()))
from get_nested_attribute import get_nested_attribute
import Parser

def get_nd2_meta(path_nd2: str):
    '''
    Input:
        path_nd2: location of nd2 file to parse

    Return:
        sizes [dict]
        channels [list]
        positions [np]
        px_size [float]
    '''
    positions = []
    with (nd2.ND2File(path_nd2)) as data_img:
        # nd2 dimensions and sizes
        sizes = data_img.sizes
        
        # channel info
        channels = get_channels(data_img.metadata.channels, sizes)

        # parse positional metadata per image
        meta_frame = data_img.frame_metadata
        image_number = get_image_number(sizes)
        for i in range(image_number):
            meta_channels = getattr(meta_frame(i), 'channels')
            meta_stageposition = get_nested_attribute(meta_channels[0], 'position/stagePositionUm')
            positions.append(meta_stageposition)
        positions = pd.DataFrame(positions, columns = ["X", "Y", "Z"])
        positions[['iT', 'iP', 'iZ']] = [[t, p, z] for t in range(sizes.get('T', 1)) for p in range(sizes.get('P', 1)) for z in range(sizes.get('Z', 1))]

        # pixel size
        px_size = get_nested_attribute(getattr(meta_frame(0), 'channels')[0], 'volume/axesCalibration')[0]

        data_img.close()
    return sizes, channels, positions, px_size

def get_nd2_sizes(path_nd2: str):
    '''
    Input:
        path_nd2: location of nd2 file to parse

    Return:
        sizes [dict]
     '''
    with (nd2.ND2File(path_nd2)) as data_img:
         # nd2 dimensions and sizes
        sizes  = data_img.sizes
        data_img.close()
    return sizes

def get_nd2_sizes_dask(path):
    with (nd2.ND2File(path)) as data_img:
        sizes = data_img.sizes
        data_img.close()
    data_img = data_img.to_dask()
    return sizes, data_img

def get_nd2_channels(path_nd2: str):
    '''
    Input:
        path_nd2: location of nd2 file to parse
    Return:
        channels [list]
    '''
    with (nd2.ND2File(path_nd2)) as data_img:
        # channel info
        channels = get_channels(data_img.metadata.channels, data_img.sizes)
        data_img.close()
    return channels

def get_channels(channels: list, sizes: dict):
    list_channels = []
    if 'C' in sizes:
        for c in range(sizes['C']):
            list_channels.append(get_nested_attribute(channels[c], 'channel/name'))
    else:
        list_channels.append(get_nested_attribute(channels[0], 'channel/name'))
    return list_channels

def get_image_number(sizes: dict):
    result = 1  # Initialize the result to 1
    for key, value in sizes.items():
        if key not in ['X', 'Y', 'C']:
            result *= value
    return result

def get_nd2_pxsize(path_nd2: str):
    '''
    Input:
        path_nd2: location of nd2 file to parse

    Return:
        px_size [float]
    '''
    with (nd2.ND2File(path_nd2)) as data_img:
        # parse positional metadata per image
        meta_frame = data_img.frame_metadata
        px_size = get_nested_attribute(getattr(meta_frame(0), 'channels')[0], 'volume/axesCalibration')[0]
        data_img.close()
    return px_size

def get_nd2_misc(path_nd2: str):
    dict_misc = {}
    with (nd2.ND2File(path_nd2)) as data_img:
        # objective info
        dict_misc['objective'] = get_nested_attribute(
            get_nested_attribute(data_img.metadata, 'channels')[0], 'microscope/objectiveName')
        # exposure time
        dict_misc['exposure'] = Parser.parse_metadata_string(data_img.text_info['capturing'])['Exposure']

        # date of acquisition
        julianTime = get_nested_attribute(
            get_nested_attribute(data_img.frame_metadata(0), 'channels')[0], 'time/absoluteJulianDayNumber')
        gregorian_date = Parser.julian_day_to_gregorian(julianTime)
        dict_misc['date'] = f"{gregorian_date[0]}-{gregorian_date[1]:02}-{gregorian_date[2]:02}"

        data_img.close()
    return dict_misc

def get_nd2_time(path_nd2: str):
    '''
    Input:
        path_nd2: location of nd2 file to parse

    Return:
        times [np]
    '''
    times = []
    with (nd2.ND2File(path_nd2)) as data_img:
        # nd2 dimensions and sizes
        sizes = data_img.sizes
        
        # parse positional metadata per image
        image_number = get_image_number(sizes)
        for i in range(image_number):
            time = get_nested_attribute(get_nested_attribute(data_img.frame_metadata(i), 'channels')[0], 'time/relativeTimeMs')
            times.append(time)
        times = pd.DataFrame(times, columns = ["time"])
        times[['iT', 'iP', 'iZ']] = [[t, p, z] for t in range(sizes.get('T', 1)) for p in range(sizes.get('P', 1)) for z in range(sizes.get('Z', 1))]

        data_img.close
    return times

def fix_nd2_single_channel(img, sizes: dict):
    '''
    For single channel data, add a dummy channel axis, and add C to sizes dict.
    
    Input
    img: nD numpy array of image data
    sizes: dict of image size

    Retuns modified image and sizes
    '''

    sizes_list = list(sizes.items())
    sizes_list.insert(-2, ('C', 1))
    sizes = dict(sizes_list)
    print(sizes)

    img = np.expand_dims(img, axis = -3)
    print(img.shape)

    return img, sizes

def proj_nd2_max(img, sizes: dict):
    '''
    Input
    img: nD numpy array of image data
    sizes: dict of image size

    Retuns modified image and sizes
    '''
    
    # max projection along Z
    axis_index_z = list(dict.keys(sizes)).index('Z')
    img = np.max(img, axis = axis_index_z)
    print(img.shape)

    # drop Z axis from sizes dict
    sizes.pop('Z', None)
    print(sizes)

    return img, sizes

def get_nd2_camera_crop(path_nd2: str):
    with nd2.ND2File(path_nd2) as nd2file:
        nd2meta = nd2file.unstructured_metadata()
        nd2file.close()
    crop_cam = nd2meta['ImageMetadataSeqLV|0']['SLxPictureMetadata']['PicturePlanes']['SampleSetting']['0']['CameraSetting']['FormatQuality']['SensorUser']['']

    # with nd2.ND2File(path_nd2) as nd2file:
    #     nd2meta = nd2file.unstructured_metadata()['ImageMetadataSeqLV|0']['SLxPictureMetadata']['PicturePlanes']['SampleSetting']
    #     nd2file.close()
    # max_key = max(int(key) for key in nd2meta.keys())
    # crop_cam = nd2meta[str(max_key)]['CameraSetting']['ROI']
    return crop_cam

def get_nd2_camera_settings(path_nd2: str):
    '''
    Input:
        path_nd2: location of nd2 file to parse

    Return:
        Dictonary of camera settings
    '''

    with (nd2.ND2File(path_nd2)) as data_img:
        info = data_img.text_info
        meta_unstructured = data_img.unstructured_metadata()
        data_img.close()

    info_capture = info['capturing'].split('\r\n')
    info_descr = info['description'].split('\r\n')

    camera = find_regex_list(info_descr, 'Camera Name: ')

    binning = find_regex_list(info_capture, 'Binning:', r'(\d+)x(\d+)')

    exposure = find_regex_list(info_capture, 'Exposure:', r'(\d+)( )(.+)')

    objective = [info['optics']]

    camera_mode = [meta_unstructured['ImageMetadataSeqLV|0']['SLxPictureMetadata']['PicturePlanes']['SampleSetting']['0']['CameraSetting']['FormatQuality']['Desc']['FormatText']]

    # image denoising
    denoise = find_regex_list(info_capture, 'Denoise.ai', '(OFF|ON)')
    if denoise:
        for i, status in enumerate(denoise):
            if status == 'OFF':
                denoise[i] = False
            else:
                denoise[i] = True

    info_dict = {
            'camera': camera,
            'binning': binning,
            'mode': camera_mode,
            'exposure': exposure,
            'objective': objective,
            'denoise': denoise}

    return info_dict

def get_nd2_tile_names(path_nd2: str):
    with (nd2.ND2File(path_nd2)) as d:
        exp = d.experiment
        d.close()

    list_positions = get_nested_attribute(exp[-1], 'parameters/points')

    names = []
    for i in list_positions:
        names.append(getattr(i, 'name'))
    return names

def find_regex_list(list_things, pattern, pattern_sub = None):
    list_matches = []
    regex_pattern =  re.compile(r'^( )*' + pattern)

    for x in list_things:
        if regex_pattern.match(x):
            if pattern_sub is not None:
                match = re.search(pattern_sub, x)
                if match is not None:
                    list_matches.append(match.group())
                else:
                    print('bad match')
            else:
                list_matches.append(x[len(pattern):])
    return list_matches
