import pandas as pd
import numpy as np
from skimage import measure
import scipy
import scipy.stats
import mahotas
from functools import partial

def img_int_std(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.std(intensities[region], ddof=1)

def img_int_mean(region, intensities):
    return np.mean(intensities[region])

def img_int_median(region, intensities):
    return np.median(intensities[region])

def img_int_max(region, intensities):
    return np.max(intensities[region])

def img_int_min(region, intensities):
    return np.min(intensities[region])

def img_int_sum(region, intensities):
    return np.sum(intensities[region])

def img_int_var(region, intensities):
    return np.var(intensities[region])

def img_int_quantiles_low(region, intensities):
    return np.percentile(intensities[region], q = (1, 5, 10, 15, 20))

def img_int_skewness(region, intensities):
    return scipy.stats.skew(intensities[region])

def img_int_kurtosis(region, intensities):
    """
    Calculate the kurtosis of the intensity values within a given region.

    Kurtosis is a measure of the 'tailed-ness' of a distribution. A high kurtosis
    indicates that most of the data points are concentrated near the mean,
    while a low kurtosis suggests a more uniform distribution.

    Parameters:
        - region (numpy array): The region of interest within which to calculate the kurtosis.
        - intensities (numpy array): The intensity values for the image.

    Returns:
        The kurtosis value as a float.
    """
    return scipy.stats.kurtosis(intensities[region])

def img_int_mode_non_zero(region, intensities):
    values = intensities[region]
    values_nonzero = [x for x in values if not x == 0]
    if len(values_nonzero):
        value_mode =  int(scipy.stats.mode(values_nonzero)[0])
    else:
        value_mode = 0
    return value_mode

def img_int_haralick_scale(region, intensities, scales = [1], return_direction_mean = False):
    '''
    Computes Haralick features in the object mask on image intensity values.
    Returns a np.array of shape 4, 13 for each object, for 2d data.
    Assumes all pixel intensities > 0.
    '''
    img = region * intensities
    img_shape = img.shape

    feats = []
    if img_shape[0] > 3 and img_shape[1] > 3 and np.min(intensities[region]) > 0:
        # at least a reasonable image size and intensity above zero
        for scale in scales:
            feats.extend(mahotas.features.haralick(img.astype('uint16'), ignore_zeros = True, return_mean = return_direction_mean, distance = scale))
    else:
        # print('too small input')
        feats = None
    return feats

def extract_bbox_features(mask):
    props = measure.regionprops_table(mask, properties = ['label', 'bbox'])
    current_df = pd.DataFrame(props)
    return(current_df)

def extract_intensity_features_img(img, mask):
    props = measure.regionprops_table(mask, intensity_image = img,
                                      properties = ['label', 'area', 'solidity', 'intensity_min', 'intensity_max', 'intensity_mean'],
                                      extra_properties = [img_int_std, img_int_sum, img_int_median, img_int_quantiles_low]) #, img_int_min, img_int_max, img_int_mean
    current_df = pd.DataFrame(props)
    return(current_df)

def extract_intensity_features_img_minimal(img, mask):
    props = measure.regionprops_table(mask, intensity_image = img,
                                      properties = ['label', 'area', 'intensity_mean'])
    current_df = pd.DataFrame(props)
    return(current_df)

def extract_intensity_features_img_mean(img, mask):
    props = measure.regionprops_table(mask, intensity_image = img,
                                      properties = ['label', 'intensity_mean'])
    current_df = pd.DataFrame(props)
    return(current_df)

def extract_intensity_features_img_median(img, mask):
    props = measure.regionprops_table(mask, intensity_image = img,
                                      properties = ['label'],
                                      extra_properties = [img_int_median])
    current_df = pd.DataFrame(props)
    return(current_df)

def extract_intensity_features_img_texture(img, mask):
    props = measure.regionprops_table(mask, intensity_image = img,
                                      properties = ['label', 'inertia_tensor', 'inertia_tensor_eigvals', 'intensity_max', 'intensity_mean',
                                                    'intensity_min', 'moments', 'moments_central', 'moments_hu', 'moments_normalized',
                                                    'moments_weighted', 'moments_weighted_central', 'moments_weighted_hu', 'moments_weighted_normalized'],
                                      extra_properties = [img_int_var, img_int_skewness, img_int_kurtosis])
    current_df = pd.DataFrame(props)
    return(current_df)

def extract_intensity_mode_non_zero(img, mask):
    props = measure.regionprops_table(mask, intensity_image = img,
                                      properties = ['label'],
                                      extra_properties = [img_int_mode_non_zero])
    current_df = pd.DataFrame(props)
    return(current_df)

def extract_shape_intensity_features(img, mask):
    props = measure.regionprops_table(mask, intensity_image = img,
                                      properties = ['label', 'centroid', 'area', 'eccentricity', 'equivalent_diameter',
                                                    'major_axis_length', 'minor_axis_length', 'orientation',
                                                    'intensity_mean', 'weighted_centroid'])
    current_df = pd.DataFrame(props)
    return(current_df)

def extract_intensity_features_img_haralick(img, mask, scales = [8, 10], return_direction_mean = True):
    img_int_haralick_scale_multi = partial(img_int_haralick_scale,
        scales = scales, return_direction_mean = return_direction_mean)
    img_int_haralick_scale_multi.__name__ = 'img_int_haralick'

    props = measure.regionprops_table(
        label_image = mask, intensity_image = img,
        properties = ['label'], extra_properties=[img_int_haralick_scale_multi])

    name_features_haralick = [
        'AngularSecondMoment', 'Contrast', 'Correlation', 'Variance',
        'InverseDifferenceMoment', 'SumAverage', 'SumVariance', 'SumEntropy', 'Entropy',
        'DifferenceVariance', 'DifferenceEntropy', 'InfoMeas1', 'InfoMeas2']

    if return_direction_mean:
        colnames_haralick = ['Haralick_' + name + '_s' + str(scale) for scale in scales for name in name_features_haralick]
        df_textures = pd.DataFrame.from_records(props['img_int_haralick'], columns = colnames_haralick)
    else:
        colnames_haralick = ['Haralick_' + name + '_s' + str(scale) + '_d' + str(direction) for scale in scales for direction in range(4) for name in name_features_haralick]
        df_textures = pd.DataFrame([np.concatenate(x) for x in props['img_int_haralick']], columns = colnames_haralick)

    return df_textures
