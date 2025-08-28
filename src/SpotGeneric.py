import numpy as np
import pandas as pd
from math import floor, ceil
from skimage.feature import blob_log
from skimage.draw import disk, rectangle
from scipy.spatial.distance import cdist
from ray.util.multiprocessing import Pool

def spot_detection(x, min_sigma = 2, max_sigma = 5, thresholdLoG = 50.0, overlap = .75):
    return blob_log(x, min_sigma = min_sigma, max_sigma = max_sigma, threshold = thresholdLoG, overlap = overlap)

def detect_spots(img, thresholdLoG = 50.0, min_sigma = 2, max_sigma = 5, overlap = .75):
    '''
    Runs spot detection using LoG method in ray parallel backend (for each image in a stack).
    '''

    def det_spot(x):
        return blob_log(x, min_sigma = min_sigma, max_sigma = max_sigma, threshold = thresholdLoG, overlap = overlap)

    # Parrallel spot finding
    pool = Pool()
    list_spots = pool.map(det_spot, img)

    # Concatenate all arrays into one including time index
    array_spots = []
    for idx, array in enumerate(list_spots):
        # Create a column with the index value
        index_column = np.full((array.shape[0], 1), idx)
        # Concatenate the index column with the current array
        array_with_index = np.hstack((index_column, array[:, :2]))
        # Add the resulting array to the list
        array_spots.append(array_with_index)

    # Concatenate all arrays along the rows
    array_spots = np.vstack(array_spots)
    return array_spots

def filter_spots_image_border(spots, shape, radius):
    '''
    Filter detections radius distance from image border

    spots: 2D array with the spot coordinates
    shape: 2D image shape tuple
    radius: radius of the circle to be used for filtering
    '''
    particles = []
    for spot in spots:
        y = spot[1]
        x = spot[2]
        if (radius < x < shape[1] - radius) and (radius < y < shape[0] - radius):
            particles.append(spot)
    particles = np.stack(particles, axis = 0)
    return particles

def measure_spots_intensity_radius(spots, img, radius):
    '''
    Code adapted from: 
    https://github.com/sylvainprigent/stracking/blob/main/stracking/properties/_intensity.py

    spots: array of detections
    img: image to measure
    radius: int, size intensity measurement
    '''
    def calculate_particle_intensity(params):
        spots, img, rr, cc, i  = params
        # Calculate intensity of one particle
        x = int(spots[i, 2])
        y = int(spots[i, 1])
        t = int(spots[i, 0])
        # get the disk coordinates
        val = img[t, cc+y, rr+x]
        mean_val = np.mean(val)
        std_val = np.std(val)
        min_val = np.min(val)
        max_val = np.max(val)
        return mean_val, std_val, min_val, max_val

    # Calculate intensity of each particle
    rr, cc = disk((0, 0), radius)
    params_list = [(spots, img, rr, cc, i) for i in range(len(spots))]

    # Use Ray pool for parallel processing
    pool = Pool()
    results = pool.map(calculate_particle_intensity, params_list)

    # Store results
    properties = {}
    properties['label'] = np.arange(len(spots))
    properties['mean_intensity'] = np.array([res[0] for res in results])
    properties['std_intensity'] = np.array([res[1] for res in results])
    properties['min_intensity'] = np.array([res[2] for res in results])
    properties['max_intensity'] = np.array([res[3] for res in results])
    return properties

def measure_spots_intensities(spots, radius, img):
    '''
    spots: array of detections  
    radius: int, size intensity measurement
    shape: tuple (Y, X), size of the image
    '''
    particles = filter_spots_image_border(spots, img.shape[-2:], radius)

    # Calculate intensity of each particle
    measurements = measure_spots_intensity_radius(particles, img, radius)
    return particles, measurements

def calc_spots_distances(spotsA, spotsB, limitRange = None, limitN = None):
    '''
    spotsA: (N, 3) array of spot coordinates
    spotsB: (M, 3) array of spot coordinates
    limitRange: maximum distance between spots to consider
    limitN: maximum number of spots to consider (distance priority)
    
    returns:
        distances: pandas DataFrame of pairwise distances between spots
        nCandidatesInRange: pandas Series of number of candidates in search range
    '''

    distances = cdist(spotsA[:, 1:], spotsB[:, 1:], 'euclidean')
    nCandidatesInRange = np.sum(distances <= limitRange, axis = 1)

    # Flatten to pandas
    rows, cols = distances.shape
    row_indices = np.repeat(np.arange(rows), cols)
    col_indices = np.tile(np.arange(cols), rows)
    values = distances.flatten()
    df = pd.DataFrame({'labelA': row_indices, 'labelB': col_indices, 'distance': values})

    # Filter out distances that are too large
    if limitRange is not None:
        df = df[df['distance'] <= limitRange]

    # Limit the number of pairs to N per detection in A
    if limitN is not None and len(spotsA):
        df = df.sort_values(by=['labelA', 'distance']).groupby('labelA').head(limitN)

    return df, nCandidatesInRange

def metric_spots_relative_intensity(spots, spotsIntensity, limitRange: int = 50, limitN: int = 10, gapMax: int = 1):
    '''
    Calculate spot intensities in a restricted search space.

    spots: array of detections
    spotsIntensity: dict of particle intensities
    limitRange: int, size of the search space
    limitN: int, number of particles to measure, ranked by lowest distance first.
    gapMax: int, frame gap limit for which to make calculations
    '''
    def absolute_relative_difference(x, y):
        return abs(x - y) / x

    def calculate(params):
        spots, spotsIntensity, indexSpots, frame, deltaT = params

        sliceSpotsA = spots[:, 0] == frame
        sliceSpotsB = spots[:, 0] == frame + deltaT

        indexSpotsA = indexSpots[sliceSpotsA]
        indexSpotsB = indexSpots[sliceSpotsB]

        spotsA = spots[sliceSpotsA]
        spotsB = spots[sliceSpotsB]

        dfSpots, nInRange  = calc_spots_distances(spotsA, spotsB, limitRange = limitRange, limitN = limitN)
        dfSpots['frameA'] = frame
        dfSpots['frameB'] = frame + deltaT

        nInRange = pd.DataFrame({'nInRange': nInRange, 'oLabelA': indexSpotsA})
        nInRange['frame'] = frame

        # Cal intensity delta
        for i, row in dfSpots.iterrows():
            # !! translate local spot index to global spot index
            dfSpots.loc[i, 'relIntDiff'] = absolute_relative_difference(
                spotsIntensity['mean_intensity'][indexSpotsA[int(row['labelA'])]],
                spotsIntensity['mean_intensity'][indexSpotsB[int(row['labelB'])]])
            # store original labels (not reindexed per frame)
            dfSpots.loc[i, 'oLabelA'] = int(indexSpotsA[int(row['labelA'])])
            dfSpots.loc[i, 'oLabelB'] = int(indexSpotsB[int(row['labelB'])])
        return dfSpots, nInRange

    def calc_spots_distances(spotsA, spotsB, limitRange = None, limitN = None):
        '''
        spotsA: (N, 3) array of spot coordinates
        spotsB: (M, 3) array of spot coordinates
        limitRange: maximum distance between spots to consider
        limitN: maximum number of spots to consider (distance priority)
        
        returns:
            distances: pandas DataFrame of pairwise distances between spots
            nCandidatesInRange: pandas Series of number of candidates in search range
        '''

        distances = cdist(spotsA[:, 1:], spotsB[:, 1:], 'euclidean')
        nCandidatesInRange = np.sum(distances <= limitRange, axis = 1)

        # Flatten to pandas
        rows, cols = distances.shape
        row_indices = np.repeat(np.arange(rows), cols)
        col_indices = np.tile(np.arange(cols), rows)
        values = distances.flatten()
        df = pd.DataFrame({'labelA': row_indices, 'labelB': col_indices, 'distance': values})

        # Filter out distances that are too large
        if limitRange is not None:
            df = df[df['distance'] <= limitRange]

        # Limit the number of pairs to N per detection in A
        if limitN is not None and len(spotsA):
            df = df.sort_values(by=['labelA', 'distance']).groupby('labelA').head(limitN)

        return df, nCandidatesInRange

    # Spot delta int restricted search space
    timeSlices = int(spots[:, 0].max())
    indexSpots = np.arange(len(spots))
    deltaTs = range(1, gapMax + 1)

    # Calculate intensity differences and distances between spots to next timeframe
    params_list = [(spots, spotsIntensity, indexSpots, frame, deltaT) for deltaT in deltaTs for frame in range((timeSlices + 1) - deltaT)]
    pool = Pool()
    results = pool.map(calculate, params_list)

    # Concatenate tables
    result = pd.concat([tup[0] for tup in results])
    result['distanceScaled'] = result['distance'] * (result['relIntDiff'] + 1)
    result.set_index(['frameA', 'frameB', 'oLabelA', 'oLabelB'], inplace=True)

    nInRange = pd.concat([tup[1] for tup in results])
    return result, nInRange

def create_centered_series(n):
    if n % 2 == 0:
        raise ValueError("n must be an odd integer")
    # Calculate the number of values around the center
    span = int((n - 1) / 2)
    # Create the series centered around zero
    series = np.arange(-span, span + 1)    
    return series

def make_grid_transformations():
    # centered 5 gives center, with 2 concentric rings
    bbox_grid_transformations = [[x, y] for x in create_centered_series(5) for y in create_centered_series(5)]
    return bbox_grid_transformations

def extract_bbox_grid_intensity(img, spots, bbox_extent):
    '''
    Extract values from np array at xy positions using concentric square bounding boxes.
    Core = 1 box & tier 1 = 8 boxes & tier 2 = 16 boxes.
    Needs post-parsing to separate the core and tiers.

    ! Single slice only

    Input:
        img: np.array: [y, x]
        spots: np.array: [y,x]
        bbox_extent: int: width == height of bbox for value extraction
    Output:
        result: np.array: [n spots, n bboxes = 25, n values = bbox_extent**2]
    '''

    def check_intersect_border(rr, cc, shape):
        #Return true if rr or cc below 0 or above shape.
        return np.min(rr) < 0 or np.min(cc) < 0 or np.max(rr) >= shape[0] or np.max(cc) >= shape[1]

    def sign(x):
        return int(x > 0) - int(x < 0)

    # offset all boxes from center by 20% of box size
    offset_base = ceil(bbox_extent*.2)

    # generate transformations (centric, 2 concentric tiers)
    bbox_grid_transformations = make_grid_transformations()

    # total number of boxes
    bbox_grid_n = len(bbox_grid_transformations)

    # base rectangle bbox
    rr, cc = rectangle((-floor(bbox_extent/2), -floor(bbox_extent/2)), extent = bbox_extent)

    # collection of measurements
    bbox_measurements = np.zeros(shape = (len(spots), bbox_grid_n, bbox_extent**2), dtype=np.float32)
    for i_spot, spot in enumerate(spots):
        # spot xy position
        spot_y, spot_x = spot
        # offset base measurement bbox to spot position
        rr_mod, cc_mod = rr + spot_y, cc + spot_x
        # loop over bbox grid
        for i_box, (bbox_base_nx, bbox_base_ny) in enumerate(bbox_grid_transformations):
            # offset box position to spot position in image
            offset_final_x = bbox_base_nx * bbox_extent + offset_base * sign(bbox_base_nx)
            offset_final_y = bbox_base_ny * bbox_extent + offset_base * sign(bbox_base_ny)

            cc_mod_at_box = cc_mod + offset_final_x
            rr_mod_at_box = rr_mod + offset_final_y

            if not check_intersect_border(rr_mod_at_box, cc_mod_at_box, img.shape):
                bbox_measurements[i_spot, i_box] = img[rr_mod_at_box, cc_mod_at_box].flatten()
            else:
                bbox_measurements[i_spot, i_box] = np.full(bbox_extent**2, float('nan'))
    return bbox_measurements

def index_bbox_grid_tiers():
    '''
    # find indexes of bbox by peak (signal) and tier (background 1 & 2) in extracted format
    '''
    bbox_grid_transformations = make_grid_transformations()

    # core bbox (signal)
    bbox_grid_center = [index for index, (x,y) in enumerate(bbox_grid_transformations) if (abs(x) + abs(y) == 0)]

    # indexes of crucial bboxes in grid (background 1)
    bbox_grid_tier1 = [index for index, (x,y) in enumerate(bbox_grid_transformations) if abs(x) <= 1 and abs(y) <= 1 and (abs(x) + abs(y) > 0)]

    # indexes of crucial bboxes in grid (background 2)
    bbox_grid_tier2 = [index for index, (x,y) in enumerate(bbox_grid_transformations) if abs(x) > 1 or abs(y) > 1]

    return bbox_grid_center, bbox_grid_tier1, bbox_grid_tier2

def fetch_int_valid(arr, mask):
    arr_out = np.array([
        row[mask_row].mean() if np.any(mask_row) else np.nan
        for row, mask_row in zip(arr, mask)
    ])
    return arr_out

def parse_bbox_grid_intensity(data, bbox_indices: tuple, legacy: bool = False, thr_bg = .5):
    '''
    parse grid intensity data by core and tiers, excluding tiered bboxes with high minmax ratio.
    '''
    # separate argument bbox indices
    bbox_grid_center, bbox_grid_tier1, bbox_grid_tier2 = bbox_indices

    # calculate stats for all boxes
    intensity_mean = np.mean(data, axis = -1)
    if legacy:
        intensity_coef = (np.min(data, axis  = -1) * 4) / np.max(data, axis  = -1)
        intensity_valid = np.max(data, axis = -1) < (np.min(data, axis = -1) * 4)
    else:
        intensity_std = np.std(data, axis = -1)
        intensity_coef = intensity_std / intensity_mean
        intensity_valid = intensity_coef < thr_bg

    # separate core and tiered boxes
    intensity_peak = intensity_mean[:, bbox_grid_center[0]]
    intensity_bg1 = intensity_mean[:, bbox_grid_tier1]
    intensity_bg2 = intensity_mean[:, bbox_grid_tier2]

    # bg calc with exclusion for invalid bboxes, tier 1
    bg1_ratio_valid = intensity_valid[:, bbox_grid_tier1]
    intensity_bg1_mean = fetch_int_valid(intensity_bg1, bg1_ratio_valid)

    # print(intensity_bg1)
    # nans = np.isnan(intensity_bg1_mean)
    # export = (np.max(data, axis = -1)[nans][:, bbox_grid_tier1], np.min(data, axis = -1)[nans][:, bbox_grid_tier1])
    # intensity_valid[nans][:, bbox_grid_tier1]
    # print(np.min(data, axis = -1)[nans][:, bbox_grid_tier1].shape)

    # bg calc with exclusion for invalid bboxes, tier 2
    bg2_ratio_valid = intensity_valid[:, bbox_grid_tier2]
    intensity_bg2_mean = fetch_int_valid(intensity_bg2, bg2_ratio_valid)

    # fill values where no valid tier 1 result with tier 2 bg
    intensity_bg1_invalid = np.nansum(bg1_ratio_valid, axis = -1) == 0
    intensity_bg1_mean[intensity_bg1_invalid] = intensity_bg2_mean[intensity_bg1_invalid]

    return intensity_peak, intensity_bg1_mean, intensity_peak - intensity_bg1_mean, intensity_coef

def draw_bbox_grid_mask(mask, spots, bbox_extent, label = None, add:bool=False):
    '''
    Paste mask areas for np array at xy positions using concentric square bounding boxes.
    Core = 1 box & tier 1 = 8 boxes & tier 2 = 16 boxes.
    
    ! Single slice only

    Input:
        mask: np.array: [y, x]
        spots: np.array: [y,x]
        bbox_extent: int: width == height of bbox for value extraction
        label: None to paste bbox index at position, np array with matching shape to fill values.
    Output:
        return: np.array: updated mask
    '''

    def check_intersect_border(rr, cc, shape):
        #Return true if rr or cc below 0 or above shape.
        return np.min(rr) < 0 or np.min(cc) < 0 or np.max(rr) >= shape[0] or np.max(cc) >= shape[1]

    def sign(x):
        return int(x > 0) - int(x < 0)

    # offset all boxes from center by 20% of box size
    offset_base = ceil(bbox_extent*.2)

    # generate transformations (centric, 2 concentric tiers)
    bbox_grid_transformations = make_grid_transformations()

    # base rectangle bbox
    rr, cc = rectangle((-floor(bbox_extent/2), -floor(bbox_extent/2)), extent = bbox_extent)

    if label is not None:
        if not isinstance(label, np.ndarray) or label.shape != (len(spots), len(bbox_grid_transformations)):
            raise ValueError("Label must be a numpy array of shape ", (len(spots), len(bbox_grid_transformations)))

    # collection of measurements
    for i_spot, spot in enumerate(spots):
        # spot xy position
        spot_y, spot_x = spot
        # offset base measurement bbox to spot position
        rr_mod, cc_mod = rr + spot_y, cc + spot_x
        # loop over bbox grid
        for i_box, (bbox_base_nx, bbox_base_ny) in enumerate(bbox_grid_transformations):
            # offset box position to spot position in image
            offset_final_x = bbox_base_nx * bbox_extent + offset_base * sign(bbox_base_nx)
            offset_final_y = bbox_base_ny * bbox_extent + offset_base * sign(bbox_base_ny)

            cc_mod_at_box = cc_mod + offset_final_x
            rr_mod_at_box = rr_mod + offset_final_y

            if not check_intersect_border(rr_mod_at_box, cc_mod_at_box, mask.shape):
                if label is None:
                    mask[rr_mod_at_box, cc_mod_at_box] = i_spot + i_box + 1
                else:
                    if add:
                        mask[rr_mod_at_box, cc_mod_at_box] += label[i_spot, i_box]
                    else:
                        mask[rr_mod_at_box, cc_mod_at_box] = label[i_spot, i_box]
    return mask
    