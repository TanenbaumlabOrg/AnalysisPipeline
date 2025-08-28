import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
import re
from psutil import cpu_count
from tifffile import imwrite
import dask.array as da

from tqdm.notebook import tqdm

sys.path.insert(1, os.path.dirname(os.getcwd()))
import FlatDarkField
import LinkPointToObject

def find_tile_region_start(sizes: dict, px_size: float, positions):
    '''
    Valid overlap assumed between 5 and 25% in either X or Y direction.

    Input
        sizes [dict]: nd2 metadata collection dimensions
        px_size [float]: size of pixels in microns, from nd2 metadata
        positions [pandas]: positions and indexes for all images in nd2 file

    Return
        tile_region_init [list]: P indexes for images not overlapping with the previous in sequence
        overlap_percent [float]: average overlap percentage for overlapping tiles
    '''

    overlap_low = .015
    overlap_high = .25

    df = positions[(positions['iT'] == 0) & (positions['iZ'] == 0)].copy()
    size_um_x = sizes['X'] * px_size
    size_um_y = sizes['Y'] * px_size

    # Shift the 'X' and 'Y' columns down by one row
    df['X_next'] = df['X'].shift(1)
    df['Y_next'] = df['Y'].shift(1)

    # Calculate the differences
    df['X_diff'] = abs(df['X_next'] - df['X'])
    df['Y_diff'] = abs(df['Y_next'] - df['Y'])

    df['X_overlap_perc'] = 1 - (df['X_diff'] / size_um_x).round(3)
    df['Y_overlap_perc'] = 1 - (df['Y_diff'] / size_um_y).round(3)

    df['overlap_boolX'] = ((df['X_overlap_perc'] >= overlap_low) & (df['X_overlap_perc'] <= overlap_high)) & ((df['Y_overlap_perc'] >= .95) & (df['Y_overlap_perc'] <= 1.05))
    df['overlap_boolY'] = ((df['Y_overlap_perc'] >= overlap_low) & (df['Y_overlap_perc'] <= overlap_high)) & ((df['X_overlap_perc'] >= .95) & (df['X_overlap_perc'] <= 1.05))
    df['overlap_bool'] = df['overlap_boolX'] ^ df['overlap_boolY']

    tile_region_init = df[df['overlap_bool'] == False]['iP'].tolist()

    overlapping_tiles_x = df[df['overlap_bool'] & df['overlap_boolX']]
    if not overlapping_tiles_x.empty:
        overlap_percent = round(overlapping_tiles_x['X_overlap_perc'].mean()*100, 1)
    else:
        print("No overlapping images")
        overlap_percent = np.nan

    return tile_region_init, overlap_percent

def tile_region_layout(positions_tr, sizes: dict, size_pixel: float, verbose: [bool] = False):
    '''
    Input
        positions_tr [pandas]: positions and indexes for a single tile region (subset)
        sizes [dict]: nd2 metadata collection dimensions
        size_pixel [float]: size of a pixel in microns
        verbose [bool]: whether to print intermediary values
    Return
        positions_tr_out [pandas]: positions and indexes for a single tile region, appended grid layout
    '''
    coordinates = positions_tr[['X', 'Y']].to_numpy()
    x = coordinates[:, 0]
    y = coordinates[:, 1]

    img_width = sizes['X'] * size_pixel
    img_height = sizes['Y'] * size_pixel

    # Step 1: Find the minimum and maximum values for x and y, which define the left top corner
    min_x, min_y = np.min(coordinates, axis = 0)

    # Step 2: Find average step size in X and Y for sequential tiles
    x_diff_abs = np.absolute(np.diff(x))
    x_diff_median = round(np.median(x_diff_abs[np.where(x_diff_abs > 5)]), ndigits = 2)

    y_diff_abs = np.absolute(np.diff(y))
    y_diff_median = round(np.median(y_diff_abs[np.where(y_diff_abs > 5)]), ndigits = 2)

    # Step 3: Find overlap between tiles for sequential tiles
    overlap_x = (img_width - x_diff_median) / img_width
    overlap_y = (img_height - y_diff_median) / img_height

    if verbose:
        print("min coord:", min_x, min_y)
        print("diff_median:", x_diff_median, y_diff_median)
        print("overlap:", overlap_x, overlap_y)

    if ((.02 < overlap_x < .3) or (.02 < overlap_y < .3)):
        x_inds = np.round((x - min_x) / x_diff_median)
        y_inds = np.round((y - min_y) / y_diff_median)

        x_grid_max = max(x_inds.astype('int')) + 1
        #y_grid_max = max(y_inds.astype('int')) + 1

        positions_tr_out = pd.DataFrame({'index': range(len(coordinates)),
                                         'grid_x': x_grid_max - x_inds.astype('int') - 1,
                                         'grid_y': y_inds.astype('int')})
    else:
        print('bad predicted overlap between tiles')
        positions_tr_out = pd.DataFrame({'index': [0], 'grid_x': [0], 'grid_y': [0]})

    return positions_tr_out

def find_tile_region_all(tr_init: list, sizes: dict, positions, size_pixel: float, verbose = False):
    '''
    For all images in collection, find their grid position in relevant tile region.
    Input
        tr_init [list]: P indexes for images not overlapping with the previous in sequence a.k.a. TR start positions
        sizes [dict]: nd2 metadata collection dimensions
        positions [pandas]: all image positions
        size_pixel [float]: size of a pixel in microns
        verbose [bool]: whether to print intermediary values
    Return
        positions_tr_all [pandas]: for all images in collection table grid position in which tile region
        ! ? first t and z only ?
    '''
    # for all TRs calculate layout
    positions_tr_all = []
    for i, tr_start in enumerate(tr_init):
        # establish what tiles belong to the current region
        if (i < len(tr_init)-1):
            # not the last region
            p_indexes_tr = range(tr_start, tr_init[i+1])
        else:
            # last region in collection
            p_indexes_tr = range(tr_start, sizes['P'])
        print("TR:", i, "len:", len(p_indexes_tr))
        
        # calculate TR grid layout
        position_table = tile_region_layout(positions[positions['iP'].isin(p_indexes_tr)], sizes, size_pixel, verbose)
        if verbose:
            print(positions[positions['iP'].isin(p_indexes_tr)])
            print(position_table)
            print(p_indexes_tr)
        position_table['TR'] = i
        position_table['P'] = p_indexes_tr
        positions_tr_all.append(position_table)
    positions_tr_all = pd.concat(positions_tr_all)
    return positions_tr_all

def export_tile_img_for_stitching(data_img, name: str, path_export_tile: str, sizes: dict,
                                  tr_init: list, positions, c_stitch: int, t_stitch: int,
                                  flatfield: bool, img_ff: list, img_df, blur: int = 0):
    '''
    Input
        data_img [dask]: dask array of nd2 imaging dataset.
        name [str]: generic name of dataset (substringed nd2 filename).
        path_export_tile [str]: location where tile images are exported.
        sizes [dict]: nd2 metadata collection dimensions.
        tr_init [list]: starting positions of tile regions.
        positions [pandas]: dataframe containing TR and positions therein for all images in dataset. Single T and Z only!
        c_stitch [int]: which channel to use for stitching.
        t_stitch [int]: which timeframe to use for stitching.
        flatfield [bool]: whether to apply illumination correction on images before stitching.
        img_ff [list of np2d]: flat field images for channels, in order.
        img_df [np 2d]: dark field image for microscope.
        blur [int]: radius of gaussian blur to apply to image before export. 0 = no blur.
    Output
        tile images as tif with name corresponding to TR and position therein.
    '''
    for tr, tr_start in enumerate(tr_init):
        positions_tr = positions[positions['TR'] == tr]
        if (len(positions_tr) > 1):
            # only if TR size > 1 tile
            name_img_base = name + "_TR" + str(tr)
            for index, p in positions_tr[['index', 'P']].to_numpy():
                # slice single tile image from dataset
                if sizes.get('T', 1) == 1:
                    img_p = data_img[p, c_stitch].compute()
                else:
                    img_p = data_img[t_stitch, p, c_stitch].compute()

                if flatfield:
                    img_p = FlatDarkField.ffdf(img_p, img_ff[c_stitch].squeeze(), img_df)
                
                if blur > 0:
                    img_p = filters.gaussian(img_p, blur, preserve_range=True)
                    img_p = np.nan_to_num(img_p)
                    img_p = np.clip(img_p, 0, 65535)
                    img_p = np.round(img_p).astype(np.uint16)

                position_p_x = positions_tr.at[index, 'grid_x'] + 1
                position_p_y = positions_tr.at[index, 'grid_y'] + 1
                name_img_p = name_img_base + "_x" + str(position_p_x).zfill(3) + "_y" + str(position_p_y).zfill(3)
                imwrite(path_export_tile + name_img_p + ".tif", img_p)

def perform_stitching_mist(ij, name: str, path_export_tile: str, tr_init: list, positions, overlap: float, px_size: float):
    '''
    Input
        ij [pyimagej env]: initiated imagej environment containing the MIST plugin.
        name [str]: generic name of dataset (substringed nd2 filename).
        path_export_tile [str]: location where tile images are exported.
        tr_init [list]: starting positions of tile regions.
        positions [pandas]: dataframe containing TR and positions therein for all images in dataset. Single T and Z only!
        overlap [float]: percentage overlap between adjacent tiles.
        px_size [float]: pizel size in microns
    Output
        mist txt file with pixel coordinates for each tile image.
    '''

    macro_mist = """
    // @ String path_images_in
    // @ String filename_pattern
    // @ String path_out
    // @ String filename_out
    // @ String grid_width
    // @ String grid_height
    // @ String overlap
    // @ String px_size
    // @ String threads
    run("MIST", "gridwidth=["+grid_width+"] gridheight=["+grid_height+"] starttilerow=1 starttilecol=1 imagedir=["+path_images_in+"] filenamepattern=["+filename_pattern+"] filenamepatterntype=ROWCOL gridorigin=UL assemblefrommetadata=false assemblenooverlap=false globalpositionsfile=[] startrow=0 startcol=0 extentwidth=["+grid_width+"] extentheight=["+grid_height+"] timeslices=1 istimeslicesenabled=false outputpath=["+path_out+"] displaystitching=false outputfullimage=false outputmeta=true outputimgpyramid=false blendingmode=OVERLAY blendingalpha=NaN compressionmode=UNCOMPRESSED outfileprefix=["+filename_out+"] unit=MICROMETER unitx=["+px_size+"] unity=["+px_size+"] programtype=AUTO numcputhreads=["+threads+"] stagerepeatability=0 horizontaloverlap=["+overlap+"] verticaloverlap=["+overlap+"] numfftpeaks=0 overlapuncertainty=25.0 isusedoubleprecision=false isusebioformats=false issuppressmodelwarningdialog=false isenablecudaexceptions=false translationrefinementmethod=SINGLE_HILL_CLIMB numtranslationrefinementstartpoints=16 headless=true loglevel=MANDATORY debuglevel=NONE");
    """

    for tr, tr_start in enumerate(tr_init):
        positions_tr = positions[positions['TR'] == tr]

        if (len(positions_tr) > 1):
            # only if TR size > 1 tile
            name_img_base = name + "_TR" + str(tr)            
            x_grid_max = max(positions_tr['grid_x'].astype('int')) + 1
            y_grid_max = max(positions_tr['grid_y'].astype('int')) + 1
            args_mist = {
                'path_images_in': path_export_tile,
                'filename_pattern': name_img_base + "_x{ccc}_y{rrr}.tif",
                'path_out': path_export_tile,
                'filename_out': name + '_TR' + str(tr) + "_",
                'grid_width': str(x_grid_max),
                'grid_height': str(y_grid_max),
                'overlap': str(overlap),
                'px_size': str(round(px_size, 5)),
                'threads': str(cpu_count()-1)
            }
            print(args_mist)
            ij.py.run_macro(macro_mist, args_mist)

def perform_stitching_stage(tr_init: list, positions, px_size: float):
    '''
    Calculate position for tile images in a tileregion according to stage coordinates alone.
    X = micron_x/px_size - (min of X), etc
    
    Input
        tr_init [list]: starting positions of tile regions.
        positions [pandas]: dataframe containing TR and positions therein for all images in dataset. Single T and Z only!
        px_size [float]: pixel size in microns
    Return
        table [pandas] with pixel coordinates for each tile image.
    '''
    list_positions = []
    for tr, _ in enumerate(tr_init):
        positions_tr = positions[positions['TR'] == tr].copy()
        if len(positions_tr) > 1:
            px_x = (1 - positions_tr['X'].values) / px_size
            px_y = positions_tr['Y'].values / px_size
            px_x_st = px_x - min(px_x)
            px_y_st = px_y - min(px_y)

            positions_tr.loc[:, 'x_px'] = np.round(px_x_st).astype(int)
            positions_tr.loc[:, 'y_px'] = np.round(px_y_st).astype(int)
            
            list_positions.append(positions_tr)
    positions = pd.concat(list_positions, axis=0)

    return positions[['index', 'grid_x', 'grid_y', 'TR', 'P', 'x_px', 'y_px']]

def parse_mist_result(name: str, path_export_tile: str, path_layout: str, tr_init: list, positions):
    '''
    Input
        name [str]: generic name of dataset (substringed nd2 filename).
        path_export_tile [str]: location where tile images are exported.
        path_layout [str]: where to store tileregion layout file
        tr_init [list]: starting positions of tile regions.
        positions [pandas]: dataframe containing TR and positions therein for all images in dataset. Single T and Z only!
    Output/return
        positions_join [pandas/csv]: positions table merged with pixel positions for in-silico stitch of tileregion.
    '''
    # Create a 2D array to store the position values
    positions_mist = []

    # loop over tile regions
    for tr, tr_start in enumerate(tr_init):
        positions_tr = positions[positions['TR'] == tr]
        if (len(positions_tr) > 1):
            # suspected path of mist output
            path_mist_global_positions = path_export_tile + name + "_TR" + str(tr) + "_global-positions-1.txt"
            # Parse positions file
            if (os.path.exists(path_mist_global_positions)):
                # Define the regular expression to match the position value
                pos_regex = r"position: \((\d+), (\d+)\)"
                grid_regex = r"grid: \((\d+), (\d+)\)"

                # Open the file and read its contents
                with open(file = path_mist_global_positions, mode = "r", encoding = "utf-8") as f:
                    content = f.read()
                # Split the content into lines
                lines = content.split("\n")
                # Loop over each line and extract the position value
                for line in lines:
                    # Use regular expression to match the position value
                    match_position = re.search(pos_regex, line)
                    match_grid = re.search(grid_regex, line)
                    if match_position and match_grid:
                        # Convert the position value to integers and add it to the array
                        x_px = int(match_position.group(1))
                        y_px = int(match_position.group(2))
                        x_i =  int(match_grid.group(1))
                        y_i =  int(match_grid.group(2))
                        
                        position = pd.DataFrame({"x_px": x_px, "y_px": y_px, "grid_x": x_i, "grid_y": y_i, "TR": tr}, index = [0])
                        positions_mist.append(position)
            else:
                print("No stitch positions file?")
        else:
            # not a TR
            positions_mist.append(pd.DataFrame({"x_px" : 0, "y_px": 0, "grid_x": 0, "grid_y": 0, "TR": tr}, index = [0]))

    # Concatenate results to dataframe
    positions_mist = pd.concat(positions_mist, axis = 0)
    positions_join = pd.merge(positions, positions_mist, on = ['grid_x', 'grid_y', 'TR'])
    positions_join.to_csv(path_or_buf = path_layout + "/" + name + "_TRLayout.csv", sep = ";", decimal = ".", index = False)
    return positions_join

def plot_tileregion_layout(positions_join, positions_first, TR_init):
    '''
    Illustrates identified tileregions in stage-coordinate space.
    '''
    # bind back stage coordinates
    coords_stage = positions_join.merge(positions_first, left_on = 'P', right_on = 'iP')

    # color by tileregion participation
    plt.subplot(1, 2, 1, aspect = 'equal')
    plt.scatter(coords_stage['X'], coords_stage['Y'], c = coords_stage['TR'], cmap = 'Pastel1')
    for TR, TR_start in enumerate(TR_init):
        # each tile region to bbox
        coords_stage_TR = coords_stage[coords_stage['TR'] == TR]
        X_min = coords_stage_TR['X'].min()
        X_max = coords_stage_TR['X'].max()
        Y_min = coords_stage_TR['Y'].min()
        Y_max = coords_stage_TR['Y'].max()

        plt.plot([X_min, X_min], [Y_min, Y_max], 'r') #left
        plt.plot([X_min, X_max], [Y_min, Y_min], 'r') #bottom
        plt.plot([X_max, X_max], [Y_min, Y_max], 'r') #right
        plt.plot([X_min, X_max], [Y_max, Y_max], 'r') #top
        plt.text(x = (X_min + X_max)/2, y = (Y_min + Y_max)/2, s = "TR" + str(TR), c = 'black')
    plt.axis('off')
    plt.title('Identified TileRegions')
    
    # color by index in region
    plt.subplot(1, 2, 2, aspect = 'equal')
    plt.scatter(x = coords_stage['X'], y = coords_stage['Y'], c = coords_stage['index'], cmap = 'spring')
    for index, row in coords_stage.iterrows():
        plt.text(row['X'], row['Y'], str(int(row['index'])), fontsize = 10, ha = 'center', va = 'bottom')
    plt.axis('off')
    plt.title('Index in TileRegion')

    # format and show
    plt.tight_layout()
    plt.show()

def create_distance_map(shape_img):
    # Define the dimensions of the array
    height, width = shape_img

    # Create arrays representing the vertical and horizontal indices
    y_indices = np.arange(height).reshape(-1, 1)  # Column vector for row indices
    x_indices = np.arange(width).reshape(1, -1)  # Row vector for column indices

    # Calculate the distances to the edges
    distance_to_top = y_indices
    distance_to_bottom = height - 1 - y_indices
    distance_to_left = x_indices
    distance_to_right = width - 1 - x_indices

    # Use pairwise np.minimum to find the smallest distance to any edge
    distance_to_edge = np.minimum(
        np.minimum(distance_to_top, distance_to_bottom),
        np.minimum(distance_to_left, distance_to_right)
    ).astype(np.uint16) + 1

    return distance_to_edge

def stitch_tiles(img, layout, dtype = np.uint16, add: bool = False):
    '''
    img [np]: stacked tile images on axis zero: [p, y, x]
    layout: pd df containing pixel coordinates for tile positions and image index in stack ['y_px', 'x_px', 'p'], where p matches index in image array.
    dtype: type of data to output, img is converted to this type
    add: whether to add or replace data in the array
    '''
    
    # check image dtype compatible?
    if img.dtype != dtype:
        img = img.astype(dtype)
    
    # dimensions
    height, width = img.shape[-2:]
    pos_p = layout['P'].values
    pos_x = layout['x_px'].values
    pos_y = layout['y_px'].values

    # make empty stitch image
    stitch = np.zeros((
        pos_y.max() + height,
        pos_x.max() + width),
        dtype = dtype)

    # put in image tiles by addition in regions
    for ymin, xmin, p in zip(pos_y, pos_x, pos_p):
        if add:
            stitch[ymin:ymin + height,
                   xmin:xmin + width] += img[p]
        else:
            stitch[ymin:ymin + height,
                   xmin:xmin + width] = img[p]
    
    return stitch

def stitch_by_gradient(img, layout):
    # dimensions
    height, width = img.shape[:-2]
    pos_p = layout['P'].values
    pos_x = layout['x_px'].values
    pos_y = layout['y_px'].values

    # distance maps
    img_dist = create_distance_map((height, width))
    layout_dist = layout.copy()
    layout_dist['P'] = 0
    img_dist_stitch = stitch_tiles(np.expand_dims(img_dist, axis = 0), layout_dist, add = True)
    
    # TODO how to work in multi channel and Z but not spend overhead on dist map calc

    # precalculate images with gradient
    img_adj = np.zeros_like(img, dtype = np.float32)
    for ymin, xmin, p in zip(pos_y, pos_x, pos_p):
        img_dist_from_stitch = img_dist_stitch[ymin:ymin + height, xmin:xmin + width]
        tile_scaling = img_dist / img_dist_from_stitch
        img_tile_scaled = img[p] * tile_scaling
        img_adj[p] = img_tile_scaled
    
    # TODO fix dtype argument process

    # make final stitch
    img_stitch = stitch_tiles(img_adj, layout, np.float32, add = True).round().astype(np.uint16)

def stitch_tile_images(data_img, tr_init: list, sizes: dict, positions,
                       flatfield: bool = False, img_ff: list = None, img_df = None,
                       verbose: bool = False, blend: bool = False):
    '''
    Does currently not allow Z axis.
    Only works for timeseries!

    Input
        data_img [dask]: dask array of nd2 imaging dataset.
        tr_init [list]: starting positions of tile regions.
        positions [pandas]: dataframe containing TR and positions therein for all images in dataset.
            positions_join, containing MIST pixel positions.
        flatfield [bool]: whether to apply illumination correction on images before stitching.
        img_ff [list of np2d]: flat field images for channels, in order.
        img_df [np 2d]: dark field image for microscope.
    Return
        stitched_image [list of np]: per tileregion a numpy array in list.
    '''

    axis_names = list(sizes.keys())

    stitched_images = []
    for tr, tr_start in tqdm(enumerate(tr_init), total = len(tr_init)):
        positions_tr = positions[positions['TR'] == tr]
        # if len(positions_tr) > 0:
        stitched_image_size = (
            sizes.get('T', 1),
            sizes.get('Z', 1),
            sizes.get('C', 1),
            positions_tr['y_px'].max() + sizes['Y'],
            positions_tr['x_px'].max() + sizes['X'],
        )
        if verbose:
            print(stitched_image_size)

        # create empty array to paste tile images in
        stitched_image = np.zeros(stitched_image_size, dtype=np.uint16 if not blend else np.float32)

        if blend:
            # distance maps
            weight_map = create_distance_map((sizes['Y'], sizes['X']))
            if verbose:
                print(weight_map.shape)
            layout_temp = positions_tr.copy()
            layout_temp['P'] = 0
            weight_map_stitch = stitch_tiles(
                np.expand_dims(weight_map, axis = 0),
                layout_temp,
                add = True
            )

        # loop in reverse order over all tile positions for current TR
        for _, row in tqdm(positions_tr[::-1].iterrows(),
                           total = len(positions_tr),
                           desc='stitching tiles TR: {tr}'.format(tr=tr)):
            # isolate imaging data for 1 position
            p = row['P']
            slice_tuple = create_slice_tuple(axis_names, {'P': p})
            img_p = data_img[slice_tuple].compute()

            if verbose:
                print(slice_tuple)

            # stitch paste positions
            y_px = row["y_px"]
            x_px = row["x_px"]

            # loop over C, Z, T, if present!
            for t in range(sizes.get('T', 1)):
                for z in range(sizes.get('Z', 1)):
                    for c in range(sizes.get('C', 1)):
                        slice_tuple = create_slice_tuple(axis_names, {'C': c, 'Z': z, 'T': t})
                        img = img_p[slice_tuple].squeeze()

                        if verbose:
                            print(slice_tuple)
                            print(img.shape)

                        if flatfield:
                            if img_df.ndim == 3:
                                img = FlatDarkField.ffdf(img, img_ff[c].squeeze(), img_df[c].squeeze(), format_out = np.uint16)
                            if img_df.ndim == 2:
                                img = FlatDarkField.ffdf(img, img_ff[c].squeeze(), img_df, format_out = np.uint16)

                        if blend and len(positions_tr) > 1:
                            weight_tile = weight_map_stitch[y_px : y_px + sizes['Y'], x_px : x_px + sizes['X']]
                            img = np.multiply(img, weight_map / weight_tile)

                            # add at tile position in full region
                            stitched_image[t, z, c, y_px : y_px + sizes['Y'], x_px : x_px + sizes['X']] += img
                        else:
                            # paste at tile position in full region
                            stitched_image[t, z, c, y_px : y_px + sizes['Y'], x_px : x_px + sizes['X']] = img

        if blend:
            np.round(stitched_image, out = stitched_image)
            stitched_image = stitched_image.astype(np.uint16, copy = False)

        stitched_images.append(stitched_image)
    return stitched_images

def create_slice_tuple(axis_names, slice_values):
    '''
    Slicing framework for index at dimension name if available
    '''
    slice_names = list(dict.keys(slice_values))
    slice_tuple = tuple([
        slice(slice_values.get(axis, None), slice_values.get(axis, None) + 1)
        if axis in slice_names else slice(None)
        for axis in axis_names
    ])
    return slice_tuple

def assign_objectid_point_per_tile(spot_df, positions_tile_df, mask, sizes: dict, buffer: int = 10):
    '''
    This function assigns identified spots in individual tile images to their parent object (mask element: eg cell/nucleus) in tileregion context.
    Also resolves spots/points identified in tile overlap, preventing repeated counting.
    In overlap spots are retained only for first acquired image (dedup), and n pixels from edge of the tile image (edge artefact).
    
    ! Single tile region and timeframe input.

    Steps:
    1. Loop over tile images
    2. Assign spots to objects
    3. Filter spots inset n px tile edge.
    4. Dedup spot record other tiles using insets from preceding images.
    
    Input
        spot_df [pandas]: table with all identified spots in all tile images, including tile index.
        positions_tile_df [pandas]: table with pixel and grid positions of all tile images per tileregion in dataset.
        mask [np]: Single mask image [Y,X].
        sizes [dict]: dataset dimensionality.
        buffer [int]: number of pixels to filter at from the image border.
    Return
        spot_df with appended columns for pixel position in stitch, object id, and deduplication indicator.
    '''

    # pad spot XY by tile coordinates
    spot_df = spot_df.merge(positions_tile_df, on = 'P')
    spot_df['X_padded'] = spot_df['X'] + spot_df['x_px']
    spot_df['Y_padded'] = spot_df['Y'] + spot_df['y_px']
    spot_df['C'] = spot_df['C'].astype(int)
    spot_df['P'] = spot_df['P'].astype(int)

    # initialize a deduplication mask
    mask_dedup = np.zeros_like(mask, dtype = bool)

    # initialize empty list, 1 entry per tile
    mask_id_list = []

    # for each tile
    for i, row in positions_tile_df.iterrows():
        x_px_start = row['x_px']
        y_px_start = row['y_px']
        x_px_stop = x_px_start + sizes['X']
        y_px_stop = y_px_start + sizes['Y']

        # all spots for current TR
        spots_df_p = spot_df[spot_df['P'] == row['P']]

        # filter spots within buffer region
        spots_df_p = spots_df_p[(spots_df_p['X_padded'] >= x_px_start + buffer) &
                                (spots_df_p['X_padded'] <= x_px_stop - buffer) &
                                (spots_df_p['Y_padded'] >= y_px_start + buffer) &
                                (spots_df_p['Y_padded'] <= y_px_stop - buffer)]
        spots_df_p_np = spots_df_p[['Y','X']].to_numpy()

        # crop mask
        mask_tile = mask[y_px_start : y_px_stop, x_px_start : x_px_stop]
        mask_dedup_tile = mask_dedup[y_px_start : y_px_stop, x_px_start : x_px_stop]

        # find object ID per spot
        mask_id = LinkPointToObject.PointToMaskID(spots_df_p_np, mask_tile)
        # find overlap with deduplication mask per spot
        dedup = LinkPointToObject.PointToMaskID(spots_df_p_np, mask_dedup_tile)

        # bind IDs
        mask_id = np.column_stack((np.array(mask_id), np.array(dedup), np.array(spots_df_p.index)))
        mask_id_list.append(mask_id)

        # update dedup mask
        mask_dedup_tile[y_px_start + buffer : y_px_stop - buffer, x_px_start + buffer : x_px_stop - buffer] = True

    mask_id_list = pd.DataFrame(np.concatenate(mask_id_list), columns = ['IDCell', 'dedup', 'index']).set_index('index')

    # bind mask id to spot dataframe
    spot_df_assigned = spot_df.join(mask_id_list)
    spot_df_assigned['C'] = spot_df_assigned['C'].astype(int)
    spot_df_assigned['P'] = spot_df_assigned['P'].astype(int)

    return spot_df_assigned

def assign_point_per_tile_dedup(spot_df, positions_tile_df, sizes: dict, buffer: int = 10):
    '''
    This function assigns identified spots in individual tile images in tileregion context.
    Also resolves spots/points identified in tile overlap, preventing repeated counting.
    In overlap spots are retained only for first acquired image (dedup), and n pixels from edge of the tile image (edge artefact).
    
    ! Single tile region and timeframe input.

    Steps:
    1. Loop over tile images
    2. Filter spots inset n px tile edge.
    3. Dedup spot record other tiles using insets from preceding images.
    
    Input
        spot_df [pandas]: table with all identified spots in all tile images, including tile index.
        positions_tile_df [pandas]: table with pixel and grid positions of all tile images per tileregion in dataset.
        sizes [dict]: dataset dimensionality.
        buffer [int]: number of pixels to filter at from the image border.
    Return
        spot_df with appended columns for pixel position in stitch, object id, and deduplication indicator.
    '''

    # pad spot XY by tile coordinates
    spot_df = spot_df.merge(positions_tile_df, on = 'P')
    spot_df['X_padded'] = spot_df['X'] + spot_df['x_px']
    spot_df['Y_padded'] = spot_df['Y'] + spot_df['y_px']
    spot_df['C'] = spot_df['C'].astype(int)
    spot_df['P'] = spot_df['P'].astype(int)

    # initialize a deduplication mask
    stitch_shape_x = positions_tile_df['x_px'].max() + sizes['X']
    stitch_shape_y = positions_tile_df['y_px'].max() + sizes['Y']

    mask_dedup = np.zeros(shape = (stitch_shape_x, stitch_shape_y), dtype = bool)

    # initialize empty list, 1 entry per tile
    mask_id_list = []

    # for each tile
    for i, row in positions_tile_df.iterrows():
        x_px_start = row['x_px']
        x_px_start_b = x_px_start + buffer
        y_px_start = row['y_px']
        y_px_start_b = y_px_start + buffer
        x_px_stop = x_px_start + sizes['X']
        x_px_stop_b = x_px_stop - buffer
        y_px_stop = y_px_start + sizes['Y'] - buffer
        y_px_stop_b = y_px_stop - buffer

        # all spots for current tile
        spots_df_p = spot_df[spot_df['P'] == row['P']]

        # filter spots within buffer region
        spots_df_p = spots_df_p[(spots_df_p['X_padded'] >= x_px_start_b) &
                                (spots_df_p['X_padded'] <= x_px_stop_b) &
                                (spots_df_p['Y_padded'] >= y_px_start_b) &
                                (spots_df_p['Y_padded'] <= y_px_stop_b)]
        spots_df_p_np = spots_df_p[['Y', 'X']].to_numpy()

        # deduplication mask handling
        # mask extracted tile region - buffer
        mask_dedup_tile = mask_dedup[y_px_start : y_px_stop, x_px_start : x_px_stop]
        # find overlap with deduplication mask per spot
        dedup = LinkPointToObject.PointToMaskID(spots_df_p_np, mask_dedup_tile)
        # update dedup mask
        mask_dedup[y_px_start_b : y_px_stop_b, x_px_start_b : x_px_stop_b] = True

        # bind IDs
        mask_id = np.column_stack((np.array(dedup), np.array(spots_df_p.index)))
        mask_id_list.append(mask_id)

    mask_id_list = pd.DataFrame(np.concatenate(mask_id_list), columns = ['dedup', 'index']).set_index('index')

    # bind mask id to spot dataframe
    spot_df_assigned = spot_df.join(mask_id_list)
    spot_df_assigned['C'] = spot_df_assigned['C'].astype(int)
    spot_df_assigned['P'] = spot_df_assigned['P'].astype(int)

    return spot_df_assigned
