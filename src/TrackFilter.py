import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.dirname(os.getcwd()))
from PointDensity import calc_point_density_2d
from LinkPointToObject import PointToMaskID

def track_filter_min_length(tracks, threshold: int = 0):
    '''
    Filter tracks by n obs per track ~ track length, keep those with nobs > threshold.
    
    input:
        tracks [pandas]: tracking table
        threshold: minimum track length threshold (frames)

    print:
        histogram of stats

    return:
        track_id [array] to retain
    '''

    # group by track_id and count the number of entries
    stats = tracks.groupby('track_id').size().reset_index(name = 'n')

    # filter groups with more than x entries
    stats_retain = stats[stats['n'] >= threshold]
    ntracks_pre = len(stats)
    ntracks_post = len(stats_retain)

    # show histogram of track length
    bins = np.linspace(0, stats['n'].max(), 50)
    plt.hist(stats['n'], bins = bins, label = 'all: ' + str(ntracks_pre))
    plt.hist(stats_retain['n'], bins = bins, label = 'retained: ' + str(ntracks_post))
    plt.axvline(x = threshold, color='r', linestyle='--')
    ax = plt.gca()
    _, ymax = ax.get_ylim()
    ax.text(threshold, ymax, '>' + str(threshold),
            fontsize=14, horizontalalignment='left', verticalalignment='top')
    plt.ylabel('n tracks')
    plt.xlabel('track length (n frames)')
    plt.title('minimum track length (frames),\nfraction retained: ' +
              str(round(ntracks_post/ntracks_pre, 4)))
    plt.legend()
    plt.show()

    return stats_retain['track_id'].values

def track_filter_density(tracks, spots, threshold: int = 2000, radius: int = 50, supplement: bool = False):
    '''
    Filter tracks by spot density feature, 90 percentile per track curation, above threshold is 'too crowded'.
    
    input:
        tracks [pandas]: tracking table
        spots [numpy]: spots table (index ~ label) [T, Y, X]
        threshold: maximum allowed spot density
        radius: density radius calculation (n pixels, maintain to reasonable size, local proximity)

    print:
        histogram of stats

    return:
        track_id [array] to retain
    '''

    # calculate density per spot
    density = calc_point_density_2d(spots[:, 1:], radius = radius)

    # link point density to tracking table
    tracks_feat = pd.merge(
        left = tracks, right = pd.DataFrame({'density': density, 'label': range(len(spots))}),
        how = 'left', on = 'label')

    # track density stats
    stats = tracks_feat.groupby('track_id')['density'].quantile(q = .9).reset_index()

    # select tracks below stat cutoff
    stats_retain = stats[stats['density'] <= threshold]
    ntracks_pre = len(stats)
    ntracks_post = len(stats_retain)

    # show histogram of track density
    bins = np.linspace(0, stats['density'].max(), 50)
    plt.hist(stats['density'], bins = bins, label = 'all: ' + str(ntracks_pre))
    plt.hist(stats_retain['density'], bins = bins, label = 'retained: ' + str(ntracks_post))
    plt.axvline(x = threshold, color='r', linestyle='--')
    ax = plt.gca()
    _, ymax = ax.get_ylim()
    ax.text(threshold, ymax, '<' + str(threshold),
            fontsize=14, horizontalalignment='right', verticalalignment='top')
    plt.ylabel('n tracks')
    plt.xlabel('track density')
    plt.title('spot density per track,\nfraction retained: ' +
              str(round(ntracks_post/ntracks_pre, 4)))
    plt.legend()
    plt.show()

    if supplement:
        # Assuming tracks_feat is your DataFrame and 'frame_y' is the column to group by
        grouped = tracks_feat.groupby('frame_y')['density']
        # Create a boxplot
        plt.figure(figsize=(10, 6))
        boxplot = plt.boxplot([group for name, group in grouped], labels=grouped.groups.keys())
        
        # Add title and axis labels (customize as needed)
        plt.title('Boxplot of Density per frame')
        plt.xlabel('frame number')
        plt.ylabel('Density')

        # Show the plot
        plt.show()

    return stats_retain['track_id'].values

def track_filter_mask(tracks, spots, mask, inside: bool = True, threshold = .1):
    '''
    Keep only tracks that fall inside (or outside) mask area for most of their life.

    Logic gate, tracks inside above threshold frame fraction will be recorded.
    Then depending on inside will be returned, of not returned.
    Cleanly only measure inside mask fraction.

    input:
        tracks [pandas]: tracking table
        spots [numpy]: spots table (index ~ label) [T, Y, X]
        mask [numpy]: 2d boolean mask [Y, X]
        inside: flip logic, inside means retain what is in. inside false: purge what is inside.
        threshold: threshold minimum track residency permitted frequency (.1 = 10% time spent inside mask area)

    print:
        histogram of stats

    return:
        track_id [array] to retain
    '''

    # bind spots to mask
    spots_in_mask = PointToMaskID(spots[:, 1:], mask > 0)

    # link points to tracking table
    tracks_feat = pd.merge(
        left = tracks, right = pd.DataFrame({'mask': spots_in_mask,'label': range(len(spots))}),
        how = 'left', on = 'label')

    # track residency stats
    stats = tracks_feat.groupby('track_id')['mask'].value_counts(normalize = True).unstack(fill_value=0)

    # select tracks above stat cutoff
    stats_retain = stats[stats[True] >= threshold]
    if not inside:
        stats_retain = stats[~np.isin(stats.index, stats_retain.index)]

    ntracks_pre = len(stats)
    ntracks_post = len(stats_retain)

    # show histogram of track residency
    bins = np.linspace(0, 1, 50)
    plt.hist(stats[True], bins = bins, label = 'all: ' + str(ntracks_pre))
    plt.hist(stats_retain[True], bins = bins, label = 'retained: ' + str(ntracks_post))
    plt.axvline(x = threshold, color='r', linestyle='--')
    ax = plt.gca()
    _, ymax = ax.get_ylim()
    if inside:
        ax.text(threshold, ymax, '>' + str(threshold),
                fontsize=14, horizontalalignment='left', verticalalignment='top')
    else:
        ax.text(threshold, ymax, '<' + str(threshold),
                fontsize=14, horizontalalignment='right', verticalalignment='top')
    plt.ylabel('n tracks')
    plt.xlabel('fraction inside')
    plt.title('track residency in mask area,\nfraction retained: ' +
              str(round(ntracks_post/ntracks_pre, 4)))
    plt.legend()
    plt.show()

    return stats_retain.index.values

def track_filter_start_window(tracks, below: bool = False, threshold: int = 0):
    '''
    Filter tracks by when they initiate.
    Default is to retain all tracks.
    Can be used to fetch tracks that start from start of imaging, or only at the end.
    Single boundary gating! (use function twice and intersect output to create a between window).
    
    input:
        tracks [pandas]: tracking table.
        below: keep tracks that initiate before threshold.
        threshold: treshold for retaining tracks based on start frame.

    print:
        histogram of stats

    return:
        track_id [array] to retain
    '''

    # group by track_id and count the number of entries
    stats = tracks.groupby('track_id')['frame_y'].min()

    # filter groups with more than x entries
    if below:
        stats_retain = stats[stats < threshold]
    else:
        stats_retain = stats[stats >= threshold]

    ntracks_pre = len(stats)
    ntracks_post = len(stats_retain)

    # show histogram of track length
    bins = np.linspace(0, stats.max(), 50)
    plt.hist(stats, bins = bins, label = 'all: ' + str(ntracks_pre))
    plt.hist(stats_retain, bins = bins, label = 'retained: ' + str(ntracks_post))
    plt.axvline(x = threshold, color='r', linestyle='--')
    ax = plt.gca()
    _, ymax = ax.get_ylim()
    if below:
        ax.text(threshold, ymax, '<' + str(threshold),
                fontsize=14, horizontalalignment='right', verticalalignment='top')
    else:
        ax.text(threshold, ymax, '>' + str(threshold),
                fontsize=14, horizontalalignment='left', verticalalignment='top')
    plt.ylabel('n tracks')
    plt.xlabel('track starting frame')
    plt.title('start frame threshold,\nfraction retained: ' +
              str(round(ntracks_post/ntracks_pre, 4)))
    plt.legend()
    plt.show()

    return stats_retain.index.values
