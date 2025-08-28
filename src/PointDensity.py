import numpy as np
import scipy.spatial as spatial

def calc_point_density_2d(points, radius: int = 50):
    """
    Return the density of points in a 2D grid.
    Input:
        points = list of points as numpy array [[x1, y1], ...].
            Shape (n*points, 2 coordinates y and x)
        radius = the radius in which to find neighbor points.
    Output:
        density = a 1D array of point densities.
    """
    tree = spatial.KDTree(np.array(points))

    neighbors = tree.query_ball_tree(tree, radius)
    frequency = np.array([len(i) for i in neighbors])
    density = (frequency/radius)**2

    return density
