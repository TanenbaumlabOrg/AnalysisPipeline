def PointToMaskID(points, mask):
    '''
    Inputs:
    - points X, Y
    - mask (single mask image)

    Return:
    - array of mask ID, len == rows(points)
    '''
    # store mask size
    image_size_x = mask.shape[1]
    image_size_y = mask.shape[0]

    arr = []
    for point in points:
        x_coordinate_point = int(round(point[1]))
        if x_coordinate_point == image_size_x:
            x_coordinate_point = x_coordinate_point -1
        y_coordinate_point = int(round(point[0]))
        if y_coordinate_point == image_size_y:
            y_coordinate_point = y_coordinate_point -1
        mask_id = mask[y_coordinate_point, x_coordinate_point]
        arr.append(mask_id)
    return arr
