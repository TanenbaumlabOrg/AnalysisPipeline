import math
def optimized_grid_shape(num_elements):
    # Calculate the square root to get the closest integer
    sqrt_num = math.sqrt(num_elements)
    # Determine the number of rows and columns
    cols = math.ceil(sqrt_num)
    rows = math.ceil(num_elements / cols)
    return rows, cols

def assign_grid_positions(df, grid_shape):
    num_rows, num_cols = grid_shape
    total_elements = num_rows * num_cols

    # Calculate row and column numbers for each row
    df['Row'] = (df.index // num_cols) % num_rows
    df['Column'] = df.index % num_cols

    return df