import numpy as np


def angle_from_points(points):
    """
    Calculates angle of an object from x,y coordinates of two points

    :param points: Array like object of size [4 : num_timepoints] Each column
     is a time point, and the rows are:
     0: Left (x)
     1: Left (y)
     2: Right (x)
     3: Right (y)
     e.g. Ear markers to calculate head angle
    :return: Absolute angle and unwrapped angle (in degrees)
    """

    left_x = points[0]
    left_y = points[1]
    right_x = points[2]
    right_y = points[3]

    absolute_head_angle = np.arctan2((left_x - right_x), (left_y - right_y))
    absolute_head_angle = 180 + absolute_head_angle * 180 / np.pi

    return absolute_head_angle
