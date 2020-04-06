import numpy as np


def calculate_speed(x, y, conversion_factor=None, pad="zero"):
    """
    From a series of x & y positions, calculate the instantaneous speed
    :param x: Array like series of x positions
    :param y: Array like series of y positions
    :param conversion_factor: Pixel per frame (in m/s). e.g. if an object moves
    1 pixel per frame, and each pixel is 0.2cm with a camera rate of 40Hz,
    then this value will be [0.002 * 40 = 0.08], i.e. 8cm per second
    :param pad: How to ensure the speed array has the same length as the input
    arrays.
        - If pad = "zero", a value of 0 will be appended at the start
        - If pad = "start", the first value will be repeated for the first two
        elements of the array
        - If pad = "end" the last value will be repeated for the last two
        elements of the array
        - If pad = "None", nothing will be done, and an array of
        len(input) -1 will be returned
    :return: Array of speed values
    """
    assert len(x) == len(y)
    speed = np.hypot(np.diff(x), np.diff(y))

    if conversion_factor is not None:
        speed = speed * conversion_factor

    if pad == "zero":
        speed = np.append(np.array(0), speed)
    elif pad == "start":
        speed = np.append(np.array(0), speed)
    elif pad == "end":
        speed = np.append(speed, speed[-1])
    elif pad is None:
        pass
    else:
        raise ValueError(
            f"Pad type: '{pad}' is not recognised. Please use "
            f"'zero', 'start', 'end' or 'None'."
        )
    return speed
