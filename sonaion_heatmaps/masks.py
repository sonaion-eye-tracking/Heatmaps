import numpy as np
from skimage.draw import ellipse, disk, rectangle


def get_elliptical_mask(center, x_radius, y_radius, dimension, value=1.0):
    """
    :param center:      Center of the Ellipse (x,y)
    :param x_radius:    The x radius of the Ellipse
    :param y_radius:    The y radius of the Ellipse
    :param dimension:   The dimension in (width, height)
    :param value:       The value set where the data is "warm"
    :return:            A Numpy Array of (height, width), where every value which is not in the mask is 0 and every other value is "value"
    """

    x_dim = max(center[0] + 2 * x_radius + 1, dimension[0])
    y_dim = max(center[1] + 2 * x_radius + 1, dimension[1])
    mask = np.zeros((y_dim, x_dim), dtype=np.float32)
    rr, cc = ellipse(center[1], center[0], y_radius, x_radius, shape=(dimension[1], dimension[0]))
    mask[rr, cc] = value
    mask = mask[:dimension[1], :dimension[0]]
    return mask


def get_circular_mask(center, radius, dimension, value=1.0):
    """
    :param center:      Center of the Circle (x,y)
    :param radius:      The radius of the Circle
    :param dimension:   The dimension in (width, height)
    :param value:       The value set where the data is "warm"
    :return:            A Numpy Array of (height, width), where every value which is not in the mask is 0 and every other value is "value"
    """

    x_dim = max(center[0] + 2 * radius + 1, dimension[0])
    y_dim = max(center[1] + 2 * radius + 1, dimension[1])
    mask = np.zeros((y_dim, x_dim), dtype=np.float32)
    rr, cc = disk((center[1], center[0]), radius, shape=(dimension[1], dimension[0]))
    mask[rr, cc] = value
    mask = mask[:dimension[1], :dimension[0]]
    return mask


def get_rectangular_mask(center, width, height, dimension, value=1.0):
    """

    :param center:      Center of the Circle (x,y)
    :param width:       The width of the rectangle
    :param height:      The height of the rectangle
    :param dimension:   The dimension in (width, height)
    :param value:       The value set where the data is "warm"
    :return:            A Numpy Array of (height, width), where every value which is not in the mask is 0 and every other value is "value"
    """

    x_dim = max(center[0] + 2 * width + 1, dimension[0])
    y_dim = max(center[1] + 2 * height + 1, dimension[1])
    mask = np.zeros((y_dim, x_dim), dtype=np.float32)
    top_left = (int(center[1] - height / 2), int(center[0] - width / 2))
    rr, cc = rectangle(top_left, extent=(height, width), shape=(dimension[1], dimension[0]))
    # Remove every negative value
    rr = rr[rr >= 0]
    cc = cc[cc >= 0]
    mask[rr, cc] = value
    mask = mask[:dimension[1], :dimension[0]]
    return mask
