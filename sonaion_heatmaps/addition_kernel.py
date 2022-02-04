from numba import cuda

@cuda.jit
def circular_addition_kernel(heatmask, center_x, center_y, radius, value, max_height, max_width):
    """

    :param heatmask:    A cuda Array of type float32
    :param center_x:    The x-coordinate of the center of the circle (integer)
    :param center_y:    The y-coordinate of the center of the circle (integer)
    :param radius:      The radius of the circle (integer)
    :param value:       The value to be added when point is inside the circle (flaot)
    :param max_height:  The height of the heatmask
    :param max_width:   The width of the heatmask
    """
    y, x = cuda.grid(2)
    if center_x - radius <= x <= center_x + radius and center_y - radius <= y <= center_y + radius:
        if y < max_height and x < max_width:
            distance = int(((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5)
            if distance <= radius:
                heatmask[y, x] += value

@cuda.jit
def rectangular_addition_kernel(heatmask, center_x, center_y, width, height, value, max_height, max_width):
    """

    :param heatmask:    A cuda Array of type float32
    :param center_x:    The x-coordinate of the center of the rectangle (integer)
    :param center_y:    The y-coordinate of the center of the rectangle (integer)
    :param width:       The width of the rectangle (integer)
    :param height:      The height of the rectangle (integer)
    :param value:       The value to be added when point is inside the rectangle (float)
    :param max_height:  The height of the heatmask
    :param max_width:   The width of the heatmask
    """
    y, x = cuda.grid(2)
    if center_x - width/2 <= x <= center_x + width/2 and center_y - height/2 <= y <= center_y + height/2:
        if y < max_height and x < max_width:

            heatmask[y, x] += value

@cuda.jit
def elliptical_addition_kernel(heatmask, center_x, center_y, x_radius, y_radius, value, max_height, max_width):
    """

    :param heatmask:    A cuda Array of type float32
    :param center_x:    The x-coordinate of the center of the ellipse (integer)
    :param center_y:    The y-coordinate of the center of the ellipse (integer)
    :param x_radius:    The x-radius of the ellipse (integer)
    :param y_radius:    The y-radius of the ellipse (integer)
    :param value:       The value to be added when point is inside the ellipse (float)
    :param max_height:  The height of the heatmask
    :param max_width:   The width of the heatmask
    """
    y, x = cuda.grid(2)
    if center_x - x_radius <= x <= center_x + x_radius and center_y - y_radius <= y <= center_y + y_radius:
        if y < max_height and x < max_width:
            distance = int(((x - center_x) ** 2 / x_radius ** 2 + (y - center_y) ** 2 / y_radius ** 2) ** 0.5)
            if distance <= 1:
                heatmask[y, x] += value


