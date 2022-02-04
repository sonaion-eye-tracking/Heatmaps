import PIL
import PIL.Image
import numpy as np
import pandas as pd

import sonaion_heatmaps.masks as masks
import sonaion_heatmaps.addition_kernel as addition_kernel
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt
import cv2
from numba import cuda
from tqdm.notebook import tqdm


def __get_mask_function(dimension, **kwargs):
    mask_function = None

    if 'use_rectangle' in kwargs:
        # Check if kwargs['use_rectangle'] is a boolean value
        if not isinstance(kwargs['use_rectangle'], bool):
            raise TypeError('kwargs[\'use_rectangle\'] must be a boolean value')
        # Check if kwargs has key 'rectangle_width' or 'rectangle_height'
        if not ('rectangle_width' in kwargs or 'rectangle_height' in kwargs):
            raise KeyError('kwargs must have key \'rectangle_width\' or \'rectangle_height\'')

        rectangle_width = kwargs['rectangle_width']
        rectangle_height = kwargs['rectangle_height']

        # Check if rectangle_width is a positive integer
        if not isinstance(rectangle_width, int):
            raise TypeError('rectangle_width must be a positive integer')
        if rectangle_width <= 0:
            raise ValueError('rectangle_width must be a positive integer')
        # Check if rectangle_height is a positive integer
        if not isinstance(rectangle_height, int):
            raise TypeError('rectangle_height must be a positive integer')
        if rectangle_height <= 0:
            raise ValueError('rectangle_height must be a positive integer')

        def tmp_mask_function(x, y, delta_t):
            return masks.get_rectangular_mask((x, y), rectangle_width, rectangle_height, dimension, delta_t)

        mask_function = tmp_mask_function

    elif 'use_circle' in kwargs:
        # Check if kwargs['use_circle'] is a boolean value
        if not isinstance(kwargs['use_circle'], bool):
            raise TypeError('kwargs[\'use_circle\'] must be a boolean value')
        # Check if kwargs has key 'circle_radius'
        if not 'circle_radius' in kwargs:
            raise KeyError('kwargs must have key \'circle_radius\'')

        circle_radius = kwargs['circle_radius']

        # Check if circle_radius is a positive integer
        if not isinstance(circle_radius, int):
            raise TypeError('circle_radius must be a positive integer')
        if circle_radius <= 0:
            raise ValueError('circle_radius must be a positive integer')

        def tmp_mask_function(x, y, delta_t):
            return masks.get_circular_mask((x, y), circle_radius, dimension, delta_t)

        mask_function = tmp_mask_function

    elif 'use_eclipse' in kwargs:
        # Check if kwargs['use_eclipse'] is a boolean value
        if not isinstance(kwargs['use_eclipse'], bool):
            raise TypeError('kwargs[\'use_eclipse\'] must be a boolean value')
        # Check if kwargs has key 'eclipse_radius'
        if not 'eclipse_radius' in kwargs:
            raise KeyError('kwargs must have key \'eclipse_radius\'')

        eclipse_width = kwargs['eclipse_width']
        eclipse_height = kwargs['eclipse_height']

        # Check if eclipse_width is a positive integer
        if not isinstance(eclipse_width, int):
            raise TypeError('eclipse_width must be a positive integer')
        if eclipse_width <= 0:
            raise ValueError('eclipse_width must be a positive integer')
        # Check if eclipse_height is a positive integer
        if not isinstance(eclipse_height, int):
            raise TypeError('eclipse_height must be a positive integer')
        if eclipse_height <= 0:
            raise ValueError('eclipse_height must be a positive integer')

        def tmp_mask_function(x, y, delta_t):
            return masks.get_elliptical_mask((x, y), eclipse_width, eclipse_height, dimension, delta_t)

        mask_function = tmp_mask_function

    else:
        raise KeyError('kwargs must have key \'use_rectangle\', \'use_circle\', or \'use_eclipse\'')

    return mask_function

def __get_mask_function_cuda(dimension, **kwargs):
    mask_function = None

    if 'use_rectangle' in kwargs:
        # Check if kwargs['use_rectangle'] is a boolean value
        if not isinstance(kwargs['use_rectangle'], bool):
            raise TypeError('kwargs[\'use_rectangle\'] must be a boolean value')
        # Check if kwargs has key 'rectangle_width' or 'rectangle_height'
        if not ('rectangle_width' in kwargs or 'rectangle_height' in kwargs):
            raise KeyError('kwargs must have key \'rectangle_width\' or \'rectangle_height\'')

        rectangle_width = kwargs['rectangle_width']
        rectangle_height = kwargs['rectangle_height']

        # Check if rectangle_width is a positive integer
        if not isinstance(rectangle_width, int):
            raise TypeError('rectangle_width must be a positive integer')
        if rectangle_width <= 0:
            raise ValueError('rectangle_width must be a positive integer')
        # Check if rectangle_height is a positive integer
        if not isinstance(rectangle_height, int):
            raise TypeError('rectangle_height must be a positive integer')
        if rectangle_height <= 0:
            raise ValueError('rectangle_height must be a positive integer')

        def tmp_mask_function(heatmap, x, y, delta_t, blockspergrid, threadsperblock):
            addition_kernel.rectangular_addition_kernel[blockspergrid, threadsperblock](heatmap, x, y, rectangle_width, rectangle_height, delta_t, dimension[1], dimension[0])

        mask_function = addition_kernel.circular_addition_kernel

    elif 'use_circle' in kwargs:
        # Check if kwargs['use_circle'] is a boolean value
        if not isinstance(kwargs['use_circle'], bool):
            raise TypeError('kwargs[\'use_circle\'] must be a boolean value')
        # Check if kwargs has key 'circle_radius'
        if not 'circle_radius' in kwargs:
            raise KeyError('kwargs must have key \'circle_radius\'')

        circle_radius = kwargs['circle_radius']

        # Check if circle_radius is a positive integer
        if not isinstance(circle_radius, int):
            raise TypeError('circle_radius must be a positive integer')
        if circle_radius <= 0:
            raise ValueError('circle_radius must be a positive integer')

        def tmp_mask_function(heatmap, x, y, delta_t, blockspergrid, threadsperblock):
            addition_kernel.circular_addition_kernel[blockspergrid, threadsperblock](heatmap, x, y, circle_radius, delta_t, dimension[1], dimension[0])

        mask_function = tmp_mask_function

    elif 'use_eclipse' in kwargs:
        # Check if kwargs['use_eclipse'] is a boolean value
        if not isinstance(kwargs['use_eclipse'], bool):
            raise TypeError('kwargs[\'use_eclipse\'] must be a boolean value')
        # Check if kwargs has key 'eclipse_radius'
        if not 'eclipse_radius' in kwargs:
            raise KeyError('kwargs must have key \'eclipse_radius\'')

        eclipse_width = kwargs['eclipse_width']
        eclipse_height = kwargs['eclipse_height']

        # Check if eclipse_width is a positive integer
        if not isinstance(eclipse_width, int):
            raise TypeError('eclipse_width must be a positive integer')
        if eclipse_width <= 0:
            raise ValueError('eclipse_width must be a positive integer')
        # Check if eclipse_height is a positive integer
        if not isinstance(eclipse_height, int):
            raise TypeError('eclipse_height must be a positive integer')
        if eclipse_height <= 0:
            raise ValueError('eclipse_height must be a positive integer')

        def tmp_mask_function(heatmap, x, y, delta_t, blockspergrid, threadsperblock):
            addition_kernel.elliptical_addition_kernel[blockspergrid, threadsperblock](heatmap, x, y, eclipse_width, eclipse_height, delta_t, dimension[1], dimension[0])

        mask_function = tmp_mask_function

    else:
        raise KeyError('kwargs must have key \'use_rectangle\', \'use_circle\', or \'use_eclipse\'')

    return mask_function


def heatmask(coordinate_array, validity_array, time_array, image, **kwargs):
    """

    :param coordinate_array:    An Array with coordinates in image coordinates
    :param validity_array:      An Array with validity of each coordinate
    :param time_array:          An Array with time of each coordinate
    :param image:               An Image object as PIL object
    :param kwargs:              Keyword arguments
    :kwargs use_rectangle:      If True, draw rectangle around each coordinate
    :kwargs rectangle_width:    Width of rectangle
    :kwargs rectangle_height:   Height of rectangle

    :kwargs use_circle:         If True, draw circle around each coordinate
    :kwargs circle_radius:      Radius of circle

    :kwargs use_eclipse:        If True, draw line between each coordinate
    :kwargs eclipse_width:      Width of eclipse
    :kwargs eclipse_height:     Height of eclipse

    :kwargs blurr:              If True blurr the image using a gaussian filter
    :kwargs blurr_sigma:        Sigma of gaussian filter

    :kwargs use_cuda:           If True, use cuda to accelerate computation
    :kwargs threads_per_block:  Number of threads per block (tuple of 2 integers)

    :return:                    A heatmap image as PIL object
    """
    # Check if image is a PIL object
    if not isinstance(image, PIL.Image.Image):
        raise TypeError('image must be a PIL object')
    # Check if coordinate_array is a numpy array
    if not isinstance(coordinate_array, np.ndarray):
        raise TypeError('coordinate_array must be a numpy array')
    # Check if validity_array is a numpy array
    if not isinstance(validity_array, np.ndarray):
        raise TypeError('validity_array must be a numpy array')
    # Check if time_array is a numpy array
    if not isinstance(time_array, np.ndarray):
        raise TypeError('time_array must be a numpy array')
    # Check if coordinate_array and validity_array have same length
    if len(coordinate_array) != len(validity_array):
        raise ValueError('coordinate_array and validity_array must have same length')
    # Check if coordinate_array and time_array have same length
    if len(coordinate_array) != len(time_array):
        raise ValueError('coordinate_array and time_array must have same length')
    # Check if coordinate_array is a numpy array of shape (n,2)
    if coordinate_array.shape != (len(coordinate_array), 2):
        raise ValueError('coordinate_array must be a numpy array of shape (n,2)')
    # Check if validity_array is a numpy array of shape (n,)
    if validity_array.shape != (len(validity_array),):
        raise ValueError('validity_array must be a numpy array of shape (n,)')
    # Check if time_array is a numpy array of shape (n,)
    if time_array.shape != (len(time_array),):
        raise ValueError('time_array must be a numpy array of shape (n,)')
    # Check if validity_array is a numpy array of boolean values
    if not np.issubdtype(validity_array.dtype, np.bool_):
        raise ValueError('validity_array must be a numpy array of boolean values')
    # Check if time_array is a numpy array of float values
    if not np.issubdtype(time_array.dtype, np.float_):
        raise ValueError('time_array must be a numpy array of float values')
    # Check if image is a PIL object
    if not isinstance(image, PIL.Image.Image):
        raise TypeError('image must be a PIL object')

    # Check if kwargs is a dictionary
    if not isinstance(kwargs, dict):
        raise TypeError('kwargs must be a dictionary')

    threads_per_block = None
    blockspergrid = None
    mask_function = None
    if 'use_cuda' in kwargs:
        if 'threads_per_block' not in kwargs:
            raise ValueError('threads_per_block must be specified if use_cuda is set')
        if not isinstance(kwargs['threads_per_block'], tuple):
            raise TypeError('threads_per_block must be a tuple')
        if len(kwargs['threads_per_block']) != 2:
            raise ValueError('threads_per_block must be a tuple of 2 integers')
        if not isinstance(kwargs['threads_per_block'][0], int):
            raise TypeError('threads_per_block[0] must be an integer')
        if not isinstance(kwargs['threads_per_block'][1], int):
            raise TypeError('threads_per_block[1] must be an integer')
        threads_per_block = kwargs['threads_per_block']
        width, height = image.size
        blockspergrid = (int(np.ceil(height / threads_per_block[1])), int(np.ceil(width / threads_per_block[0])))
        mask_function = __get_mask_function_cuda(image.size, **kwargs)
    else:
        mask_function = __get_mask_function(image.size, **kwargs)

    # Create a new numpy array with same shape as image
    new_image_array = np.zeros((image.size[1], image.size[0]), dtype=np.float32)
    if 'use_cuda' in kwargs:
        new_image_array = cuda.to_device(new_image_array)

    # Iterate over all coordinates
    for (x, y), valid, time in zip(coordinate_array, validity_array, time_array):
        # Check if coordinate is valid
        if valid:
            if 'use_cuda' in kwargs:
                mask_function(new_image_array, x, y, time, blockspergrid, threads_per_block)
            else:
                current_heat = mask_function(x, y, float(time))
                new_image_array += current_heat

    if 'use_cuda' in kwargs:
        new_image_array = new_image_array.copy_to_host()

    # Check if blurr is in kwargs
    if 'blurr' in kwargs:
        # Check if blurr is a boolean
        if not isinstance(kwargs['blurr'], bool):
            raise TypeError('blurr must be a boolean')
        # Check if blurr is True
        if kwargs['blurr']:
            # Check if blurr_sigma is in kwargs
            if 'blurr_sigma' in kwargs:
                # Check if blurr_sigma is a float
                if not isinstance(kwargs['blurr_sigma'], float):
                    raise TypeError('blurr_sigma must be a float')
                # Check if blurr_sigma is positive
                if kwargs['blurr_sigma'] <= 0:
                    raise ValueError('blurr_sigma must be positive')
                # Blurr the image
                new_image_array = scipy.ndimage.gaussian_filter(new_image_array, kwargs['blurr_sigma'])

    return new_image_array


def heatmap_image(coordinate_array, validity_array, time_array, image, **kwargs):
    """

    :param coordinate_array:    An Array with coordinates in image coordinates
    :param validity_array:      An Array with validity of each coordinate
    :param time_array:          An Array with time of each coordinate
    :param image:               An Image object as PIL object
    :param kwargs:              Keyword arguments
    :kwargs use_rectangle:      If True, draw rectangle around each coordinate
    :kwargs rectangle_width:    Width of rectangle
    :kwargs rectangle_height:   Height of rectangle

    :kwargs use_circle:         If True, draw circle around each coordinate
    :kwargs circle_radius:      Radius of circle

    :kwargs use_eclipse:        If True, draw line between each coordinate
    :kwargs eclipse_width:      Width of eclipse
    :kwargs eclipse_height:     Height of eclipse

    :kwargs blurr:              If True blurr the image using a gaussian filter
    :kwargs blurr_sigma:        Sigma of gaussian filter

    :kwargs key_out_threshold:  Threshold for key out in percent of max heat

    :kwargs image_opacity:      Opacity of image in the heatmap

    :kwargs use_cuda:           If True, use cuda to accelerate computation
    :kwargs threads_per_block:  Number of threads per block (tuple of 2 integers)

    :return:                    A Tuple with:
                                    - A numpy array which contains the heatmap image
                                    - A numpy array which contains the heatmask normalized to 1.0
                                    - The maximum heat value before the normalization
    """

    cmap = plt.cm.get_cmap('jet')
    current_heatmask = heatmask(coordinate_array, validity_array, time_array, image, **kwargs)
    max_value = np.max(current_heatmask)
    current_heatmask = current_heatmask / max_value

    # Check if key_out_threshold is in kwargs
    key_out_threshold = 0.0
    if 'key_out_threshold' in kwargs:
        # Check if key_out_threshold is a float
        if not isinstance(kwargs['key_out_threshold'], float):
            raise TypeError('key_out_threshold must be a float')
        # Check if key_out_threshold is between 0 and 1
        if kwargs['key_out_threshold'] < 0 or kwargs['key_out_threshold'] > 1:
            raise ValueError('key_out_threshold must be between 0 and 1')
        key_out_threshold = kwargs['key_out_threshold']

    key_out_mask = current_heatmask >= key_out_threshold
    current_heatmask = current_heatmask * 255
    current_heatmask = current_heatmask.astype(np.uint8)
    current_heatmask = cmap(current_heatmask)
    current_heatmask = current_heatmask * 255
    current_heatmask = current_heatmask.astype(np.uint8)

    # Check if image_opacity is in kwargs
    image_opacity = 0.5
    if 'image_opacity' in kwargs:
        # Check if image_opacity is a float
        if not isinstance(kwargs['image_opacity'], float):
            raise TypeError('image_opacity must be a float')
        # Check if image_opacity is between 0 and 1
        if kwargs['image_opacity'] < 0 or kwargs['image_opacity'] > 1:
            raise ValueError('image_opacity must be between 0 and 1')
        image_opacity = kwargs['image_opacity']

    hmap = np.array(image)
    hmap[key_out_mask] = image_opacity * hmap[key_out_mask] + (1.0 - image_opacity) * current_heatmask[key_out_mask]
    return hmap, current_heatmask, max_value


def heatmap_video(coordinate_array, validity_array, time_array, video_capture, **kwargs):
    """
    :param coordinate_array:        An Array with coordinates in image coordinates
    :param validity_array:          An Array with validity of each coordinate
    :param time_array:              An Array with time of each coordinate in ms
    :param video_capture:           An Image object as PIL object
    :param kwargs:                  Keyword arguments
    :kwargs use_rectangle:          If True, draw rectangle around each coordinate
    :kwargs rectangle_width:        Width of rectangle
    :kwargs rectangle_height:       Height of rectangle

    :kwargs use_circle:             If True, draw circle around each coordinate
    :kwargs circle_radius:          Radius of circle

    :kwargs use_eclipse:            If True, draw line between each coordinate
    :kwargs eclipse_width:          Width of eclipse
    :kwargs eclipse_height:         Height of eclipse

    :kwargs blurr:                  If True blurr the image using a gaussian filter
    :kwargs blurr_sigma:            Sigma of gaussian filter

    :kwargs key_out_threshold:      Threshold for key out in percent of max heat

    :kwargs image_opacity:          Opacity of image in the heatmap

    :kwargs time_range:             Time range into the past for the heatmap in ms
    :kwargs fps:                    Frames per second of the video
    :kwargs fourcc:                 Fourcc of the video
    :kwargs file_name:              Name of the output file
    :kwargs tqdm:                   If True, show a tqdm progress bar
    :kwargs skip_frames:            Number of frames to skip between each frame

    :kwargs use_cuda:               If True, use cuda to accelerate computation
    :kwargs threads_per_block:      Number of threads per block (tuple of 2 integers)

    :return:                        True if successful, False if not
    """

    # Check if coordinate_array is a numpy array
    if not isinstance(coordinate_array, np.ndarray):
        raise TypeError("coordinate_array must be a numpy array")
    # Check if validity_array is a numpy array
    if not isinstance(validity_array, np.ndarray):
        raise TypeError("validity_array must be a numpy array")
    # Check if time_array is a numpy array
    if not isinstance(time_array, np.ndarray):
        raise TypeError("time_array must be a numpy array")
    # Check if coordinate_array and validity_array have the same length
    if len(coordinate_array) != len(validity_array):
        raise ValueError("coordinate_array and validity_array must have the same length")
    # Check if coordinate_array and time_array have the same length
    if len(coordinate_array) != len(time_array):
        raise ValueError("coordinate_array and time_array must have the same length")
    # Check if coordinate_array is a numpy array with shape (n, 2)
    if coordinate_array.shape != (len(coordinate_array), 2):
        raise ValueError("coordinate_array must be a numpy array with shape (n, 2)")
    # Check if validity_array is a numpy array with shape (n,)
    if validity_array.shape != (len(validity_array),):
        raise ValueError("validity_array must be a numpy array with shape (n,)")
    # Check if time_array is a numpy array with shape (n,)
    if time_array.shape != (len(time_array),):
        raise ValueError("time_array must be a numpy array with shape (n,)")
    # Check if validity_array is a numpy array of boolean values
    if not np.issubdtype(validity_array.dtype, np.bool_):
        raise ValueError("validity_array must be a numpy array of boolean values")
    # Check if time_array is a numpy array of float values
    if not np.issubdtype(time_array.dtype, np.float):
        raise ValueError("time_array must be a numpy array of float values")
    # Check if video_capture is a opencv object
    if not isinstance(video_capture, cv2.VideoCapture):
        raise TypeError("video_capture must be a opencv VideoCapture object")

    # Check if kwargs is a dictionary
    if not isinstance(kwargs, dict):
        raise TypeError("kwargs must be a dictionary")

    # Check if key_out_threshold is in kwargs
    key_out_threshold = 0.0
    if 'key_out_threshold' in kwargs:
        # Check if key_out_threshold is a float
        if not isinstance(kwargs['key_out_threshold'], float):
            raise TypeError('key_out_threshold must be a float')
        # Check if key_out_threshold is between 0 and 1
        if kwargs['key_out_threshold'] < 0 or kwargs['key_out_threshold'] > 1:
            raise ValueError('key_out_threshold must be between 0 and 1')
        key_out_threshold = kwargs['key_out_threshold']

    # Check if image_opacity is in kwargs
    image_opacity = 0.5
    if 'image_opacity' in kwargs:
        # Check if image_opacity is a float
        if not isinstance(kwargs['image_opacity'], float):
            raise TypeError('image_opacity must be a float')
        # Check if image_opacity is between 0 and 1
        if kwargs['image_opacity'] < 0 or kwargs['image_opacity'] > 1:
            raise ValueError('image_opacity must be between 0 and 1')
        image_opacity = kwargs['image_opacity']

    dimension = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), (video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    dt = (time_array[1] - time_array[0]) / 1000.0
    df = pd.DataFrame({
        "x": coordinate_array[:, 0],
        "y": coordinate_array[:, 1],
        "validity": validity_array,
        "time": time_array})

    # Create a new numpy array with same shape as image
    destin_fps = kwargs["fps"] if "fps" in kwargs else video_capture.get(cv2.CAP_PROP_FPS)
    skip_frames = kwargs["skip_frames"] if "skip_frames" in kwargs else int(1.0 / (destin_fps * dt))
    file_name = kwargs["file_name"] if "file_name" in kwargs else "./heatmap.mp4"
    fourcc = kwargs["fourcc"] if "fourcc" in kwargs else cv2.VideoWriter_fourcc(*"mp4v")
    use_tqdm = kwargs["tqdm"] if "tqdm" in kwargs else False
    time_range = kwargs["time_range"] if "time_range" in kwargs else time_array[-1]
    # source_fps =  video_capture.get(cv2.CAP_PROP_FPS)
    # frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    cmap = plt.cm.get_cmap(kwargs["cmap"] if "cmap" in kwargs else "jet")
    monitor_size = (int(dimension[1]), int(dimension[0]))
    video_output = cv2.VideoWriter(file_name, fourcc, destin_fps, monitor_size)

    for idx in (tqdm(range(len(df)), total=len(df)) if use_tqdm else df.iterrows()):
        if idx % skip_frames != 0:
            continue
        if not video_output.isOpened():
            raise ValueError("Video output could not be opened")
        current_time_stamp = df.iloc[idx]["time"]
        start_time = max(0, current_time_stamp - time_range)
        df_tmp = df[(df["time"] >= start_time) & (df["time"] <= current_time_stamp)]
        x_values = df_tmp["x"].values
        y_values = df_tmp["y"].values
        current_coordinates = np.stack((x_values, y_values), axis=1)
        current_validity = df_tmp["validity"].values
        current_time = df_tmp["time"].values

        video_capture.set(cv2.CAP_PROP_POS_MSEC, current_time_stamp / 1000.0)
        ret, np_frame = video_capture.read()
        if not ret:
            raise ValueError("Video could not be read")

        frame = PIL.Image.fromarray(np_frame)
        current_heatmask = heatmask(current_coordinates, current_validity, current_time, frame, **kwargs)

        max_value = np.max(current_heatmask)
        if max_value == 0.0:
            max_value = 1.0
        current_heatmask = current_heatmask / max_value

        # Check if key_out_threshold is in kwargs
        key_out_mask = current_heatmask >= key_out_threshold
        current_heatmask = current_heatmask * 255
        current_heatmask = current_heatmask.astype(np.uint8)
        current_heatmask = cmap(current_heatmask)
        current_heatmask = current_heatmask * 255
        current_heatmask = current_heatmask.astype(np.uint8)
        current_heatmask = cv2.cvtColor(current_heatmask, cv2.COLOR_RGB2BGR)

        hmap = np_frame
        current_heatmask = current_heatmask[:, :, :3]
        hmap[key_out_mask] = image_opacity * hmap[key_out_mask] + (1.0 - image_opacity) * current_heatmask[key_out_mask]
        cv2.imshow("Video", hmap)
        if cv2.waitKey(1) == ord("q"):
            break
        video_output.write(hmap)

    video_output.release()
    cv2.destroyAllWindows()
    return True
