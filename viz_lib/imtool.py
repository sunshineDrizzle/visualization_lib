import os
import numpy as np

from PIL import Image


def get_imlist(path, suffix):
    """
    get files' path which end with suffix from the directory specified by path
    :param path: the directory to find files
    :type path: str
    :param suffix: file suffix
    :type suffix: str
    :return:
    imlist: image file list
    """
    imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(suffix)]
    return imlist


def arr_resize(arr, size):
    """
    resize image array by using PIL
    :param arr: image array
    :type arr: np.ndarray
    :param size: the size for new array after resizing
    :type size: tuple
    :return:
    new_arr: new array after resizing
    """
    pil_im = Image.fromarray(arr)
    new_arr = np.array(pil_im.resize(size))
    return new_arr


def histeq(arr, nbr_bins=256):
    """
    do histogram equalization for the image array
    :param arr: image array
    :type: arr: np.ndarray
    :param nbr_bins:
    :returns:
    new_arr: new image array after histogram equalization
    cdf: cumulative distribution function
    """
    imhist, bins = np.histogram(arr.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalized

    new_arr = np.interp(arr.flatten(), bins[:-1], cdf)
    new_arr = new_arr.reshape(arr.shape)
    return new_arr, cdf
