"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import cv2

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 209337161


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv2.imread(filename)
    if representation == LOAD_GRAY_SCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return (img - img.min()) / (img.max() - img.min())
    else:  # we should represent in RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (img - img.min()) / (img.max() - img.min())


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    plt.imshow(img)
    plt.gray()  # to change the view of the picture to actual greyscale
    plt.show()  # no need to do this is jupyter


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    YIQ_conversion = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    imgYIQ = np.dot(imgRGB, YIQ_conversion.transpose())
    return imgYIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    YIQ_conversion = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    YIQ_conversion_inverse = np.linalg.inv(YIQ_conversion)
    imgRGB = np.dot(imgYIQ, YIQ_conversion_inverse.transpose())
    return imgRGB


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    # first check if the image is RGB if so convert to YIQ
    shape = imgOrig.shape
    is_RGB = False
    if len(shape) > 2:
        is_RGB = True
        YIQOrig = transformRGB2YIQ(imgOrig)
        imgOrig = YIQOrig[:, :, 0]

    # normalize from 0-1 to 0-255
    imgOrig = cv2.normalize(imgOrig, None, 0, 255, cv2.NORM_MINMAX)
    imgOrig = imgOrig.astype('uint8')

    # first we calculate the original image's histogram
    img_flat = imgOrig.ravel()
    histOrg = np.zeros(256)
    for pix in img_flat:
        histOrg[pix] += 1

    # calculate the cum_sum
    cum_sum = np.zeros_like(histOrg)
    cum_sum[0] = histOrg[0]
    for i in range(1, len(histOrg)):
        cum_sum[i] = histOrg[i] + cum_sum[i - 1]

    # Create look up table
    cum_sum_norm = cum_sum / cum_sum.max()
    LUT = np.floor(cum_sum_norm * 255).astype('uint8')

    # with new picture replace each intensity i with LUT[i]
    imEq = np.zeros_like(imgOrig, dtype=float)
    for pix in range(256):
        imEq[imgOrig == pix] = LUT[pix]

    # normalize and calculate histogram
    imEq = cv2.normalize(imEq, None, 0, 255, cv2.NORM_MINMAX)
    imEq = imEq.astype('uint8')
    imgNew_flat = imEq.ravel()
    histEQ = np.zeros(256)
    for pix in imgNew_flat:
        histEQ[pix] += 1

    # If the picture was RGB we need to return it to RGB form
    if is_RGB is True:
        YIQOrig[:, :, 0] = imEq / 255  # we do this because we need to return an image (0,1)
        imEq = transformYIQ2RGB(YIQOrig)

    return imEq, histOrg, histEQ


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    # start by checking if its RGB and if so convert to YIQ similar to Histogram
    shape = imOrig.shape
    is_RGB = False
    if len(shape) > 2:
        is_RGB = True
        YIQOrig = transformRGB2YIQ(imOrig)
        imOrig = YIQOrig[:, :, 0]

    # initialize the lists we will return
    images = []
    errors = []

    # normalize from 0-1 to 0-255
    imOrig = cv2.normalize(imOrig, None, 0, 255, cv2.NORM_MINMAX)
    imOrig = imOrig.astype('uint8')

    # calculate histogram
    imOrig_flat = imOrig.ravel()
    hist = np.zeros(256)
    for pix in imOrig_flat:
        hist[pix] += 1

    # start quantization process
    # create initial borders
    borders = np.zeros(nQuant + 1, dtype=np.float)
    border_count = 0
    for i in range(nQuant + 1):
        border_count += 255.0 / nQuant
        borders[i] = border_count

    # main loop which we run nIter times
    for i in range(nIter):
        
        for k in range(nQuant):
            intense = hist[borders[k]:borders[k + 1]]
            q = np.average(intense, weights=hist[borders[k]: borders[k + 1] + 1])

    if is_RGB is True:
        YIQOrig[:, :, 0] = imEq / 255  # we do this because we need to return an image (0,1)
        imEq = transformYIQ2RGB(YIQOrig)
