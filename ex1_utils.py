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
    img_copy = imOrig
    if len(shape) > 2:
        is_RGB = True
        YIQOrig = transformRGB2YIQ(img_copy)
        img_copy = YIQOrig[:, :, 0]

    # initialize the lists we will return
    images = []
    errors = []

    # normalize from 0-1 to 0-255
    img_copy = cv2.normalize(img_copy, None, 0, 255, cv2.NORM_MINMAX)

    # calculate histogram
    imOrig_flat = img_copy.ravel().astype(int)
    hist = np.zeros(256)
    for pix in imOrig_flat:
        hist[pix] += 1

    # start quantization process
    # create initial borders
    borders = np.zeros(nQuant + 1, dtype=np.int)
    border_count = 0
    for i in range(nQuant + 1):
        borders[i] = i * (255.0 / nQuant)
    borders[-1] = 256
    # main loop which we run nIter times
    for i in range(nIter):
        # create array of q's .Each entry will be the appropriate q we calculate between borders z.
        q_array = np.zeros(nQuant, dtype=np.int)
        # calculate each q according to formula
        for k in range(nQuant):
            q = hist[borders[k]:borders[k + 1]]
            rng = np.arange(int(borders[k]), int(borders[k + 1]))
            q_array[k] = (rng * q).sum() / (q.sum()).astype(int)
        # recalculate borders
        z_0 = borders[0]
        z_last = borders[-1]
        borders = np.zeros_like(borders)
        for j in range(1, nQuant):
            borders[j] = (q_array[j - 1] + q_array[j]) / 2
        borders[0] = z_0
        borders[-1] = z_last

        # recolor image
        temp_img = np.zeros_like(img_copy)
        for h in range(nQuant):
            z_temp = borders[h]
            temp_img[img_copy > z_temp] = q_array[h]

        images.append(temp_img)

        # calculate and add MSE
        errors.append(np.sqrt((img_copy - temp_img) ** 2).mean())
        # if we converge then break as written in instructions
        if len(errors) > 1 and abs(errors[-2] - errors[-1]) < 0.001:
            break
    # if picture is RGB return it from YIQ to RGB before adding it to the list of images.
    if is_RGB is True:
        for i in range(len(images)):
            YIQOrig[:, :, 0] = images[i] / 255  # we do this because we need to return an image (0,1)
            images[i] = transformYIQ2RGB(YIQOrig)
            images[i][images[i] > 1] = 1
            images[i][images[i] < 0] = 0
    return images, errors
