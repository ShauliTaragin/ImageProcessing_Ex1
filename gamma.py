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
from ex1_utils import LOAD_GRAY_SCALE
import cv2
import numpy as np

max_num = 255


def adjust_gamma(brightness=0):
    gamma = float(brightness) / 100
    invGamma = 1000
    if gamma != 0:
        invGamma = 1.0 / gamma
    gammaTable = np.array([((i / float(max_num)) ** invGamma) * max_num
                           for i in np.arange(0, max_num + 1)]).astype("uint8")
    img_gamma = cv2.LUT(img_to_gamma, gammaTable)
    cv2.imshow("Gamma Correction", img_gamma)


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global img_to_gamma
    if rep == LOAD_GRAY_SCALE:
        img_to_gamma = cv2.imread(img_path, 2)
    # img is RGB
    else:
        img_to_gamma = cv2.imread(img_path, 1)

    cv2.namedWindow("Gamma Correction")
    trackbar_name = 'Gamma'
    # since the requirement was to represent from 0 to 2 with resolution 0.01 but we use int we will represent from 0-100
    cv2.createTrackbar(trackbar_name, "Gamma Correction", 100, 200, adjust_gamma)
    adjust_gamma(100)
    # when user presses a key we respond(and if need exit the program
    cv2.waitKey()


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
