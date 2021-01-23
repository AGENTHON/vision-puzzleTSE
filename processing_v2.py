import math as math
import numpy as np
from matplotlib import pyplot as plt

from copy import copy

import cv2 as cv

""" paths definition """
filenames = [ './Banque/piece_' + str(i) + '.png' for i in range(1, 11) ]


""" get all mathematical elements from image """
def get_all_elements(img):
    # convert to gray scale then binary
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # calculate moments and centroid coordinates of image
    M = cv.moments(gray)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    centroided = img.copy()
    cv.circle(centroided, (cX,cY), 5, (255,0,0), -1)
    
    # compute Shi-Tomasi detection on standard image
    shiTomGray = gray.copy()
    shiTomGray = cv.cvtColor(gray.copy(), cv.COLOR_GRAY2RGB)
    
    corners = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)
    corners = np.int0(corners)
    
    for i in corners:
        x, y = i.ravel()
        cv.circle(shiTomGray, (x,y), 5, (255,0,0), -1)

    # get Canny image
    canny = cv.Canny(gray, 100, 255)

    # compute Shi-Tomasi detection on Canny image
    shiTomCanny = cv.cvtColor(canny.copy(), cv.COLOR_GRAY2RGB)
    
    corners = cv.goodFeaturesToTrack(canny, 25, 0.01, 10)
    corners = np.int0(corners)
    
    for i in corners:
        x, y = i.ravel()
        cv.circle(shiTomCanny, (x,y), 5, (255,0,0), -1)

    # return elements
    return centroided, shiTomGray, shiTomCanny


""" plot images """
def plot_all_images(img, centroided, shiTomGray, shiTomCanny):
    # plot using subplots
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(2, 2, 2)
    plt.imshow(centroided)
    plt.title('w/ centroid')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 3)
    plt.imshow(shiTomGray)
    plt.title('Shi-Tomasi on gray')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 4)
    plt.imshow(shiTomCanny)
    plt.title('Shi-Tomasi on Canny')
    plt.xticks([]), plt.yticks([])
    
    plt.show()


""" main part of file """
if __name__ == '__main__':
    for fname in filenames:
        # original
        img = cv.imread(fname)

        # modified
        centroided, shiTomGray, shiTomCanny = get_all_elements(img)

        # show everything
        plot_all_images(img, centroided, shiTomGray, shiTomCanny)


