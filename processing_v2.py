import math as math
import numpy as np
from matplotlib import pyplot as plt

from copy import copy

import cv2 as cv

""" paths definition """
filenames = [ './Banque/piece_' + str(i) + '.png' for i in range(1, 11) ]


""" get centroid of piece """
def get_centroid(path):
    # read image from file
    gray = np.float32(cv.imread(path, 0))
    _, thresh = cv.threshold(gray, 127, 255, 0)
    
    # calculate moments and centroid coordinates of binary image
    M = cv.moments(thresh)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # return centroid
    return (cX, cY)


""" Canny edge detector """
def get_canny_img(path):
    # read image from file
    gray = cv.imread(path, 0)
    edges = cv.Canny(gray, 100, 255)

    # return Canny image
    return edges


""" plot images """
def plot_all_images(path, centroid, canny):
    # generate images
    image = generate_image(path)
    cannyWithCentroid = generate_centroid_image(canny, centroid)

    # plot using subplots
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap = 'gray')
    plt.title('Original')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(1, 2, 2)
    plt.imshow(cannyWithCentroid, cmap = 'gray')
    plt.title('Canny + centroid')
    plt.xticks([]), plt.yticks([])
    
    plt.show()


""" generate image method """
def generate_image(path):
    # return image
    return cv.imread(path)


""" generate centroid image method """
def generate_centroid_image(img, centroid):
    # add all points to basic image
    cX, cY = centroid
    for (a,b) in [ (cX + i, cY + j) for i in range(-4,5) for j in range(-4,5) ]:
        # replace on image
        img[b, a] = 255 # blanc
    
    # return image
    return img


""" main part of file """
if __name__ == '__main__':
    for fname in filenames:
        centroid = get_centroid(fname)
        canny = get_canny_img(fname)
        plot_all_images(fname, centroid, canny)


