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

    # Find contours
    contours, _ = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    (x,y), (MA,ma), angle = cv.fitEllipse(cnt)
    
    kernel = np.ones((5,5), np.uint8)
    img = cv.line(img, (int(x), int(y)), (int(x+MA*math.cos(math.pi*angle/180)), int(y+MA*math.sin(math.pi*angle/180))), (1.0,0,0), 2)
    img = cv.line(img, (int(x), int(y)), (int(x+ma*math.cos(math.pi*angle/180+math.pi/2)), int(y+ma*math.sin(math.pi*angle/180+math.pi/2))), (0,1.0,0), 2)
    
    # Find the convex hull object for each contour
    contours, _ = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    hull_list = []
    for i in range(len(contours)):
        hull = cv.convexHull(contours[i])
        hull_list.append(hull)
    
    # Draw contours + hull results
    convex = np.zeros((canny.shape[0], canny.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        colorContour = (255, 0, 0)
        colorEdge = (255, 255, 255)
        cv.drawContours(convex, contours, i, colorEdge)
        cv.drawContours(convex, hull_list, i, colorContour)

    # minimal area bounding box
    cnt = contours[0]
    
    box = cv.boxPoints(cv.minAreaRect(cnt))
    box = np.int0(box)
    minArea = cv.drawContours(img.copy(), [box], 0, (255, 255, 0), 2)

    # compute Shi-Tomasi detection on Canny image
    shiTomCanny = cv.cvtColor(canny.copy(), cv.COLOR_GRAY2RGB)
    
    corners = cv.goodFeaturesToTrack(canny, 25, 0.01, 120)
    corners = np.int0(corners)
    
    for i in corners:
        x, y = i.ravel()
        cv.circle(shiTomCanny, (x,y), 5, (255,0,0), -1)

    # return elements
    return centroided, convex, minArea, shiTomGray, shiTomCanny



""" plot images """
def plot_all_images(img, centroided, convex, minArea, shiTomGray, shiTomCanny):
    # plot using subplots
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(2, 3, 2)
    plt.imshow(centroided)
    plt.title('w/ centroid')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 3)
    plt.imshow(convex)
    plt.title('convex hull')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 4)
    plt.imshow(minArea)
    plt.title('minimal area bb')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 5)
    plt.imshow(shiTomGray)
    plt.title('Shi-Tomasi on gray')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 6)
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
        centroided, convex, minArea, shiTomGray, shiTomCanny = get_all_elements(img)

        # show everything
        plot_all_images(img, centroided, convex, minArea, shiTomGray, shiTomCanny)


