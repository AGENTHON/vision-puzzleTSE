import math as math
import numpy as np
from matplotlib import pyplot as plt

from copy import copy

import cv2 as cv


""" paths definition """
filenames = [ './Banque/piece_' + str(i) + '.png' for i in range(1, 7) ]


""" euclidean distance """
def euclidean_distance(x1, y1, x2, y2):
    # euclidean distance formula
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))


""" select corners by maximum of detector """
def select_corners(points):
    # get 4 most likely
    corners = points[:4]

    # compute centroid of corners
    centroid = ( sum([c[0] for c in corners]) / len(corners), sum([c[1] for c in corners]) / len(corners) )

    # get radial angle of each corner
    radial = [ None for i in range(4) ]
    for i in range(len(corners)):
        # get point coordinates
        (x,y) = corners[i]
        (a,b) = centroid

        # radial angle formula
        radial[i] = - math.degrees(math.atan2(y - b, x - a)) % 360
    
    # sort corners by angle
    radial, corners = (list(t) for t in zip(*sorted(zip(radial, corners))))
    
    # return clockwise corners
    return corners


""" use points near the center of each edge to guess the edge's shape """
def guess_edge_shape(img, corners):
    # init variables
    edge_points, string = [] , ""

    # loop through corners
    for i in range(len(corners)):
        (x1, y1) , (x2, y2) = corners[i], corners[(i+1) % len(corners)]
        percent = 0.12
        dx, dy = x2 - x1 , y2 - y1
        mx, my = (x1 + x2) // 2 , (y1 + y2) // 2

        # get correct line direction
        if abs(np.sign(dx) - np.sign(dy)) <= 1:
            (a1, b1) = ( mx + int(percent * abs(dy)) , my - int(percent * abs(dx)) )
            (a2, b2) = ( mx - int(percent * abs(dy)) , my + int(percent * abs(dx)) )

        else:
            (a1, b1) = ( mx + int(percent * abs(dy)) , my + int(percent * abs(dx)) )
            (a2, b2) = ( mx - int(percent * abs(dy)) , my - int(percent * abs(dx)) )

        # add points to variable
        edge_points += [ (a1, b1) , (a2, b2) ]

        # print this edge's shape
        pixels = [ tuple(img[a1,b1]) , tuple(img[a2,b2]) ]
        
        if (0,0,0) in pixels and (255,255,255) in pixels:
            string += "Bord - "
        elif (255,255,255) in pixels:
            string += "Bosse - "
        else:
            string += "Creux - "

    # print analysis string
    print("Edge analysis: " + string[:-3] + "\n----------\n")

    # return special edge points
    return edge_points


""" get indexes of N max values in image """
def get_n_max_value_indexes(img, N = 10, dist = 3):
    # create storage variable
    points = []
    
    # get N indexes of maximum
    for i in range(N):
        # get coordinates & update list
        (x,y) = np.unravel_index(img.argmax(), img.shape)
        points.append( (x,y) )

        # set point + neighborhood to 0
        neigh = [ (x+a,y+b) for a in range(-dist, dist) for b in range(-dist, dist) ]
        for (a,b) in neigh:
            if 0 <= a < img.shape[0] and 0 <= b < img.shape[1]:
                img[a][b] = 0

    # return value
    return points


""" this function uses Harris corner detector to analyze a jigsaw piece """
def get_harris_points(path):
    # read image from file
    img = cv.imread(path)
    gray = np.float32(cv.imread(path, 0))
    
    # return boolean np.ndarray with features as True
    dst = cv.cornerHarris(gray, 2, 3, 0.04)

    # get index corresponding to first maxima in image || default values { N = 10 and dist = 3 }
    points = get_n_max_value_indexes(dst)

    # get corners (4 Harris detection corner maxima)
    corners = select_corners(points)
    
    # print edge analysis result
    edge_points = guess_edge_shape(img, corners)

    # return variables
    return (img, points, corners, edge_points)


""" show image function """
def show_image(img, points, corners, edge_points):
    # add all points to basic image
    basic = img.copy()
    nb = [ (i,j) for i in range(-4,5) for j in range(-4,5) ]
    for (a,b) in points:
        # replace on image
        for (i,j) in nb:
            basic[a+i, b+j] = (255,0,0) # R

    # openCV line plotting variable
    thickness = 2

    # draw edge lines on detailled image
    detailled = img.copy()
    colors = [ (255,0,0) , (0,255,0) , (0,0,255) , (255,255,0) ] # R, G, B, Y
    for i in range(len(corners)):
        # get variables
        (y1, x1), (y2, x2) = corners[i], corners[(i+1) % len(corners)]
        color = colors[i]
        
        # plot line to image
        detailled = cv.line(detailled, (x1, y1), (x2, y2), color, thickness)

    # draw edge special lines on detailled image
    for i in range(len(corners)):
        # get start and end
        (y1, x1), (y2, x2) = edge_points[2*i], edge_points[2*i+1]
        
        # plot line to image
        detailled = cv.line(detailled, (x1, y1), (x2, y2), (255, 0, 255), thickness)

    # plot using subplots
    plt.subplot(1, 3, 1), plt.imshow(img)
    plt.title('Original')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 3, 2), plt.imshow(basic)
    plt.title('w/ all feature points')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 3, 3), plt.imshow(detailled)
    plt.title('w/ plotted lines')
    plt.xticks([]), plt.yticks([])
    
    plt.show()


""" main part of file """
if __name__ == '__main__':
    for fname in filenames:
        img, points, corners, edge_points = get_harris_points(fname)
        show_image(img, points, corners, edge_points)


