import math as math
import numpy as np
from matplotlib import pyplot as plt

from copy import copy

import cv2 as cv


""" paths definition """
filenames = [ './Banque/piece_' + str(i) + '.png' for i in range(1, 7) ]


""" Enum Types Definition """
FLAT, HOLLOW, DENT = 'bord', 'creux', 'bosse'


""" euclidean distance """
def euclidean_distance(x1, y1, x2, y2):
    # euclidean distance formula
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))


""" distance from point to line """
def distance_point_line(p1, p2, p):
    # this function gets the distance from p to the line going through p1 and p2
    (x1, y1) = p1
    (x2, y2) = p2
    (x0, y0) = p

    # Formula here: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
    return abs( (x2-x1) * (y1-y0) - (x1-x0) * (y2-y1) ) / euclidean_distance(x1, y1, x2, y2)


""" this function uses Harris corner detector to analyze a jigsaw piece """
def get_harris_points(path):
    # read image from file
    img = cv.imread(path)
    gray = np.float32(cv.imread(path, 0))
    
    # apply Harris corner detection algorithm
    harr = cv.cornerHarris(gray, 2, 3, 0.04)
    
    # get index corresponding to first maxima in image || default values { N = 10 and dist = 3 }
    points = get_n_max_value_indexes(harr)
    
    # get corners (4 Harris detection corner maxima)
    corners = select_corners(points)
    
    # print edge analysis result
    shapes, edge_points = guess_edge_shape(img, corners)
    
    # classify edge points
    points_per_edge = classify_points(path, corners)
    
    # return variables
    return (points, corners, edge_points, shapes, points_per_edge)


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
    edge_points, shapes = [] , []
    
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
            shapes.append(FLAT)
        elif (255,255,255) in pixels:
            shapes.append(DENT)
        else:
            shapes.append(HOLLOW)
    
    # DEBUG: print here for now
    print(shapes)
    
    # return shapes and special edge points
    return (shapes, edge_points)


""" classify edge points from laplacian using minimal distance to a line """
def classify_points(path, corners):
    # start by initializing variables
    laplacian = generate_laplacian_image(path)
    points_per_edge = [ [] for i in range(len(corners)) ]
    
    # get index of not null coordinates
    indices = np.where(laplacian == [255])
    coordinates = [ (indices[0][i], indices[1][i]) for i in range(len(indices[0])) ]

    # get nearest line
    for (a,b) in coordinates:
        dist = distance_point_line(corners[0], corners[1], (a,b))
        point_class = 0

        for i in range(1, len(corners)):
            new_dist = distance_point_line(corners[i], corners[(i+1) % len(corners)], (a,b))
            if new_dist < dist:
                dist, point_class = new_dist, i

        points_per_edge[point_class].append( (a,b) )
    
    # return computed values
    return points_per_edge


""" plot images """
def plot_all_images(path, feature_points, corners, edge_points, points_per_edge):
    # generate images
    image = generate_image(path)
    basic = generate_basic_image(path, feature_points)
    detailled = generate_detailled_image(path, corners, edge_points)
    laplacian = generate_laplacian_image(path)
    separated = generate_separated_image(path, points_per_edge)

    # plot using subplots
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title('Original')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(2, 3, 2)
    plt.imshow(basic)
    plt.title('w/ all feature points')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(2, 3, 3)
    plt.imshow(detailled)
    plt.title('w/ plotted lines')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(2, 3, 5)
    plt.imshow(laplacian, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title('Binary of Laplacian')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(2, 3, 6)
    plt.imshow(separated)
    plt.title('w/ edge separated')
    plt.xticks([]), plt.yticks([])
    
    plt.show()


""" generate image method """
def generate_image(path):
    # return image
    return cv.imread(path)


""" generate basic image method """
def generate_basic_image(path, feature_points):
    # add all points to basic image
    basic = cv.imread(path)
    nb = [ (i,j) for i in range(-4,5) for j in range(-4,5) ]
    for (a,b) in feature_points:
        # replace on image
        for (i,j) in nb:
            basic[a+i, b+j] = (255,0,0) # R
    
    # return image
    return basic


""" generate detailled image method """
def generate_detailled_image(path, corners, edge_points):
    # openCV line plotting variable
    thickness = 2
    
    # draw edge lines on detailled image
    detailled = cv.imread(path)
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
    
    # return image
    return detailled


""" generate laplacian image method """
def generate_laplacian_image(path):
    # read image from file
    gray = cv.imread(path, 0)
    
    # apply laplacian, separate 0 from others
    laplacian = cv.Laplacian(gray, cv.CV_64F)
    laplacian[laplacian != 0] = 255
    
    # return image
    return laplacian


""" separate points in each edge """
def generate_separated_image(path, points_per_edge):
    # get size of image
    img = cv.imread(path)
    
    # create separated image
    separated = np.zeros(img.shape, dtype = "uint8")
    colors = [ (255,0,0) , (0,255,0) , (0,0,255) , (255,255,0) ] # R, G, B, Y
    for i in range(len(points_per_edge)):
        color = colors[i]
        for (x,y) in points_per_edge[i]:
            separated[x][y] = color
    
    # return separated image
    return separated


""" main part of file """
if __name__ == '__main__':
    for fname in filenames:
        feature_points, corners, edge_points, shapes, points_per_edge = get_harris_points(fname)
        plot_all_images(fname, feature_points, corners, edge_points, points_per_edge)


