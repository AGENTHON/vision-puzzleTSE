import math as math
import numpy as np
from matplotlib import pyplot as plt

import cv2 as cv


""" paths definition """
filenames = [ './Banque/piece_' + str(i) + '.png' for i in range(1, 7) ]


""" euclidean distance """
def euclidean_distance(x1, y1, x2, y2):
    # euclidean distance formula
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))


""" get radial angle """
def get_radial_angle(px, py, cx, cy):
    # radial angle formula
    return - math.degrees(math.atan2(py - cy, px - cx)) % 360


""" sort the corners by radial angle """
def order_corners(corners):
    # compute centroid of corners
    x = [c[0] for c in corners]
    y = [c[1] for c in corners]
    centroid = ( sum(x) / len(corners), sum(y) / len(corners) )

    # get radial angle of each corner
    radial = [ None for i in range(4) ]
    for i in range(len(corners)):
        # get point coordinates
        (x,y) = corners[i]
        (a,b) = centroid

        # compute radial angle
        radial[i] = get_radial_angle(x, y, a, b)
    
    # order corners by angle
    ordered = []
    while len(corners):
        # init mini & angle
        mini, angle = corners[0], radial[0]

        for i in range(len(corners)):
            if radial[i] < angle:
                # update mini & angle
                mini = corners[i]
                angle = radial[i]

        # update storage variables
        index = corners.index(mini)
        ordered.append(mini)
        
        del corners[index]
        del radial[index]

    # return clockwise corners
    return ordered


""" get specific points around the center of each edge to get the shape -- corners MUST be ordered """
def guess_shape_per_edge(img, corners):
    # init storage variable
    specific = [ [ None for i in range(2) ] for j in range(len(corners)) ]
    for i in range(len(corners)):
        (x1, y1) , (x2, y2) = corners[i], corners[(i+1) % len(corners)]
        size = euclidean_distance(x1, y1, x2, y2)
        dx, dy = abs(x2 - x1) / size , abs(y2 - y1) / size
        mx, my = (x1 + x2) // 2 , (y1 + y2) // 2

        specific[i][0] = ( mx + int(dy * 20) , my + int(dx * 20) )
        specific[i][1] = ( mx - int(dy * 20) , my - int(dx * 20) )

    return specific


""" this function uses Harris corner detector to return 4 corners of the puzzle piece """
def get_harris_points(path):
    # read image from file as binary
    img = cv.imread(path, 0)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # return boolean np.ndarray with features as True
    gray = np.float32(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    dst = cv.cornerHarris(gray, 2, 1, 0.04)
    
    # get feature points
    corn = dst > 0.01 * dst.max()
    corn_points = np.where(corn == True)
    corn_points = zip(corn_points[0], corn_points[1])
    points = [x for x in corn_points]

    # remove too-close points (dist = 15)
    new_points = [ points[0] ]
    for (x,y) in points:
        # compare with already saved points
        comparison = [ euclidean_distance(x, y, a, b) < 15 for (a,b) in new_points ]
        if True not in comparison:
            new_points.append( (x,y) )
    points = new_points
    
    # get distances to nearest neighbour
    distances = []
    for (x,y) in points:
        # init variables
        mini = math.inf
        for (i,j) in points:
            if (x,y) != (i,j):
                mini = min(mini, euclidean_distance(x, y, i, j))
        distances.append(mini)

    # get 4 most distant ones
    corners = []
    for i in range(4):
        # update variables
        max_dist = max(distances)
        index = distances.index(max_dist)
        corners.append( points[index] )

        del points[index]
        del distances[index]
    
    # call generic functions
    corners = order_corners(corners)
    specific = guess_shape_per_edge(img, corners)

    # return variables
    return (img , corners, specific)


""" show image function """
def show_image(img, corners, specific):
    # add corner points to image
    withPoints = img.copy()
    nb = [ (i,j) for i in range(-4,5) for j in range(-4,5) ]
    colors = [ (255,0,0) , (0,255,0) , (0,0,255) , (255,255,0) ] # R, G, B, Y
    for i in range(len(corners)):
        # get variables
        (a,b) = corners[i]
        color = colors[i]

        # replace on image
        for (i,j) in nb:
            withPoints[a+i, b+j] = color

    # add specific points to image
    nb = [ (i,j) for i in range(-4,5) for j in range(-4,5) ]
    color = (255,0,255) # M
    for i in range(len(corners)):
        for j in range(2):
            # get variables
            (a,b) = specific[i][j]

            # replace on image
            for (x,y) in nb:
                withPoints[a+x, b+y] = color

        # analyze shape of this edge
        pixels = [ tuple(img[a,b]) for (a,b) in specific[i] ]

        if (0,0,0) in pixels and (255,255,255) in pixels:
            print("Bord")

        elif (255,255,255) in pixels:
            print("Bosse")

        else:
            print("Creux")

    print("----------")

    # show
    plt.subplot(1, 2, 1), plt.imshow(img)
    plt.title('Original')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 2, 2), plt.imshow(withPoints)
    plt.title('With feature points')
    plt.xticks([]), plt.yticks([])
    
    plt.show()


""" main part of file """
if __name__ == '__main__':
    for fname in filenames:
        img, corners, specific = get_harris_points(fname)
        show_image(img, corners, specific)




