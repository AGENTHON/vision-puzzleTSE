import math as math
import numpy as np
import cv2 as cv

from copy import copy, deepcopy

""" path definition """
filename = './jigsaw_puzzle.png'


""" euclidean distance """
def euclidean_distance(x1, y1, x2, y2):
    # euclidean distance formula
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))


""" get Harris points function """
def get_harris_points(path):
    # read image from file
    img = cv.imread(path)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # return boolean np.ndarray with features as True
    dst = cv.cornerHarris(gray, 2, 1, 0.04)
    corn = dst > 0.01 * dst.max()

    # get all True indexes as zip
    corn_points = np.where(corn == True)

    # unzip them
    corn_points = zip(corn_points[0], corn_points[1])
    points = [x for x in corn_points]

    # remove too-close points (dist = 15)
    new_points = [ points[0] ]
    for (x,y) in points:
        # compare with already saved points
        comparison = [ euclidean_distance(x, y, a, b) < 15 for (a,b) in new_points ]

        # save point
        if True not in comparison:
            new_points.append( (x,y) )

    # replace points variable
    points = new_points
    
    # init storage variable
    distances = []

    # loop through points
    for (x,y) in points:
        # init variables
        mini = math.inf
        
        # compare and get minimum distance
        for (i,j) in points:
            if (x,y) != (i,j):
                mini = min(mini, euclidean_distance(x, y, i, j))

        # add to distances variable
        distances.append(mini)


    # DEBUG: create associative array
    print(sorted(distances))

    # get 4 most distant ones
    corners = []

    for i in range(4):
        # get max distance among variable
        max_dist = max(distances)
        print(max_dist)
        index = distances.index(max_dist)

        # append to corners
        corners.append( points[index] )

        # update lists of points / distances
        del points[index]
        del distances[index]

    # remove corner points from points
    points = list(set(points) - set(corners))

    # return variables
    return (img, points, corners)


""" show image function """
def show_image(img, points, corners):
    # add feature points in red to image
    nb = [ (i,j) for i in range(-2,3) for j in range(-2,3) ]
    for (a,b) in points:
        for (i,j) in nb:
            img[a+i, b+j] = (0, 0, 255)

    # add corner points in green to image
    nb = [ (i,j) for i in range(-2,3) for j in range(-2,3) ]
    for (a,b) in corners:
        for (i,j) in nb:
            img[a+i, b+j] = (0, 255, 0)
    
    # show
    cv.imshow('dst', img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


""" main part of file """
if __name__ == '__main__':
    img, points, corners = get_harris_points(filename)
    print(corners)
    
    show_image(img, points, corners)




