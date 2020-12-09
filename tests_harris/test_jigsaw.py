import math as math
import numpy as np
import cv2 as cv

""" image definition """
filename = 'jigsaw_puzzle.png'


""" get Harris corners function """
def get_harris_corners(path):
    # read image from file
    img = cv.imread(path)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # return boolean np.ndarray with features as True
    dst = cv.cornerHarris(gray, 2, 1, 0.04)
    corn = dst > 0.01 * dst.max()
    return [ img , corn ]


""" get coordinates of feature points """
def get_coordinates(booleanArray):
    # get all True indexes as zip
    corn_points = np.where(corn == True)

    # unzip them and return
    corn_points = zip(corn_points[0], corn_points[1])
    return [x for x in corn_points]


""" get only important coordinates (without really close neighbors) """
def clean_coordinates(points, distMin):
    # new array
    cleanedPt = []

    # loop and add (if not close enough)
    for (x,y) in points:
        addIt = True
        for (i,j) in cleanedPt:
            if math.sqrt(math.pow(x - i, 2) + math.pow(y - j, 2)) <= distMin:
                addIt = False

        if addIt:
            cleanedPt.append( (x,y) )

    # return cleaned points
    return cleanedPt


""" add points to image and show it """
def show_image_features(img, points):
    # add feature points in green to image (artificially bigger)
    nb = [ (i,j) for i in range(-2,3) for j in range(-2,3) ]
    for (a,b) in points:
        for (i,j) in nb:
            img[a+i, b+j] = (0, 0, 255)

    # show
    cv.imshow('dst', img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


""" main part of file """
img, corn = get_harris_corners(filename)

points = get_coordinates(corn)
cleaned = clean_coordinates(points, 20)
print(cleaned)

show_image_features(img, cleaned)





