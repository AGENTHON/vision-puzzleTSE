import numpy as np
import cv2 as cv

from copy import copy, deepcopy

""" path definition """
filename = './jigsaw_puzzle.png'


""" morphological gradient function """
def get_morpho_gradient(path):
    # read image from file
    img = cv.imread(path)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # convert to binary
    _, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
    
    # structural element
    element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    
    # dilate and erode of image
    erosion = cv.erode(binary, element)
    dilatation = cv.dilate(binary, element)

    # morphological gradient
    gradient = dilatation - erosion

    # return result
    return gradient


""" show image function """
def show_image(img):
    # show
    cv.imshow('dst', img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


""" main part of file """
if __name__ == '__main__':
    img = get_morpho_gradient(filename)
    show_image(img)



