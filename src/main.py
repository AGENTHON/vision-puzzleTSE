import cv2 as cv

from pre_processing import pre_traitement
from processing import traitement
from post_processing import post_traitement


# Resources
IMG_SCENE = cv.cvtColor( cv.imread("./resources/dedede_scene.png") , cv.COLOR_BGR2RGB )
IMG_PIECES = cv.cvtColor( cv.imread("./resources/dedede_pieces.png") , cv.COLOR_BGR2RGB )
NB_ROWS, NB_COLS = 4, 5


# Main
if __name__ == "__main__":
    # pre-processing
    imagettes, barycentres, bounding_boxes = pre_traitement(IMG_PIECES)

    # processing
    positions = traitement(IMG_SCENE, imagettes, barycentres)

    # post-processing
    post_traitement(NB_ROWS, NB_COLS, positions, barycentres, bounding_boxes, IMG_PIECES, IMG_SCENE)
