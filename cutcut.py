import cv2 as cv

""" image a decouper """
IMG_PATH = "TF2/meurtre_1.png"

""" le répertoire doit exister """
IMG_SAVE_PATH = "TF2/tf2_charcute/"

""" longueur de l'image / 5 , arrondi inférieur """
WIDTH = 768

""" hauteur de l'image / 4 , arrondi inférieur """
HEIGHT = 540


# read img
img = cv.imread(IMG_PATH)

# loop and cut image
for i in range(4):
    for j in range(5):
        # crop image in subpieces
        crop_img = img[540*i : 540*(i+1), 768*j : 768*(j+1)]

        # save cropped
        name = IMG_SAVE_PATH + str(i) + "_" + str(j) + ".png"
        cv.imwrite(name, crop_img)
