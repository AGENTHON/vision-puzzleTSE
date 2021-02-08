import math as math
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv


""" paths definition """
img_scene = cv.cvtColor( cv.imread('./Dedede/dedede_1.png') , cv.COLOR_BGR2RGB )

imagettes = [ cv.cvtColor( cv.imread('./Dedede/dedede_charcute/' + str(i) + '_' + str(j) + '.png'), cv.COLOR_BGR2RGB ) for i in range(4) for j in range(5) ]
barycentres = [ (0,0) for i in range(20) ]


def euclidian_distance(x1, y1, x2, y2):
    """ distance par norme euclidienne """
    return math.sqrt( math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2) )


def detect_with_orb(img_scene, imagettes, barycentres, showMatches = False):
    """ Use Orb to detect subimages in the scene """

    # declare Orb parameters
    orb = cv.ORB_create(nfeatures = 100000)

    # read scene
    keypoints_scene, descriptors_scene = orb.detectAndCompute(img_scene, None)
    print("Nombre de keypoints : " + str(len(keypoints_scene)))

    # show keypoints of scene
    showKeyPts = cv.drawKeypoints(img_scene, keypoints_scene, cv.DRAW_MATCHES_FLAGS_DEFAULT)
    plt.imshow(showKeyPts)
    plt.show()

    # coordinates object for return
    coordinates = []
    positions = []

    # for all objects in scene
    for img_object in imagettes:
        # read object
        keypoints_obj, descriptors_obj = orb.detectAndCompute(img_object, None)

        # match descriptors using a FLANN based matcher { norm L2 }
        matcher = cv.BFMatcher()
        knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)

        # filter matches using the Lowe's ratio test
        ratio_thresh = 0.75
        good_matches = []

        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        # None if not matched
        if len(good_matches) < 4:
            print("Pas assez de points pour l'imagette.")
            positions.append(None)
            continue

        # localize the object
        obj = np.empty((len(good_matches), 2) , dtype = np.float32)
        scene = np.empty((len(good_matches), 2) , dtype = np.float32)
        
        # keypoints from the good matches
        for i in range(len(good_matches)):
            obj[i, 0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
            obj[i, 1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
            scene[i, 0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
            scene[i, 1] = keypoints_scene[good_matches[i].trainIdx].pt[1]

        H, _ =  cv.findHomography(obj, scene, cv.LMEDS)

        # get corners from the scene ( the object to be "detected" )
        obj_corners = np.empty( (4,1,2) , dtype = np.float32)
        obj_corners[0,0,0] = 0
        obj_corners[0,0,1] = 0
        obj_corners[1,0,0] = img_object.shape[1]
        obj_corners[1,0,1] = 0
        obj_corners[2,0,0] = img_object.shape[1]
        obj_corners[2,0,1] = img_object.shape[0]
        obj_corners[3,0,0] = 0
        obj_corners[3,0,1] = img_object.shape[0]

        # scene corners
        scene_corners = cv.perspectiveTransform(obj_corners, H)

        # corner coordinates of piece
        a = ( int(scene_corners[0,0,0]) , int(scene_corners[0,0,1]) )
        b = ( int(scene_corners[1,0,0]) , int(scene_corners[1,0,1]) )
        c = ( int(scene_corners[2,0,0]) , int(scene_corners[2,0,1]) )
        e = ( int(scene_corners[3,0,0]) , int(scene_corners[3,0,1]) )

        positions.append( (a,b,c,e) )

    # return coordinates of sub-images
    return positions



positions = detect_with_orb(nbRows, nbCols, img_scene, imagettes, barycentres)
print(positions)


### END OF FILE ###
