import math as math
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv


""" paths definition """
filename_scene = './Dedede/dedede_1.png'
filename_objects = [ './Dedede/dedede_charcute/' + str(i) + '_' + str(j) + '.png' for i in range(4) for j in range(5) ]


def detect_with_orb(filename_scene, filename_objects, showMatches = False):
    """ Use Orb to detect subimages in the scene """

    # declare Orb parameters
    orb = cv.ORB_create(nfeatures = 5000)

    # read scene
    img_scene = cv.imread(filename_scene)
    img_scene = cv.cvtColor(img_scene, cv.COLOR_BGR2RGB)
    keypoints_scene, descriptors_scene = orb.detectAndCompute(img_scene, None)

    # show keypoints of scene
    showKeyPts = cv.drawKeypoints(img_scene, keypoints_scene, cv.DRAW_MATCHES_FLAGS_DEFAULT)
    plt.imshow(showKeyPts)
    plt.show()

    # coordinates object for return
    coordinates = []

    # for all objects in scene
    for fname in filename_objects:
        # read object
        img_object = cv.imread(fname)
        img_object = cv.cvtColor(img_object, cv.COLOR_BGR2RGB)
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

        # draw matches
        img_matches = np.empty( (max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1] + img_scene.shape[1], 3), dtype = np.uint8)
        cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, img_matches, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # localize the object
        obj = np.empty((len(good_matches), 2) , dtype = np.float32)
        scene = np.empty((len(good_matches), 2) , dtype = np.float32)

        # keypoints from the good matches
        for i in range(len(good_matches)):
            obj[i, 0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
            obj[i, 1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
            scene[i, 0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
            scene[i, 1] = keypoints_scene[good_matches[i].trainIdx].pt[1]

        # don't break the loop
        if len(good_matches) < 4:
            print("Pas assez de points pour l'image : '" + fname + "'")
            continue

        H, _ =  cv.findHomography(obj, scene, cv.LMEDS)

        # get corners from the scene ( the object to be "detected" )
        obj_corners = np.empty((4,1,2), dtype=np.float32)
        obj_corners[0,0,0] = 0
        obj_corners[0,0,1] = 0
        obj_corners[1,0,0] = img_object.shape[1]
        obj_corners[1,0,1] = 0
        obj_corners[2,0,0] = img_object.shape[1]
        obj_corners[2,0,1] = img_object.shape[0]
        obj_corners[3,0,0] = 0
        obj_corners[3,0,1] = img_object.shape[0]

        scene_corners = cv.perspectiveTransform(obj_corners, H)
        coordinates.append(scene_corners)

        # show detected matches if boolean is True
        if showMatches:
            # draw lines between the corners (object detected in the scene)
            start_point = ( int(scene_corners[0,0,0] + img_object.shape[1]) , int(scene_corners[0,0,1]) )
            end_point = ( int(scene_corners[1,0,0] + img_object.shape[1]) , int(scene_corners[1,0,1]) )
            cv.line(img_matches, start_point, end_point, (0,255,0), 4)

            start_point = ( int(scene_corners[1,0,0] + img_object.shape[1]) , int(scene_corners[1,0,1]) )
            end_point = ( int(scene_corners[2,0,0] + img_object.shape[1]) , int(scene_corners[2,0,1]) )
            cv.line(img_matches, start_point, end_point, (0,255,0), 4)

            start_point = ( int(scene_corners[2,0,0] + img_object.shape[1]) , int(scene_corners[2,0,1]) )
            end_point = ( int(scene_corners[3,0,0] + img_object.shape[1]) , int(scene_corners[3,0,1]) )
            cv.line(img_matches, start_point, end_point, (0,255,0), 4)

            start_point = ( int(scene_corners[3,0,0] + img_object.shape[1]) , int(scene_corners[3,0,1]) )
            end_point = ( int(scene_corners[0,0,0] + img_object.shape[1]) , int(scene_corners[0,0,1]) )
            cv.line(img_matches, start_point, end_point, (0,255,0), 4)

            # show detected matches
            plt.imshow(img_matches)
            plt.title('Good Matches & Object detection')
            plt.show()

    # return coordinates of sub-images
    return coordinates


def reconstruct(filename_scene, coordinates):
    """ reconstruct the image from the detected subimages using Orb """

    # read scene
    img_scene = cv.imread(filename_scene)
    img_scene = cv.cvtColor(img_scene, cv.COLOR_BGR2RGB)
    shape = img_scene.shape

    # create black image
    black_scene = np.zeros(shape, dtype = np.uint8)

    for coord in coordinates:
        # get sub image from scene
        X1, X2 = int(coord[0,0,0]), int(coord[1,0,0])
        Y1, Y2 = int(coord[0,0,1]), int(coord[3,0,1])

        # crop img
        sub_img = img_scene[Y1 : Y2, X1 : X2]

        # replace space in black scene
        black_scene[Y1 : Y2, X1 : X2] = sub_img

    # plot black scene
    plt.imshow(black_scene)
    plt.title("Black scene with sub images")
    plt.show()



coordinates = detect_with_orb(filename_scene, filename_objects)
reconstruct(filename_scene, coordinates)


### END OF FILE ###
