import math as math
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv


""" calibrate ORB Detector / compute the descriptors """
minHessian = 400
orb = cv.ORB_create(nfeatures = 1500)


""" paths definition """
filename_scene = './Test_Orb/dedede_1.png'
# filename_object = [ './dedede_charcute/' + str(i) + '_' + str(j) + '.png' for i in range(4) for j in range(4) ]
filename_object = [ './Test_Orb/dedede_head_1.png' ]


""" read scene """
img_scene = cv.imread(filename_scene)
img_scene = cv.cvtColor(img_scene, cv.COLOR_BGR2RGB)
keypoints_scene, descriptors_scene = orb.detectAndCompute(img_scene, None)


""" for all objects in scene """
for fname in filename_object:
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

    H, _ =  cv.findHomography(obj, scene, cv.RANSAC)

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


### END OF FILE ###
