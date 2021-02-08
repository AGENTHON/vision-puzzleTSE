import math as math
import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt
import matplotlib.patches as patches


def distance(pt1, pt2):
    pt1 = np.array((pt1[0], pt1[1]))
    pt2 = np.array((pt2[0], pt2[1]))
    return np.linalg.norm(pt1 - pt2)


def closest_point(point, points):
    pt = []
    dist = 9999999
    for n in points:
        if distance(point, n) <= dist:
            dist = distance(point, n)
            pt = n
    return pt


def post_traitement(nbRows, nbCols, positions, barycentres, listeBoundingBox, image_initiale, img_scene):
    """ plot les carrés et positions sur les imagettes à partir des barycentres """
    height, width = img_scene.shape[:2]
    scene_positions=[]
    center_positions=[]
    for i in range(nbCols):
        for j in range(nbRows):
            scene_positions.append((int((width/nbCols)/2+i*(width/nbCols)),int( (height/nbRows)/2+j*(height/nbRows))))
    
    for k in positions:
        if(k!=None):
            (a,b,c,e)=k
            center_positions.append((int((b[0]-a[0]) / 2 ),int((e[1]-a[1]) / 2 )))
        else:
            center_positions.append(None)


    for i in range(len(center_positions)):
        if(center_positions[i] is not None):
            (x,y) = closest_point(center_positions[i],scene_positions)
            font = cv.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            fontColor = (255,0,0)
            lineType = 2
          
            cv.putText(img_scene,str((x,y)), 
                (x,y), 
                font, 
                fontScale,
                fontColor,
                lineType)
     
            cv.rectangle(image_initiale, (int(listeBoundingBox[i][0]), int(listeBoundingBox[i][1])), \
              (int(listeBoundingBox[i][0]+listeBoundingBox[i][3]), int(listeBoundingBox[i][1]+listeBoundingBox[i][2])), fontColor, 2)
    
    

    plt.imshow(image_initiale)
    plt.show()


### END OF FILE ###
    
