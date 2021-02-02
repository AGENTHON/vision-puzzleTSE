import cv2 
import numpy as np

path = 'puzzle Dedede\pieces_1.png'
img = cv2.imread(path)

cv2.imshow('oui',img)
cv2.waitKey(2)
cv2.destroyAllWindows()



gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = gray.shape

cv2.imshow('oui',gray)
cv2.waitKey(2)
cv2.destroyAllWindows()

font = gray[1,1]

font = np.where(gray[:,:]==gray[1,1])
#objet = np.where(gray[:,:]!=240)


gray[font] = 0
 #gray[objet] = 255


ret,bw = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)

cv2.imshow('imgbw',bw)
cv2.waitKey(2)
cv2.destroyAllWindows()


im_floodfill = bw.copy()

# Mask used to flood filling.

# Notice the size needs to be 2 pixels than the image.

h, w = bw.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

# Floodfill from point (0, 0)

cv2.floodFill(im_floodfill, mask, (0,0), 255);

# Invert floodfilled image

im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.

im_out = bw | im_floodfill_inv

# Display images.
# cv2.imshow("Foreground", im_out)

# cv2.waitKey(2)


label = cv2.connectedComponents(im_out)