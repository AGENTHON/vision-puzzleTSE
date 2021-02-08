import cv2 
import numpy as np
import matplotlib.pyplot as plt


path = 'puzzleDedede/full_pieces.png'
img = cv2.imread(path)

plt.imshow(img, cmap='viridis')
plt.show()


gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
height, width = gray.shape

plt.imshow(gray, cmap='gray')
plt.show()

font = gray

font1 = np.where(gray[:,:]==gray[1,1]-1)
font2 = np.where(gray[:,:]==gray[1,1])                 
font3 = np.where(gray[:,:]==gray[1,1]+1)                 
#objet = np.where(gray[:,:]!=240)


font[font1] = 0
font[font2] = 0
font[font3] = 0
 #gray[objet] = 255


ret,bw = cv2.threshold(font,0,255,cv2.THRESH_BINARY)

plt.imshow(bw, cmap='binary')
plt.title('bw')
plt.show()

# Equivalent d'un bwfill

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
plt.imshow(im_out, cmap='binary')
plt.title('im_out')
plt.show()


label = cv2.connectedComponents(im_out)

nblabel = label[0]
imglabelOrigine = label[1]

print(label[0])
plt.imshow(imglabelOrigine, cmap='gray')
plt.title('label')
plt.show()

centresImg = []
imgPieces = []
BB = []
err = True

for i in range(1,nblabel):
    
    idxLabel = np.where(imglabelOrigine[:,:]!=i)
    imgLabel = img.copy()
    
    imgLabel[:,:,0][idxLabel]  = 0
    imgLabel[:,:,1][idxLabel]  = 0
    imgLabel[:,:,2][idxLabel]  = 0
    
    imgCentroid = im_out.copy()
    imgCentroid[idxLabel] = 0
    
    # plt.imshow(imgLabel, cmap='gray')
    # plt.show()    
    
    
    contours, hierarchy = cv2.findContours(imgCentroid,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        #calculate moments for each contour
        M = cv2.moments(c)
        if (M["m00"]==0) :
            err=False
            continue
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    if err:
        imgLabel = imgLabel[y:y+h,x:x+w,:]
        # plt.imshow(imgLabel, cmap='viridis')
        # plt.show()    
        
        imgPieces.append(imgLabel)      
        centresImg.append((cX,cY))
        BB.append((x,y,h,w))
    err = True
    
