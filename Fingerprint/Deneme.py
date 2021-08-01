import cv2
import matplotlib.pyplot as plt
import numpy as np
import random as rng

img = cv2.imread("black-fingerprints-vector-260nw-662016160.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# plt.figure()
# plt.axis('off')
# plt.imshow(img,cmap = "gray") #color map ile griye çevirdik
# plt.show()

#Thresholding

_,thresh_img = cv2.threshold(img,thresh=75,maxval=255,type=cv2.THRESH_BINARY_INV)

plt.figure()
plt.axis('off')
plt.imshow(thresh_img,cmap = "gray") #color map ile griye çevirdik
plt.show()



#Adapting Threshold
adapting = cv2.adaptiveThreshold(thresh_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,3,2)

plt.figure()
plt.axis('off')
plt.imshow(adapting,cmap = "gray") #color map ile griye çevirdik
plt.show()

# contours

(contours,_) = cv2.findContours(adapting.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
cv2.imshow('Contours', img)
    