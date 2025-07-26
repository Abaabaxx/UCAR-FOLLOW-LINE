# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 20:34:14 2025

@author: 寻木
"""

import cv2
import numpy as np
x=0
y=0

img=cv2.imread("1.png")
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_new=np.zeros((img.shape[0],img.shape[1]))

H=[
   
[1.9194,
    6.0129,
 -901.9355],
    [0.0000,
    8.0484,
 -919.3548],
    [0.0000,
    0.0061,
    1]

   ]
for i in range (0,img.shape[1]):
    for j in range (0,img.shape[0]):
        x=(H[0][0]*i+H[0][1]*j+H[0][2])/(H[2][0]*i+H[2][1]*j+H[2][2])
        y=(H[1][0]*i+H[1][1]*j+H[1][2])/(H[2][0]*i+H[2][1]*j+H[2][2])
        print(x,y)
        x=int(x)
        y=int(y)
        print(x,y,i,j)
        if x>=img.shape[1] or y >=img.shape[0] or x <= 0 or y <= 0 or i<=0 or j<=0:
            continue
        else:
            img_new[y][x]=img[j][i]
        
       
cv2.imshow("1",img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()