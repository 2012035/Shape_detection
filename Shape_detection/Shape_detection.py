import cv2 as c
import numpy as np
# https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18(Defination of contour)

image=c.imread('image/shape.png')
fin=c.resize(image,(300,400))
image1=c.cvtColor(fin,c.COLOR_BGR2GRAY)
canny=c.Canny(image1,300,350)

contour,h=c.findContours(canny,c.RETR_EXTERNAL,c.CHAIN_APPROX_NONE)

for con in contour:
    area=c.contourArea(con)

    # print(area)
    if area>1:

        c.drawContours(fin, contour, 1, (0, 0, 0))
        len=c.arcLength(con,True)
        approx=c.approxPolyDP(con,0.05*len,True)
        x,y,w,h=c.boundingRect(approx)
        c.rectangle(fin,(x,y),(x+w,y+h),(0,0,0),4)
        # print(x,y,w,h)
# stack=np.hstack((image,fin))

c.imshow('Original',image)
c.imshow('Shape_detection',fin)
while True:
    if(c.waitKey(1) & 0xFF== ord('z')):
        break

