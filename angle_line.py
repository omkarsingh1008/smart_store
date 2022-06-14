import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

frame = cv2.imread('1648123042.jpg')
frame = cv2.resize(frame,(1000,1000))

(startx,starty),angle,length=(429,789),5,100
# unpack the first point
angle1 = 37
angle = 180-angle
angle1 = 180-angle1
x1 = startx
y1 = starty
x2 = x1 + math.cos(angle) * length
y2 = y1 + math.sin(angle) * length
x2_ = x1 + math.cos(angle1) * length
y2_ = y1 + math.sin(angle1) * length

print(x1, y1, x2, y2)
cv2.line(frame, (x1,y1), (int(x2),int(y2)), (255,0,0), 2)
cv2.line(frame, (x1,y1), (int(x2_),int(y2_)), (255,0,0), 2)
cv2.circle(frame,(x1,y1),10,(0,0,0),5)
cv2.imshow('Drawing_Line', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()