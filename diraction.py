import torch
import cv2
from collections import deque
import numpy as np
model_yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = model_yolov5.names
model_yolov5.to(device) 

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width,frame_height)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
output = cv2.VideoWriter('output_direction.mp4', fourcc, fps, (1000,1000))
#Used in deque structure to store no. of given buffer points
buffer = 20

#Points deque structure storing 'buffer' no. of object coordinates
pts = deque(maxlen = buffer)
#Counts the minimum no. of frames to be detected where direction change occurs
counter = 0
#Change in direction is stored in dX, dY
(dX, dY) = (0, 0)
#Variable to store direction string
direction = ''
while True:
    _,frame = cap.read()
    frame = cv2.resize(frame, (1000,1000))
    
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    frame1 = [frame]
    det = model_yolov5(frame1)
    labels, cord = det.xyxyn[0][:, -1], det.xyxyn[0][:, :-1]
    
    X = int(x_shape * (20/100))
    start_point = (X,0)
    end_point = (X,y_shape)
    for i in range(len(cord)):
        bbox=cord[i]
        c=labels[i]
        
        if int(c)==39:
            x1_, y1_, x2_, y2_ = int(bbox[0] * x_shape), int(bbox[1] * y_shape), int(bbox[2] * x_shape), int(bbox[3] * y_shape)
            center_right = (int((x2_+x2_)/2),int((y2_+y1_)/2))
            bgr = (0, 255, 0)
            center=( int((x1_ + x2_) / 2), int((y1_ + y2_) / 2) ) 
            cv2.circle(frame,center_right,5,(0,0,0),-1)
            cv2.rectangle(frame, (x1_, y1_), (x2_, y2_), bgr, 2)
            pts.appendleft(center)
    for i in np.arange(1, len(pts)):
        print(i)
        #If no points are detected, move on.
        if(pts[i-1] == None or pts[i] == None):
            continue

        #If atleast 10 frames have direction change, proceed
        if counter >= 10 and i == 1 and pts[-2] is not None:
            #Calculate the distance between the current frame and 10th frame before
            dX = pts[-2][0] - pts[i][0]
            dY = pts[-2][1] - pts[i][1]
            (dirX, dirY) = ('', '')

            #If distance is greater than 100 pixels, considerable direction change has occured.
            if np.abs(dX) > 100:
                dirX = 'West' if np.sign(dX) == 1 else 'East'

            if np.abs(dY) > 100:
                dirY = 'North' if np.sign(dY) == 1 else 'South'

            #Set direction variable to the detected direction
            direction = dirX if dirX != '' else dirY

        #Draw a trailing red line to depict motion of the object.
        thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
    
    cv2.putText(frame, direction, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    counter += 1
    output.write(frame)
    cv2.imshow('smart store', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
output.release()
cv2.destroyAllWindows()