import torch
import cv2
from collections import deque
import numpy as np
from motrackers import CentroidTracker
from urtils import postprocess_yolov5,draw_tracks,tracking,check_id,pointInRect

tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
tracker_bottel = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')

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
bottel_id={}
tracks_id={}
tracklets_id={}
pickup_count={}
pic=0
while True:
    ids_bottel = {}
    
    ids = {}
    tracks_draw={}
    _,frame = cap.read()

    frame = cv2.resize(frame, (1000,1000))
    
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    frame1 = [frame]
    det = model_yolov5(frame1)
    bboxes, scores,labels,bboxes_bottel,scores_bottel,labels_bottel=postprocess_yolov5(x_shape, y_shape,det)
    tracks_bottel = tracker_bottel.update(bboxes_bottel, scores_bottel, labels_bottel)
    tracks = tracker.update(bboxes, scores, labels)
        
    frame,ids_bottel = draw_tracks(frame, tracks_bottel,ids_bottel)
    frame,ids = draw_tracks(frame, tracks,ids)
    tracks_draw,tracks_id,tracklets_id=tracking(ids,frame,tracks_id,tracklets_id,tracks_draw)

    for i,bbox in ids_bottel.items():
        x1_, y1_, x2_, y2_ = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        center_right = (int((x2_+x2_)/2),int((y2_+y1_)/2))
        bgr = (0, 255, 0)
        center=( int((x1_ + x2_) / 2), int((y1_ + y2_) / 2) ) 
        cv2.circle(frame,center_right,5,(0,0,0),-1)
        cv2.rectangle(frame, (x1_, y1_), (x2_, y2_), bgr, 2)
        if i not in bottel_id:
            bottel_id[i]=deque(maxlen = buffer)

        bottel_id[i].appendleft(center)
    #print(bottel_id)
    dirction_bottel={}
    filtered_dict = {k:v for (k,v) in bottel_id.items() if k in ids_bottel}
    for id,pts in filtered_dict.items():
        for i in np.arange(1, len(pts)):
            #print(i)
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
                if np.abs(dX) > 30:
                    dirX = 'West' if np.sign(dX) == 1 else 'East'

                if np.abs(dY) > 30:
                    dirY = 'North' if np.sign(dY) == 1 else 'South'

                #Set direction variable to the detected direction
                direction = dirX if dirX != '' else dirY

            #Draw a trailing red line to depict motion of the object.
            thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        dirction_bottel[id]=[direction,pts[0]]
       
        cv2.putText(frame, direction, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        counter += 1
    print(dirction_bottel)
    X = int(x_shape * (50/100))
    X_60 = int(x_shape * (70/100))
    start_point_50 = (X,0)
    end_point_50 = (X,y_shape)
    start_point_60 = (X_60,0)
    end_point_60 = (X_60,y_shape)

    for id,dic in dirction_bottel.items():
        if dic[-1][0] > X and dic[-1][0] < X_60:
            print(dic)
            if  dic[0] == "East":
                pickup_count[id] = [True,dic[-1]]
                pic+=1
            elif dic[0] == "West":
                pickup_count[id] = [False,dic[-1]]
                pic=-1
    print(pickup_count)
    


    for id,bbox in tracks_draw.items():
        cv2.putText(frame, str(id), bbox[:2], 1, cv2.FONT_HERSHEY_DUPLEX, (0, 0, 255), 3)
        cv2.rectangle(frame, bbox[:2], bbox[2:], (0, 255, 0), 1)

    output.write(frame)
    cv2.line(frame, start_point_50, end_point_50, (255,0,0), 2)
    cv2.line(frame, start_point_60, end_point_60, (255,0,0), 2)
    cv2.imshow('smart store', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
output.release()
cv2.destroyAllWindows()