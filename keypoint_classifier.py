from cmath import cos
import logging
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter
import cv2
import mediapipe as mp
import torch
import cv2
import numpy as np
import argparse
from motrackers import CentroidTracker
from reid import ids_feature_,distance_,distance_list,ids_feature_list
import time
import cv2
import numpy as np
from openvino.inference_engine import IECore
import torch
from math import atan
import openvino_models as models
import monitors
from images_capture import open_images_capture
from pipelines import get_user_config, AsyncPipeline
from performance_metrics import PerformanceMetrics
from helpers import resolution
from urtils import load,id_assign
from datetime import datetime
import math
from shapely.geometry import Polygon
from shapely.geometry import LineString
from tensorflow.keras.models import load_model
default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

colors = (
        (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85),
        (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0),
        (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
        (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255),
        (0, 170, 255))

def findAngle(M1, M2):
    PI = 3.14159265
     
    # Store the tan value  of the angle
    angle = abs((M2 - M1) / (1 + M1 * M2))
 
    # Calculate tan inverse of the angle
    ret = atan(angle)
 
    # Convert the angle from
    # radian to degree
    val = (ret * 180) / PI
 
    # Print the result
    return (round(val, 4))
def slope(point1,point2):
    x = (point1[0],point2[0])
    y = (point1[1],point2[1])
    slope_, intercept = np.polyfit(x,y,1)
    return slope_

def getrect(actual_line):
    cd_length = 200
    ab = LineString([actual_line[1],actual_line[0]])
    left = ab.parallel_offset(cd_length / 2, 'left')
    right = ab.parallel_offset(cd_length / 2, 'right')
    c = left.boundary[1]
    d = right.boundary[0]  # note the different orientation for right offset
    cd = LineString([c, d])
    ef = LineString([(int(c.x),int(c.y)), (int(d.x),int(d.y))])
    left = ef.parallel_offset(200 , 'left')
    right = ef.parallel_offset(100 , 'right')
    e = left.boundary[1]
    f = right.boundary[0]
    pol = Polygon([(int(c.x),int(c.y)), (int(d.x),int(d.y)),(int(e.x),int(e.y))])
    xmin,ymin,xmax,ymax = pol.bounds
    return int(xmin),int(ymin),int(xmax),int(ymax)
def perpendicular(point1,point2):
    actual_line = (point1,point2)
    x_line = ((point1[0]+100,point1[1]),(point1[0]-100,point1[1]))
    y_line = ((point1[0],point1[1]+100),(point1[0],point1[1]-100))
    print("*"*50)
    print(actual_line)
    print(x_line)
    print(y_line)
    print("*"*50)
    try:
        ac_slope = slope(actual_line[0],actual_line[1])
        x_slope = slope(x_line[0],x_line[1])
        y_slope = slope(y_line[0],y_line[1])
        x_angle = findAngle(ac_slope,x_slope)
        y_angle = findAngle(ac_slope,y_slope)
    except Exception as e:
        print(e)
        x_angle=0
        y_angle=0
    
    print("*"*50)
    print("x:",x_angle)
    print("y:",y_angle)
    print("*"*50)
    ts = str(datetime.now())
    
    xmin,ymin,xmax,ymax = getrect(actual_line)
    crop = frame[ymin:ymax,xmin:xmax]
    print(crop.shape)
    try:
        img = crop/255
        img = cv2.resize(img, (150, 150))
        img = img.reshape((1,150,150,3))
        flag=float(format(classifier.predict(img)[0][0],".5f"))
    except Exception as e:
        print(e)
        flag=0
    #cv2.imwrite("/media/omkar/omkar3/media_pipe/mediapipe-tracking/crop_image/"+ts+".jpg", crop)
    cv2.rectangle(frame,(int(xmin),int(ymin)),(int(xmax),int(ymax)), (0, 255, 0), 4)
    return flag

def draw(img,poses, point_score_threshold,output_transform,skeleton=default_skeleton,draw_ellipses=False):
    img = output_transform.resize(img)
    if poses.size == 0:
        return img
    stick_width = 4

    img_limbs = np.copy(img)
    for pose in poses:
        points = pose[:,:2].astype(np.int32)
        points = output_transform.scale(points)
        points_scores = pose[:,2]

        for i,(p,v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
               
                flag_10=perpendicular(points[10],points[8])
                flag_9=perpendicular(points[9],points[7])
                if flag_10 < 0.5:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, "picked", points[10], font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                if flag_9 < 0.5:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, "picked", points[9], font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                if draw_ellipses:
                    middle = (points[i] + points[j]) // 2
                    vec = points[i] - points[j]
                    length = np.sqrt((vec * vec).sum())
                    angle = int(np.arctan2(vec[1], vec[0]) * 180 / np.pi)
                    polygon = cv2.ellipse2Poly(tuple(middle), (int(length / 2), min(int(length / 50), stick_width)),
                                               angle, 0, 360, 1)
                    cv2.fillConvexPoly(img_limbs, polygon, colors[j])
                else:
                    cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=stick_width)
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img

def preprocess(frame,size):
    n,c,h,w = size
    input_image = cv2.resize(frame, (w,h))
    input_image = input_image.transpose((2,0,1))
    input_image.reshape((n,c,h,w))
    return input_image
#------------------------------------tensorflow----------------
classifier = load_model("transfer_model.h5")
#--------------------------------------------------------------
plugin_config = {'CPU_BIND_THREAD': 'NO', 'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'}
ie= IECore()
model = models.OpenPose(ie, "human-pose-estimation-0001/FP32/human-pose-estimation-0001.xml", target_size=None, aspect_ratio=1,
                                prob_threshold=0.1)
input=model.image_blob_name
out_pool=model.pooled_heatmaps_blob_name
out_ht=model.heatmaps_blob_name
out_paf=model.pafs_blob_name
n,c,h,w = model.net.inputs[input].shape
exec_net = ie.load_network(network=model.net,config=plugin_config,device_name="CPU",num_requests = 1)
#--------------------------------------------yolov5--------------------------------------------
model_yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = model_yolov5.names
model_yolov5.to(device)    
#-----------------------------------------end--------------------------------------------------
cap = cv2.VideoCapture(2)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width,frame_height)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
output = cv2.VideoWriter('output_class.mp4', fourcc, fps, (1000,1000))
count=0
prev_frame_time = 0
new_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    _,frame = cap.read()
    frame = cv2.resize(frame, (1000,1000))
    #frame1 = frame.copy()
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    frame1 = [frame]
    det = model_yolov5(frame1)
    output_transform = models.OutputTransform(frame.shape[:2], None)
    output_resolution = (frame.shape[1], frame.shape[0])
    inputs, preprocessing_meta = model.preprocess(frame)
    infer_res = exec_net.start_async(request_id=0,inputs={input:inputs["data"]})
    status=infer_res.wait()
    results_pool = exec_net.requests[0].outputs[out_pool]
    results_ht = exec_net.requests[0].outputs[out_ht]
    results_paf = exec_net.requests[0].outputs[out_paf]
    results={"heatmaps":results_ht,"pafs":results_paf,"pooled_heatmaps":results_pool}
    poses,scores=model.postprocess(results,preprocessing_meta)
    #points = output_transform.scale(poses)
    #if count==10:
    frame = draw(frame,poses,0.1,output_transform)
    #    count=0
    #count+=1
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    cv2.putText(frame, "Fps:-"+str(fps), (20, 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    output.write(frame)
    cv2.imshow('smart store', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
output.release()
cv2.destroyAllWindows()