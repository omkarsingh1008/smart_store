from cmath import cos
import cv2
import numpy as np
import time
from openvino.inference_engine import IECore
from math import atan
import openvino_models as models
from datetime import datetime
from shapely.geometry import Polygon
from shapely.geometry import LineString
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
from classifier_urtils import draw,preprocess_tensorflow,classifier_tensorflow,preprocess,classifier
import torch
from urtils import postprocess_yolov5,draw_tracks,tracking,check_id
from motrackers import CentroidTracker

tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
#------------------------------------tensorflow----------------
model_c = load_model("mymodel_vgg1")

#ie= IECore()

#net_c = ie.read_network(model="my_16/saved_model.xml",weights="my_16/saved_model.bin")
#input_layer_c = next(iter(net_c.inputs))
#n_c,c_c,h_c,w_c = net_c.inputs[input_layer_c].shape
#size_c = [n_c,c_c,h_c,w_c]
#output_layer_c = next(iter(net_c.outputs))
#exec_net_c = ie.load_network(network=net_c,device_name="CPU",num_requests = 1)
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
model_yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
device = 'cpu'
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
count_=0
prev_frame_time = 0
new_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX
tracks_id={}
tracklets_id={}
while True:
    _,frame = cap.read()
    ids = {}
    tracks_draw={}
    id_={"":""}
    frame = cv2.resize(frame, (1000,1000))
    frame_dummy = frame
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    frame1 = [frame]
    det = model_yolov5(frame1)
    bboxes, scores,labels,bboxes_bottel,scores_bottel,labels_bottel=postprocess_yolov5(x_shape, y_shape,det)
    tracks = tracker.update(bboxes, scores,labels)
        
    frame,ids = draw_tracks(frame, tracks,ids)
       
    tracks_draw,tracks_id,tracklets_id=tracking(ids,frame,tracks_id,tracklets_id,tracks_draw)
    
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
    try:
        frame,rights,lefts = draw(frame,poses,0.1,output_transform)
        #flag_right,flag_left = pickup(frame_dummy,rights,lefts.model_c)
        for right,left in zip(rights,lefts):
            img_right = frame_dummy[right[1]:right[3],right[0]:right[2]]
            img_left = frame_dummy[left[1]:left[3],left[0]:left[2]]  
            print(img_right.shape)

            #img_right = preprocess(img_right,size_c)
            #img_left = preprocess(img_left,size_c)
            #flag_right=classifier(img_right,input_layer_c,output_layer_c,exec_net_c)
            #flag_left=classifier(img_left,input_layer_c,output_layer_c,exec_net_c)
            img_right = preprocess_tensorflow(img_right)
            img_left = preprocess_tensorflow(img_left)
            flag_right = classifier_tensorflow(img_right,model_c)
            flag_left = classifier_tensorflow(img_left,model_c)
            print(flag_right)
            print(flag_left)
            print(img_left.shape)
            if flag_right[0][0]<0.5:
                print("picked_10")
            if flag_left[0][0]<0.5:
                print("picked_9")   
    except Exception as e:
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
    
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    for id,bbox in tracks_draw.items():
        cv2.putText(frame, str(id), bbox[:2], 1, cv2.FONT_HERSHEY_DUPLEX, (0, 0, 255), 3)
        cv2.rectangle(frame, bbox[:2], bbox[2:], (0, 255, 0), 1)
    cv2.putText(frame, "Fps:-"+str(fps), (20, 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    output.write(frame)
    cv2.imshow('smart store', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
output.release()
cv2.destroyAllWindows()
