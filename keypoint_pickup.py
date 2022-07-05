
import cv2
import torch
from motrackers import CentroidTracker
import numpy as np
from openvino.inference_engine import IECore
import openvino_models as models
from urtils import load,id_assign
from urtils import postprocess_yolov5,draw_tracks,tracking,check_id,pointInRect
from motrackers import CentroidTracker

default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

colors = (
        (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85),
        (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0),
        (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
        (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255),
        (0, 170, 255))


def draw_pose(img,poses, point_score_threshold,output_transform,skeleton=default_skeleton,draw_ellipses=False):
    """draw_pose is fuction which use for drawing pose on image

    Args:
        img ([array]): [description]
        poses ([array]): [description]
        point_score_threshold ([array]): [description]
        output_transform ([function]): [description]
        skeleton ([type], optional): [description]. Defaults to default_skeleton.
        draw_ellipses (bool, optional): [description]. Defaults to False.

    Returns:
        img: with draw poses
    """
    
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
                pass
        cv2.circle(frame,points[4],10,(0,0,0),-1)    
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

#---------------------------tracking---------------------------------------
tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
#--------------------------------------------------------------------------

#----------------------------pose------------------------------------------
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
#---------------------------------------------------------------------------------------------------
#--------------------------------------------yolov5--------------------------------------------
model_yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = model_yolov5.names
model_yolov5.to(device)    
#-----------------------------------------end--------------------------------------------------
cap = cv2.VideoCapture("http://192.168.0.104:8080/video")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width,frame_height)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
output = cv2.VideoWriter('output9.mp4', fourcc, fps, (1000,1000))
count=0
tracks_id={}
tracklets_id={}
while True:
    _,frame = cap.read()
    ids = {}
    tracks_draw={}
    id_={"":""}
    frame = cv2.resize(frame, (1000,1000))
    
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    frame1 = [frame]
    det = model_yolov5(frame1)
    #--------------------------------------tracking--------------------------------------------------------
    bboxes, scores,labels,bboxes_bottel,scores_bottel,labels_bottel=postprocess_yolov5(x_shape, y_shape,det)
    tracks = tracker.update(bboxes, scores,labels)
        
    frame,ids = draw_tracks(frame, tracks,ids)
       
    tracks_draw,tracks_id,tracklets_id=tracking(ids,frame,tracks_id,tracklets_id,tracks_draw)
    #-------------------------------------------------------------------------------------------------------
    #------------------------------------pose---------------------------------------------------------------
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
    
    frame = draw_pose(frame,poses,0.1,output_transform)
    #-------------------------------------------------------------------------------------------------------
    #-----------------------------------bottle------------------------------------------------------------
    labels, cord = det.xyxyn[0][:, -1], det.xyxyn[0][:, :-1]
    
    X = int(x_shape * (20/100))
    start_point = (X,0)
    end_point = (X,y_shape)
    id_bottle={}
    bottle_number=0
    for i in range(len(cord)):
        bbox=cord[i]
        c=labels[i]
        
        if int(c)==39:
            x1_, y1_, x2_, y2_ = int(bbox[0] * x_shape), int(bbox[1] * y_shape), int(bbox[2] * x_shape), int(bbox[3] * y_shape)
            center_right = (int((x2_+x2_)/2),int((y2_+y1_)/2))
            bgr = (0, 255, 0)
            #if x1_>=X:
            id_bottle=id_assign(center_right,poses,output_transform,id_bottle,bottle_number)
            cv2.circle(frame,center_right,5,(0,0,0),-1)
            cv2.rectangle(frame, (x1_, y1_), (x2_, y2_), bgr, 2)
            bottle_number+=1
    bottle_number=0
    #---------------------------------------------------------------------------------------------------------------
    #print(id_bottle)
    #-------------------------------------------------draw---------------------------------------------------------
    pickup_draw={}
    for i in id_bottle.keys():
        pose = poses[int(i[0])]
        points = pose[:,:2].astype(np.int32)
        points = output_transform.scale(points)
        point_0 = points[0]
   
        for id,bbox in tracks_draw.items():
            flag = pointInRect(point_0,bbox)
            if flag==1:
                pickup_draw[str(id)+i[-1]]=flag
    #print(pickup_draw)
    for id,bbox in tracks_draw.items():
        cv2.putText(frame, str(id), bbox[:2], 1, cv2.FONT_HERSHEY_DUPLEX, (0, 0, 255), 3)
        cv2.rectangle(frame, bbox[:2], bbox[2:], (0, 255, 0), 1)

    pick_total={}
    for pick,flag in pickup_draw.items():
        if pick[0] not in pick_total:
            pick_total[pick[0]]=flag
        else:
            pick_total[pick[0]]=flag+ pick_total[pick[0]]
    print(pick_total)
    y=20
    for person,number in pick_total.items():
        color = colors[int(person)]
        cv2.putText(frame, "person number"+person+" : "+"pick up "+str(number)+" bottle", (20,y), 1, cv2.FONT_HERSHEY_DUPLEX, color, 3)
        y+=30
        
    #---------------------------------------------------------------------------------------------------------------
    output.write(frame)
    cv2.imshow('smart store', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
output.release()
cv2.destroyAllWindows()