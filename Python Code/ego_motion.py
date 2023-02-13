import os
import cv2
import numpy as np
import tensorflow as tf
import sys

import serial
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
from speedy import estimateSpeed
from stitch import stitch
#arduino = serial.Serial('COM9',9600)

global front_camera 
front_camera = 'Safe'
global back_camera 
back_camera = 'Safe'

global left_camera 
left_camera = 'Safe'
global right_camera 
right_camera = 'Safe'

global ignore 
ignore = 4

width = 640
height = 480


# initial coordinates

prev_ymin = 0
prev_xmin = 0
prev_ymax = 0
prev_xmax = 0


MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')
NUM_CLASSES = 90
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

def cameras_detection(frame,a):
    
    global ignore
    global front_camera
    global back_camera
    global left_camera
    global right_camera

    if a == 1:
        front_camera = 'Safe'
    if a == 2:
        back_camera = 'Safe'
    if a == 3:
        left_camera = 'Safe'
    if a == 4:
        right_camera = 'Safe'

    frame_expanded = np.expand_dims(frame, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=5,
        min_score_thresh=0.6)
    for i, b in enumerate(boxes[0]):
        if scores[0][i] >= 0.6:
            mid_x = (boxes[0][i][3] + boxes[0][i][1])/2
            mid_y = (boxes[0][i][2] + boxes[0][i][0])/2

            ymin = boxes[0][i][0]*height
            xmin = boxes[0][i][1]*width
            ymax = boxes[0][i][2]*height
            xmax = boxes[0][i][3]*width
            
            speed = estimateSpeed([ymin,xmin,ymax,xmax],[prev_ymin,prev_xmin,prev_ymax,prev_xmax])
            speed = speed - 30
            
            if speed < 0 :
                speed = 0
            
            ymin = prev_ymin
            xmin = prev_xmin
            xmax = prev_xmax
            xmin = prev_xmin
            
            cv2.putText(frame, str(int(speed)/10) + " km/hr",(int(mid_x*700)-50, int(mid_y*600)) ,cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            apx_distance = 1-(boxes[0][i][3] - boxes[0][i][1])
            
            if a == 1:
                if mid_x > 0.3 and mid_x < 0.7:
                    if apx_distance < 0.75:  
                        cv2.putText(frame, 'Front_Warning !!!', (int(mid_x*height)-50, int(mid_y*width)),font,1,(0,0,255),4)
                        front_camera = 'Warning'
                    else:
                        cv2.putText(frame, 'Safe !!!', (int(mid_x*height)-50, int(mid_y*width)),font,1,(0,0,255),4)
                        front_camera = 'Safe'
            if a == 2:        #back camera camera
                if apx_distance < 0.25:  
                    cv2.putText(frame, 'Back_Warning !!! ', (int(mid_x*height)-50, int(mid_y*width)),font,1,(0,0,255),4)
                    back_camera = 'Warning'
                else:
                    cv2.putText(frame, 'Safe', (int(mid_x*height)-50, int(mid_y*width)),font,1,(0,0,255),4)
                    back_camera = 'Safe'

            if a == 3:        #left camera
                if apx_distance < 0.25:  
                    cv2.putText(frame, 'Left_Warning !!!', (int(mid_x*height)-50, int(mid_y*width)),font,1,(0,0,255),4)
                    left_camera = 'Warning'
                else:
                    cv2.putText(frame, 'Safe', (int(mid_x*height)-50, int(mid_y*width)),font,1,(0,0,255),4)
                    left_camera = 'Safe'

            if a == 4:        #right camera
                if apx_distance < 0.25:  
                    cv2.putText(frame, 'Right_Warning !!!', (int(mid_x*height)-50, int(mid_y*width)),font,1,(0,0,255),4)
                    right_camera = 'Warning'
                else:
                    cv2.putText(frame, 'Safe', (int(mid_x*height)-50, int(mid_y*width)),font,1,(0,0,255),4)
                    right_camera = 'Safe'

            if ignore > 0 :
                ignore = ignore -1
                pass
            else:
                if front_camera == 'Warning' and back_camera == 'Warning' and left_camera == 'Warning' and right_camera == 'Warning' :
                    print('Stop')
                    #arduino.write(str.encode('S'))    #Stop the Car
            
                elif front_camera == 'Warning' and back_camera == 'Warning' and left_camera == 'Warning' and right_camera == 'Safe' :
                    print('Right')
                    #arduino.write(str.encode('R'))       # Turn right to avoid obstacle

                elif front_camera == 'Warning' and back_camera == 'Safe' and left_camera == 'Warning' and right_camera == 'Safe' :
                    print('Right')
                    #arduino.write(str.encode('R'))       # Turn right to avoid obstacle   
            
                elif front_camera == 'Warning' and back_camera == 'Warning' and left_camera == 'Safe' and right_camera == 'Warning' :
                    print('Left')
                    #arduino.write(str.encode('L'))       # Turn left to avoid obstacle     

                elif front_camera == 'Warning' and back_camera == 'Safe' and left_camera == 'Safe' and right_camera == 'Warning' :
                    print('Left')
                    #arduino.write(str.encode('L'))       # Turn left to avoid obstacle  

                elif front_camera == 'Warning' and back_camera == 'Safe' and left_camera == 'Safe' and right_camera == 'Safe' :
                    print('Left')
                    #arduino.write(str.encode('L'))       # Turn left to avoid obstacle 

                elif front_camera == 'Warning' and back_camera == 'Warning' and left_camera == 'Safe' and right_camera == 'Safe' :
                    print('Left')
                    #arduino.write(str.encode('L'))       # Turn left to avoid obstacle 

                elif front_camera == 'Warning' and back_camera == 'Safe' and left_camera == 'Warning' and right_camera == 'Warning' :
                    print('Backward')
                    #arduino.write(str.encode('B'))       # Backward car to avoid obstacle             
                else:
                    print('Straight')
                    #arduino.write(str.encode('F'))

    return frame

front_cam = cv2.VideoCapture('g:/fyp/road_video.mp4')
left_cam   = cv2.VideoCapture('g:/fyp/road_video.mp4')
right_cam  = cv2.VideoCapture('g:/fyp/road_video.mp4')
back_cam  = cv2.VideoCapture('g:/fyp/1.MKV')

while(True):
    t1 = cv2.getTickCount()
    
    ret1,front = front_cam.read()
    ret2,left = left_cam.read()
    ret3,right = right_cam.read()
    ret4,back = back_cam.read()
    if ret1 and ret2 and ret3 and ret4:

        front = cameras_detection(front,1)
        back = cameras_detection(back,2)
        left = cameras_detection(left,3)
        right = cameras_detection(right,4)

        frame1 = stitch(cv2.resize(front,(350,300)),cv2.resize(back,(350,300)))
        frame2 = stitch(cv2.resize(left,(350,300)),cv2.resize(right,(350,300)))
        frame = stitch(frame1,frame2)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
        cv2.imshow('Front--Back--Left--Right',frame)
        #print(np.shape(frame))

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        if cv2.waitKey(1) == 27:
            break

front_cam.release()
left_cam.release()
right_cam.release()
back_cam.release()

cv2.destroyAllWindows()
arduino.write(str.encode('S'))

