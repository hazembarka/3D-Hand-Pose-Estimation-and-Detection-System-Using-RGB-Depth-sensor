import cv2
import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
from collections import defaultdict
import imutils
from imutils.video import VideoStream
import time
from django_project.Object_detectin_part.utils import label_map_util


MARGIN = 1/200
#=====================================================Utilities for object detector.===========================================================
#==============================================================================================================================================
detection_graph = tf.Graph()

TRAINED_MODEL_DIR = 'django_project/Object_detectin_part/detection_frozen_graphs'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = TRAINED_MODEL_DIR + '/ssd5_optimized_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = TRAINED_MODEL_DIR + '/hand_label_map.pbtxt'
NUM_CLASSES = 1
# load label map using utils provided by tensorflow object detection api
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load a frozen infrerence graph into memory
def load_inference_graph():

    # load frozen tensorflow model into memory
    print("> ====== Loading frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")
    return detection_graph, sess

# Show fps value on image.
def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)



num_hands_detect = 1
score_thresh = 0.2
im_width = 640
im_height = 480

def detection_results(detection_graph,frame,sess,score_threshold=0.2):
    global score_thresh
    score_thresh = score_threshold
    boxes, scores, classes = detect_objects(frame, detection_graph, sess)
    frame_with_infos,cropped,limits = draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, frame)

    return frame_with_infos,cropped,limits




# Drawing bounding boxes and distances onto image
def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, classes,
 im_width, im_height, image_np):
    color = None
    color0 = (0,0,255)
    color1 = (0,50,255)
    cropped_hand = np.zeros((200,200,3))
    [xmin,xmax,ymin,ymax]=[0,im_height,0,im_width]
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            if classes[i] == 1: id = 'open'
            if classes[i] == 2:
                id ='closed'
            if i == 0: color = color0
            else: color = color1

            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
           # print("left :" ,left,"   right: ",right,"    top : " ,top,"  bottom: " ,bottom)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            
            #cropping the hand box to the keypoints detector :
            cropped_hand = None

            xmin = max(int(top) - int((bottom-top)*MARGIN),0)
            xmax = min(int(bottom) + int((bottom-top)*MARGIN),im_height)
            ymin = max(int(left) - int((right-left)*MARGIN),0)
            ymax = min(int(right) + int((right-left)*MARGIN),im_width)

            cropped_hand = np.copy(image_np[xmin : xmax, ymin:ymax ,:])
            
            cv2.rectangle(image_np, p1, p2, color , 3, 1)
            cv2.putText(image_np, 'hand '+str(i)+': '+id, (int(left), int(top)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , color, 2)
            cv2.putText(image_np, 'confidence: '+str("{0:.2f}".format(scores[i])),
                        (int(left),int(top)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    return image_np,cropped_hand,[xmin,xmax,ymin,ymax]

def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)