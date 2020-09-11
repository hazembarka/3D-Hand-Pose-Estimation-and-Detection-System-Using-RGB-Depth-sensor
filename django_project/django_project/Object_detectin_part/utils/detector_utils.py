# Utilities for object detector.

import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from django_project.Object_detectin_part.utils import label_map_util
from collections import defaultdict
import pyautogui

detection_graph = tf.Graph()

TRAINED_MODEL_DIR = 'frozen_graphs'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = TRAINED_MODEL_DIR + '/ssd5_optimized_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = TRAINED_MODEL_DIR + '/hand_label_map.pbtxt'

NUM_CLASSES = 2
# load label map using utils provided by tensorflow object detection api
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
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


# Drawing bounding boxes and distances onto image
def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, classes,
 im_width, im_height, image_np,avg_width=10,avg_hight=20,focalLength=930):
    color = None
    color0 = (255,0,0)
    color1 = (0,50,255)
    cropped_hand = np.zeros((200,200,3))
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

            xmin = max(int(top) - int((bottom-top)/2),0)
            xmax = min(int(bottom) + int((bottom-top)/2),im_height)
            ymin = max(int(left) - int((right-left)/2),0)
            ymax = min(int(right) + int((right-left)/2),im_width)

            cropped_hand = np.copy(image_np[xmin : xmax, ymin:ymax ,:])

            dist  = distance_to_camera(avg_width, focalLength, int(bottom-top),int(right-left),id)
            
            #pyautogui.dragTo(100, 150)

            cv2.rectangle(image_np, p1, p2, color , 3, 1)

            cv2.putText(image_np, 'hand '+str(i)+': '+id, (int(left), int(top)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , color, 2)

            cv2.putText(image_np, 'confidence: '+str("{0:.2f}".format(scores[i])),
                        (int(left),int(top)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            cv2.putText(image_np, 'distance: '+str("{0:.2f}".format(dist)+' cm'),
                        (int(im_width*0.7),int(im_height*0.9+30*i)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return cropped_hand 


# Show fps value on image.
def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
# compute and return the distance from the hand to the camera using triangle similarity
def distance_to_camera(knownWidth, focalLength, pixelHight_V,pixelWidth_H,hand_status):
    if hand_status=='closed':
        pixelWidth = min(pixelWidth_H,pixelHight_V)
    else:
        pixelWidth = min(pixelWidth_H,pixelHight_V)
    print('Hand : ',hand_status,'  Proportion = ',(pixelHight_V / pixelWidth_H),'    Horizontal = ',int((knownWidth * focalLength) / pixelWidth_H),'      H = ',pixelWidth_H,'     Vertical = ' ,int((knownWidth * focalLength) / pixelHight_V),'   V = ',pixelHight_V)
    return knownWidth * focalLength / pixelWidth

# Actual detection .. generate scores and bounding boxes given an image
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

