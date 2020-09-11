
import sys
import numpy as np
import tensorflow as tf
import cv2

import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

from primesense import openni2
from primesense import _openni2 as c_api


from django_project.Hand_pose_estimation_part.data.importers import DepthImporter
from django_project.Hand_pose_estimation_part.netlib.basemodel import basenet2
from django_project.Hand_pose_estimation_part.util.realtimehandposepipeline import RealtimeHandposePipeline
from django_project.Object_detectin_part.hand_detection_RGBD import detection_results




##===================================Overview======================================================

# The system is composed from 2 main steps  :
#  1-  Hand Detection : to crop the area of interest containing the hand
#  2-  Hand Pose Estimation : to get the (X,Y) in pixels and (Z) in mm coordinates for each hand joint
# 	   using the cropped image, the final output is the araay J_Pixels

# PS : -the function model.run_sess is responsible for all the process
#
# 	   -we can add more functions to the system to exploit J_Pixels' information like getting the lowest
# 	   joint in term of depth and store it in a JSON file to controle the mose by finger mouvment
# 	   
#      - Object detection can be done by training an SSD mobilenet detector and using the same approach 
#	   of the RGB-D Hand Detector so that the 3 models can run simultaneously 





#=========================Depth parameters========================================================
EXACT_TABLE_CAMERA_DISTANCE = 2000	  # in mm, greater depth values will be replaced by this one
DEPTH_THRESHOLD = 200 # in mm     ,  smaller depth values will be replaced by this one



## depth correction Coefs in case there are some problems

CORRECTION_COEF_1_AFTER_LOADING = 1
DEPTH_CORRECTION = 1 
ADD_RANGE_CORRECTION = 0 












## A function to rescale depth values from maximum-minimum to 0-255
## RGB-D Hand detector uses RGB(0-255) encoded format instead of 1 depth channel with mm values
min_depth_for_scaling = 0   # changes for each image
max_depth_for_scaling = 1500
def Depth_to_255_scale(entry):
    if entry>max_depth_for_scaling or entry<50:
        return 0
    else :
        return 255-((entry-min_depth_for_scaling)/(max_depth_for_scaling-min_depth_for_scaling)*255)

Depth_to_255_scale = np.vectorize(Depth_to_255_scale)










## A function to remove depth noise (noisy values are zeros)
def remove_noise(entry):
	global m
	mapper_dict = {0:m}   ## noise replaced by maximum depth in every frame

	return mapper_dict[entry] if entry in mapper_dict else entry

remove_noise = np.vectorize(remove_noise)
m = 0


def DepthMapCorrection(depth):
		"""
		Correcting the depth map
		:return: image data of depth image
		"""
		global m,MAXIMUM_DEPTH_RANGE,CORRECTION_COEF_1_AFTER_LOADING,DEPTH_CORRECTION, min_depth_for_scaling
		
		## Storing maximum depth
		m = np.max(depth)
		## removing noise
		depth = remove_noise(depth)
		##storing minimum depth 
		min_depth_for_scaling = np.min(depth)
		## Converting to a numpy array
		imgdata = np.asarray(depth, np.float32)/CORRECTION_COEF_1_AFTER_LOADING*DEPTH_CORRECTION + ADD_RANGE_CORRECTION

		return imgdata



## The hand Pose estimation Model :
class model_setup():
	def __init__(self,dataset,model_path):
		self._dataset=dataset
		self.model_path=model_path
		self.inputs = tf.placeholder(dtype=tf.float32, shape=(None, 96, 96, 1))  ## Image size is fixed
		self.hand_tensor=None
		self.model()
		self.saver = tf.train.Saver(max_to_keep=15)

### the model's configurations
	def __self_dict(self):  
		return (14,9,5)


	def __config(self):
		flag=-1
		di = DepthImporter(fx=535.4, fy=flag*539.2, ux=320.1, uy=247.6)
		config = {'fx': di.fx, 'fy': abs(di.fy), 'cube': (250,250, 250), 'im_size': (96, 96)}
		return di, config

	def __crop_cube(self):
		return self.__config()[1]['cube'][0]
	def __joint_num(self):
		return self.__self_dict()[0]

## the model's initialization
	def model(self):
		outdims=self.__self_dict()
		fn = layers.l2_regularizer(1e-5)
		fn0 = tf.no_regularizer
		with slim.arg_scope([slim.conv2d, slim.fully_connected],
							weights_regularizer=fn,
							biases_regularizer=fn0, normalizer_fn=slim.batch_norm):
			with slim.arg_scope([slim.batch_norm],
								is_training=False,
								updates_collections=None,
								decay=0.9,
								center=True,
								scale=True,
								epsilon=1e-5):
				pred_comb_ht, pred_comb_hand, pred_hand, pred_ht = basenet2(self.inputs, kp=1, is_training=False,
																			outdims=outdims)

		self.hand_tensor=pred_hand



### the entire process :

	def sess_run(self,detection_graph=0,detection_sess=0):
		
		## rtp library, containing some useful functions
		_di,_config=self.__config()

		rtp = RealtimeHandposePipeline(1, config=_config, di=_di, verbose=False, comrefNet=None,maxDepth=EXACT_TABLE_CAMERA_DISTANCE,minDepth=DEPTH_THRESHOLD)
		joint_num=self.__joint_num()  ## 14 joints
		cube_size=self.__crop_cube()  ##  to rescale the cropped hand image
		

		
		with tf.Session() as sess:
			init = tf.global_variables_initializer()
			sess.run(init)
			self.saver.restore(sess, self.model_path)


			### Initialize OpenNI with its libraries

			frameRate = 30
			width = 640 		# Width of image
			height = 480		# height of image
			openni2.initialize("django_project\\orbbec-astra\\Redist") #The OpenNI2 Redist folder
			# Open a device
			dev = openni2.Device.open_any()
			# Open two streams: one for color and one for depth
			depth_stream = dev.create_depth_stream()
			depth_stream.start()
			depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX = width, resolutionY = height, fps = frameRate))

			color_stream = dev.create_color_stream()
			color_stream.start()
			color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX = width, resolutionY = height, fps = frameRate))

			# Register both streams so they're aligned
			dev.set_image_registration_mode(c_api.OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR)
			# Set exposure settings
			# Camera Exposure
			bAutoExposure = Fals
			exposure = 100
			if (color_stream.camera != None):
				color_stream.camera.set_auto_exposure(bAutoExposure)
				color_stream.camera.set_exposure(exposure)
			
		

			## ===================================main loop==============================================
			while True:

				## Loading and correcting depth frame
				frame = depth_stream.read_frame()
				frame_data = frame.get_buffer_as_uint16()
				depthPix = np.frombuffer(frame_data, dtype=np.uint16)
				depthPix.shape = (height, width)
				
				depthPix = DepthMapCorrection(depthPix)

				## ____________________RGB-D based hand detection :____________________________________
				# transform 1 channel depth image to RGB channels  
				detection_frame = np.array(Image.fromarray(Depth_to_255_scale(depthPix)).convert('RGB'))

				##Detection :
				processed_image, cropped,[xmin,xmax,ymin,ymax] = detection_results( detection_graph,detection_frame ,detection_sess,score_threshold=0.2)                
				cv2.imshow('detection_frame',processed_image)

				depthPix_original = depthPix.copy()
				depthPix = depthPix[xmin : xmax, ymin:ymax]
				cv2.imshow('cropped',cropped)
				## returning cropped depth image :depthPix  (original size 200x200 : change it in the hand detection folder )

				try:
					## loc is the center's coordinates of our bounding box in (pixels,pixels,mm), used to make the center of mass of the hand closer to it instead of the closest object to the camera
					loc_rgbd_detector = [depthPix.shape[0]//2,depthPix.shape[1]//2,depthPix[depthPix.shape[0]//2,depthPix.shape[1]//2]]
					
					## Countour based Hand detection + image preprocessing + the hand's center of mass com3D
					crop1, M, com3D,bounds = rtp.detect(depthPix,loc_rgbd_detector)
					if com3D[0]==0:
						continue # skipping errors during hand detection using COM and contours
					
					crop = crop1.reshape(1, crop1.shape[0], crop1.shape[1], 1).astype('float32')
					
					##____________________________Hand pose estimation______________________________________
					pred_ = sess.run(self.hand_tensor, feed_dict={self.inputs: crop})
					norm_hand = np.reshape(pred_, (joint_num, 3))
					
					## from normalized and relative to com3D to global coordinates
					pose = norm_hand * cube_size / 2. + com3D

					## Show results
					img,J_pixels = rtp.show3(depthPix_original, pose,self._dataset,(np.float32(xmin), np.float32(ymin)))
					cv2.imshow('frame', img)
					cv2.imshow('crop1', crop1)
					
					print('com3D :',com3D,' pose[0] :',pose[0],' pixels[0] ' ,J_pixels[0])
				except: 
					print('error happened')
					continue
				if cv2.waitKey(1) >= 0:
					break
		openni2.unload()

		cv2.destroyAllWindows()
				
