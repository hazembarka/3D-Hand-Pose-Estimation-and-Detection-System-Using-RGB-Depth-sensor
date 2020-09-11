
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
from django_project.Object_detectin_part.hand_detection import detection_results






#=========================Depth parameters========================================================
EXACT_TABLE_CAMERA_DISTANCE = 1500	  # in mm
DEPTH_THRESHOLD = 200 # in mm


CORRECTION_COEF_1_AFTER_LOADING = 1
DEPTH_CORRECTION = 1 
ADD_RANGE_CORRECTION = 0 














def mp(entry):
	global m
	mapper_dict = {0:m}

	return mapper_dict[entry] if entry in mapper_dict else entry

mp = np.vectorize(mp)

m = 0


def loadDepthMap(depth):
		"""
		Read a depth-map
		:param filename: file name to load
		:return: image data of depth image
		"""
		global m,MAXIMUM_DEPTH_RANGE,CORRECTION_COEF_1_AFTER_LOADING,DEPTH_CORRECTION
		m = np.max(depth)

		depth = mp(depth)
		dpt = depth
		imgdata = np.asarray(dpt, np.float32)/CORRECTION_COEF_1_AFTER_LOADING*DEPTH_CORRECTION + ADD_RANGE_CORRECTION

		return imgdata

class model_setup():
	def __init__(self,dataset,model_path):
		self._dataset=dataset
		self.model_path=model_path
		self.inputs = tf.placeholder(dtype=tf.float32, shape=(None, 96, 96, 1))

		self.hand_tensor=None
		self.model()
		self.saver = tf.train.Saver(max_to_keep=15)

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

	def sess_run(self,detection_graph=0,detection_sess=0):
		_di,_config=self.__config()

		rtp = RealtimeHandposePipeline(1, config=_config, di=_di, verbose=False, comrefNet=None,maxDepth=EXACT_TABLE_CAMERA_DISTANCE,minDepth=DEPTH_THRESHOLD)

		joint_num=self.__joint_num()
		cube_size=self.__crop_cube()

		
		with tf.Session() as sess:
			init = tf.global_variables_initializer()
			sess.run(init)
			self.saver.restore(sess, self.model_path)


			frameRate = 30
			width = 640 		# Width of image
			height = 480		# height of image
			openni2.initialize("django_project\\orbbec-astra\\Redist") #The OpenNI2 Redist folder

			# Initialize OpenNI with its libraries

			# Open a device
			dev = openni2.Device.open_any()
			# Check to make sure it's not None

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
			bAutoExposure = False
			exposure = 100
			if (color_stream.camera != None):
				color_stream.camera.set_auto_exposure(bAutoExposure)
				color_stream.camera.set_exposure(exposure)
			
		

			
			while True:
				frame = depth_stream.read_frame()
				frame_data = frame.get_buffer_as_uint16()
				depthPix = np.frombuffer(frame_data, dtype=np.uint16)
				depthPix.shape = (height, width)
				
				# This depth is in 1 mm units
				#depthPix = cv2.resize(depthPix[80:400,80:560],(640,480))

				frame = color_stream.read_frame()
				frame_data = frame.get_buffer_as_uint8()
				colorPix = np.frombuffer(frame_data, dtype=np.uint8)
				colorPix.shape = (height, width, 3)

				#colorPix = cv2.resize(colorPix[80:400,80:560,:],(640,480))

				#colorPix = cv2.cvtColor(colorPix, cv2.COLOR_BGR2RGB)
				depthPix = loadDepthMap(depthPix)
				processed_image, cropped,[xmin,xmax,ymin,ymax] = detection_results( detection_graph,colorPix ,detection_sess,score_threshold=0.2)                
				depthPix_original = depthPix.copy()
				depthPix = depthPix[xmin : xmax, ymin:ymax]
				cv2.imshow('cropped',cropped)
				
				cv2.imshow('Detection',cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

				#try:
				if True:
					crop1, M, com3D,bounds = rtp.detect(depthPix)
					if com3D[0]==0:
						continue # skipping error happened during hand detection using COM and contours
					crop = crop1.reshape(1, crop1.shape[0], crop1.shape[1], 1).astype('float32')

					pred_ = sess.run(self.hand_tensor, feed_dict={self.inputs: crop})
					norm_hand = np.reshape(pred_, (joint_num, 3))
					pose = norm_hand * cube_size / 2. + com3D
	
					img,J_pixels = rtp.show3(depthPix_original, pose,self._dataset,(np.float32(xmin), np.float32(ymin)))
					cv2.imshow('frame', img)
					cv2.imshow('crop1', crop1)

					print('com3D :',com3D,' pose[0] :',pose[0],' pixels[0] ' ,J_pixels[0])
				#except: 
				#	print('error happened')
				#	continue
				if cv2.waitKey(1) >= 0:
					break
		openni2.unload()

		cv2.destroyAllWindows()
				
