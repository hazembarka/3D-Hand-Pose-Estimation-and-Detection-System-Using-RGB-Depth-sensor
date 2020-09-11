
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


def getCrop( dpt, xstart, xend, ystart, yend ):
		"""
		Crop patch from image
		:param dpt: depth image to crop from
		:param xstart: start x
		:param xend: end x
		:param ystart: start y
		:param yend: end y

		:return: cropped image
		"""
		if len(dpt.shape) == 2:
			cropped = dpt[max(ystart, 0):min(yend, dpt.shape[0]), max(xstart, 0):min(xend, dpt.shape[1])].copy()
			# add pixels that are out of the image in order to keep aspect ratio
			cropped = np.pad(cropped, ((abs(ystart)-max(ystart, 0),
										   abs(yend)-min(yend, dpt.shape[0])),
										  (abs(xstart)-max(xstart, 0),
										   abs(xend)-min(xend, dpt.shape[1]))), mode='constant', constant_values=0)
		elif len(dpt.shape) == 3:
			cropped = dpt[max(ystart, 0):min(yend, dpt.shape[0]), max(xstart, 0):min(xend, dpt.shape[1]), :].copy()
			# add pixels that are out of the image in order to keep aspect ratio
			cropped = np.pad(cropped, ((abs(ystart)-max(ystart, 0),
										   abs(yend)-min(yend, dpt.shape[0])),
										  (abs(xstart)-max(xstart, 0),
										   abs(xend)-min(xend, dpt.shape[1])),
										  (0, 0)), mode='constant', constant_values=0)
		else:
			raise NotImplementedError()


		return cropped



def loadDepthMap(filename):
		"""
		Read a depth-map
		:param filename: file name to load
		:return: image data of depth image
		"""
		img = Image.open(filename)
		# top 8 bits of depth are packed into green channel and lower 8 bits into blue
		#assert len(img.getbands()) == 3
		plt.imshow(img)
		plt.show()
		print('Before max: ',np.max(img),' min: ',np.min(img))

		r, g, b = img.split()[:3]
		r = np.asarray(r, np.int32)
		g = np.asarray(g, np.int32)
		b = np.asarray(b, np.int32)
		dpt = np.bitwise_or(np.left_shift(g, 8), b)
		imgdata = np.asarray(dpt, np.float32)
		print('After shape :',imgdata.shape,' max: ',np.max(imgdata),' min: ',np.min(imgdata))

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
		di = DepthImporter(fx=475.268, fy=flag*475.268, ux=313.821, uy=246.075)
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

	def sess_run(self,detection_graph,detection_sess,imageFolder='E:\\dataset\\test\\'):
		_di,_config=self.__config()

		rtp = RealtimeHandposePipeline(1, config=_config, di=_di, verbose=False, comrefNet=None)

		joint_num=self.__joint_num()
		cube_size=self.__crop_cube()

	 
		
		with tf.Session() as sess:
			init = tf.global_variables_initializer()
			sess.run(init)
			self.saver.restore(sess, self.model_path)
			files = os.listdir(imageFolder)
			vs = cv2.VideoCapture(0)
			Stream_FPS = 20
			for f in files:
			   

				aaa ,color_frame = vs.read()
				color_frame = cv2.resize(color_frame, (640, 480))
				try:
					color_frame = cv2.cvtColor(cv2.flip(color_frame, 1), cv2.COLOR_BGR2RGB)
				except:
					print("Error converting to RGB")
				
				
				processed_image, cropped,[xmin,xmax,ymin,ymax] = detection_results( detection_graph,color_frame ,detection_sess)
				
				
				cv2.imshow('cropped',cropped)
				cv2.imshow('Detection',processed_image)
				depth_frame = loadDepthMap(os.path.join(imageFolder,f))
				print(os.path.join(imageFolder,f))
				frame2 = depth_frame.copy()
				#print(' shape :',depth_frame.shape,' max: ',np.max(depth_frame),' min: ',np.min(depth_frame))
				crop1, M, com3D,bounds = rtp.detect(frame2)

				frame3 = getCrop(frame2,bounds[0]-80,bounds[1]+80,bounds[2]-80,bounds[3]+80 )
				
				crop1, M, com3D,bounds = rtp.detect(frame3)
				crop = crop1.reshape(1, crop1.shape[0], crop1.shape[1], 1).astype('float32')
				pred_ = sess.run(self.hand_tensor, feed_dict={self.inputs: crop})
				norm_hand = np.reshape(pred_, (joint_num, 3))
				pose = norm_hand * cube_size / 2. + com3D
				img,J_pixels = rtp.show2(frame3, pose,self._dataset)
				img = rtp.addStatusBar(img)
				cv2.imshow('img',img)
				cv2.imshow('crop1',crop1)
				#print('bounds :',bounds1)
				print('com3D :',com3D,' pose[0] :',pose[0],' pixels[0] ',J_pixels[0])#,' max crop:',np.max(depth_frame) )
				cv2.imwrite(os.path.join(imageFolder,'modif'+f), img)
				cv2.imwrite(os.path.join(imageFolder,'modif'+f), crop1)
		
				if cv2.waitKey(1) >= 0:
					break
		cv2.destroyAllWindows()
				
