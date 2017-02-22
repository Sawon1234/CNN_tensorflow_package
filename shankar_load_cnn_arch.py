# ------------------------------------------------------------------------
# Module of functions for loading the various CNN Architecture Choices
# Author : Sukrit Shankar
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
# Necessary Imports
from __future__ import division, absolute_import

import os
import numpy as np
import scipy 
import scipy.io

import tensorflow as tf
from math import ceil

import tflearn
import tflearn.activations as activations 
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d, highway_conv_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import *
from tflearn.optimizers import Momentum
from tflearn.layers.normalization import batch_normalization
from tflearn.activations import relu

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Function For AlexNet
def load_alexnet(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation): 
	input_layer = input_data(shape=[None, cnn_image_shape[0], cnn_image_shape[1], cnn_image_shape[2]], data_preprocessing = cnn_img_prep, data_augmentation = cnn_img_aug)
	network = conv_2d(input_layer, 96, 11, strides=4, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = conv_2d(network, 256, 5, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = conv_2d(network, 384, 3, activation='relu')
	network = conv_2d(network, 384, 3, activation='relu')
	network = conv_2d(network, 256, 3, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)

	network = fully_connected(network, 4096, activation='tanh')
	network = dropout(network, cnn_keep_probability)
	network = fully_connected(network, 4096, activation='tanh')
	network = dropout(network, cnn_keep_probability)
	loss_layer = fully_connected(network, num_output_classes, regularizer=cnn_regularization_type, weight_decay=cnn_regularization_weight_decay, activation=cnn_loss_layer_activation)

	# Return net
	return loss_layer

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Function For GoogleNet Inception V3
def load_googlenet_v1(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation):
	# Input Layer 
	input_layer = input_data(shape=[None, cnn_image_shape[0], cnn_image_shape[1], cnn_image_shape[2]], data_preprocessing = cnn_img_prep, data_augmentation = cnn_img_aug)

	# Basic Conv Layers
	conv1_7_7 = conv_2d(input_layer, 64, 7, strides=2, activation='relu', name = 'conv1_7_7_s2')
	pool1_3_3 = max_pool_2d(conv1_7_7, 3,strides=2)
	pool1_3_3 = local_response_normalization(pool1_3_3)
	conv2_3_3_reduce = conv_2d(pool1_3_3, 64,1, activation='relu',name = 'conv2_3_3_reduce')
	conv2_3_3 = conv_2d(conv2_3_3_reduce, 192,3, activation='relu', name='conv2_3_3')
	conv2_3_3 = local_response_normalization(conv2_3_3)
	pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')

	# Inception Module 3a
	inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
	inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96,1, activation='relu', name='inception_3a_3_3_reduce')
	inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128,filter_size=3,  activation='relu', name = 'inception_3a_3_3')
	inception_3a_5_5_reduce = conv_2d(pool2_3_3,16, filter_size=1,activation='relu', name ='inception_3a_5_5_reduce' )
	inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name= 'inception_3a_5_5')
	inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
	inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')
	inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

	# Inception Module 3b
	inception_3b_1_1 = conv_2d(inception_3a_output, 128,filter_size=1,activation='relu', name= 'inception_3b_1_1' )
	inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
	inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3,  activation='relu',name='inception_3b_3_3')
	inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name = 'inception_3b_5_5_reduce')
	inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5,  name = 'inception_3b_5_5')
	inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
	inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1,activation='relu', name='inception_3b_pool_1_1')
	inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

	# Inception Module 4a
	pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
	inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
	inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
	inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
	inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
	inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
	inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
	inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')
	inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')

	# Inception Module 4b
	inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
	inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu', name='inception_4b_3_3_reduce')
	inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', name='inception_4b_3_3')
	inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu', name='inception_4b_5_5_reduce')
	inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4b_5_5')
	inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1,  name='inception_4b_pool')
	inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu', name='inception_4b_pool_1_1')
	inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')

	# Inception Module 4c
	inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu',name='inception_4c_1_1')
	inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_3_3_reduce')
	inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256,  filter_size=3, activation='relu', name='inception_4c_3_3')
	inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu', name='inception_4c_5_5_reduce')
	inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64,  filter_size=5, activation='relu', name='inception_4c_5_5')
	inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
	inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu', name='inception_4c_pool_1_1')
	inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3,name='inception_4c_output')

	# Inception Module 4d
	inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
	inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu', name='inception_4d_3_3_reduce')
	inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', name='inception_4d_3_3')
	inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu', name='inception_4d_5_5_reduce')
	inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4d_5_5')
	inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1,  name='inception_4d_pool')
	inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu', name='inception_4d_pool_1_1')
	inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], mode='concat', axis=3, name='inception_4d_output')

	# Inception Module 4e
	inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
	inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu', name='inception_4e_3_3_reduce')
	inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_4e_3_3')
	inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu', name='inception_4e_5_5_reduce')
	inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128,  filter_size=5, activation='relu', name='inception_4e_5_5')
	inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1,  name='inception_4e_pool')
	inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu', name='inception_4e_pool_1_1')
	inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5,inception_4e_pool_1_1],axis=3, mode='concat')

	# Max Pool 
	pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')

	# Inception Module 5a
	inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
	inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
	inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_5a_3_3')
	inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
	inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5,  activation='relu', name='inception_5a_5_5')
	inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1,  name='inception_5a_pool')
	inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1,activation='relu', name='inception_5a_pool_1_1')
	inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3,mode='concat')

	# Inception Module 5b
	inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1,activation='relu', name='inception_5b_1_1')
	inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu', name='inception_5b_3_3_reduce')
	inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384,  filter_size=3,activation='relu', name='inception_5b_3_3')
	inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu', name='inception_5b_5_5_reduce')
	inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce,128, filter_size=5,  activation='relu', name='inception_5b_5_5' )
	inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1,  name='inception_5b_pool')
	inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu', name='inception_5b_pool_1_1')
	inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3, mode='concat')

	# Average Pool 
	pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)

	# Dropout 
	pool5_7_7 = dropout(pool5_7_7, cnn_keep_probability)

	# Loss Layer 
	loss_layer = fully_connected(pool5_7_7, num_output_classes, regularizer=cnn_regularization_type, weight_decay=cnn_regularization_weight_decay, activation=cnn_loss_layer_activation)

	# Return net
	return loss_layer

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# GoogleLeNet wih Batch Normalization
def conv_2d_bn(incoming,nb_filter, filter_size, strides=1, padding='same', activation=None, bias=True, weights_init='uniform_scaling', bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True, restore=True, reuse=False, scope=None, name='Conv2D_BN'):
	return tflearn.relu(batch_normalization(conv_2d(incoming, 
		nb_filter, 
		filter_size, 
		strides=strides, 
		activation=activation,
		padding=padding,
		bias=bias,
		weights_init=weights_init,
		regularizer=regularizer,
		name=name))
	)
	# conv_bn = conv_2d(incoming, 
	# 	nb_filter, 
	# 	filter_size, 
	# 	strides=strides, 
	# 	name=name)
	# conv_bn = tflearn.batch_normalization(conv_bn)
	# conv_bn = tflearn.activation(conv_bn, 'relu')
	# # import pdb; pdb.set_trace();
	# # conv_bn = batch_normalization(conv_bn)
	# # conv_bn = activation(conv_bn,'relu')
	# return conv_bn

# ------------------ Main Function ------------------------
def load_googlenet_bn(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation):
	# Input Layer 
	input_layer = input_data(shape=[None, cnn_image_shape[0], cnn_image_shape[1], cnn_image_shape[2]], data_preprocessing = cnn_img_prep, data_augmentation = cnn_img_aug)

	# Basic Conv Layers
	conv1_7_7 = conv_2d_bn(input_layer, 64, 7, strides=2, activation='relu', name = 'conv1_7_7_s2')
	pool1_3_3 = max_pool_2d(conv1_7_7, 3,strides=2)
	# pool1_3_3 = local_response_normalization(pool1_3_3)
	conv2_3_3_reduce = conv_2d_bn(pool1_3_3, 64,1, activation='relu',name = 'conv2_3_3_reduce')
	conv2_3_3 = conv_2d_bn(conv2_3_3_reduce, 192,3, activation='relu', name='conv2_3_3')
	# conv2_3_3 = local_response_normalization(conv2_3_3)
	pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')

	# Inception Module 3a
	inception_3a_1_1 = conv_2d_bn(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
	inception_3a_3_3_reduce = conv_2d_bn(pool2_3_3, 96,1, activation='relu', name='inception_3a_3_3_reduce')
	inception_3a_3_3 = conv_2d_bn(inception_3a_3_3_reduce, 128,filter_size=3,  activation='relu', name = 'inception_3a_3_3')
	inception_3a_5_5_reduce = conv_2d_bn(pool2_3_3,16, filter_size=1,activation='relu', name ='inception_3a_5_5_reduce' )
	inception_3a_5_5 = conv_2d_bn(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name= 'inception_3a_5_5')
	inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
	inception_3a_pool_1_1 = conv_2d_bn(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')
	inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

	# Inception Module 3b
	inception_3b_1_1 = conv_2d_bn(inception_3a_output, 128,filter_size=1, activation='relu', name= 'inception_3b_1_1' )
	inception_3b_3_3_reduce = conv_2d_bn(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
	inception_3b_3_3 = conv_2d_bn(inception_3b_3_3_reduce, 192, filter_size=3,  activation='relu',name='inception_3b_3_3')
	inception_3b_5_5_reduce = conv_2d_bn(inception_3a_output, 32, filter_size=1, activation='relu', name = 'inception_3b_5_5_reduce')
	inception_3b_5_5 = conv_2d_bn(inception_3b_5_5_reduce, 96, filter_size=5, activation='relu', name = 'inception_3b_5_5')
	inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
	inception_3b_pool_1_1 = conv_2d_bn(inception_3b_pool, 64, filter_size=1,activation='relu', name='inception_3b_pool_1_1')
	inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

	# Inception Module 4a
	pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
	inception_4a_1_1 = conv_2d_bn(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
	inception_4a_3_3_reduce = conv_2d_bn(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
	inception_4a_3_3 = conv_2d_bn(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
	inception_4a_5_5_reduce = conv_2d_bn(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
	inception_4a_5_5 = conv_2d_bn(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
	inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
	inception_4a_pool_1_1 = conv_2d_bn(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')
	inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')

	# Inception Module 4b
	inception_4b_1_1 = conv_2d_bn(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
	inception_4b_3_3_reduce = conv_2d_bn(inception_4a_output, 112, filter_size=1, activation='relu', name='inception_4b_3_3_reduce')
	inception_4b_3_3 = conv_2d_bn(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', name='inception_4b_3_3')
	inception_4b_5_5_reduce = conv_2d_bn(inception_4a_output, 24, filter_size=1, activation='relu', name='inception_4b_5_5_reduce')
	inception_4b_5_5 = conv_2d_bn(inception_4b_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4b_5_5')
	inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1,  name='inception_4b_pool')
	inception_4b_pool_1_1 = conv_2d_bn(inception_4b_pool, 64, filter_size=1, activation='relu', name='inception_4b_pool_1_1')
	inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')

	# Inception Module 4c
	inception_4c_1_1 = conv_2d_bn(inception_4b_output, 128, filter_size=1, activation='relu',name='inception_4c_1_1')
	inception_4c_3_3_reduce = conv_2d_bn(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_3_3_reduce')
	inception_4c_3_3 = conv_2d_bn(inception_4c_3_3_reduce, 256,  filter_size=3, activation='relu', name='inception_4c_3_3')
	inception_4c_5_5_reduce = conv_2d_bn(inception_4b_output, 24, filter_size=1, activation='relu', name='inception_4c_5_5_reduce')
	inception_4c_5_5 = conv_2d_bn(inception_4c_5_5_reduce, 64,  filter_size=5, activation='relu', name='inception_4c_5_5')
	inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
	inception_4c_pool_1_1 = conv_2d_bn(inception_4c_pool, 64, filter_size=1, activation='relu', name='inception_4c_pool_1_1')
	inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3,name='inception_4c_output')

	# Inception Module 4d
	inception_4d_1_1 = conv_2d_bn(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
	inception_4d_3_3_reduce = conv_2d_bn(inception_4c_output, 144, filter_size=1, activation='relu', name='inception_4d_3_3_reduce')
	inception_4d_3_3 = conv_2d_bn(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', name='inception_4d_3_3')
	inception_4d_5_5_reduce = conv_2d_bn(inception_4c_output, 32, filter_size=1, activation='relu', name='inception_4d_5_5_reduce')
	inception_4d_5_5 = conv_2d_bn(inception_4d_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4d_5_5')
	inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1,  name='inception_4d_pool')
	inception_4d_pool_1_1 = conv_2d_bn(inception_4d_pool, 64, filter_size=1, activation='relu', name='inception_4d_pool_1_1')
	inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], mode='concat', axis=3, name='inception_4d_output')

	# Inception Module 4e
	inception_4e_1_1 = conv_2d_bn(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
	inception_4e_3_3_reduce = conv_2d_bn(inception_4d_output, 160, filter_size=1, activation='relu', name='inception_4e_3_3_reduce')
	inception_4e_3_3 = conv_2d_bn(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_4e_3_3')
	inception_4e_5_5_reduce = conv_2d_bn(inception_4d_output, 32, filter_size=1, activation='relu', name='inception_4e_5_5_reduce')
	inception_4e_5_5 = conv_2d_bn(inception_4e_5_5_reduce, 128,  filter_size=5, activation='relu', name='inception_4e_5_5')
	inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1,  name='inception_4e_pool')
	inception_4e_pool_1_1 = conv_2d_bn(inception_4e_pool, 128, filter_size=1, activation='relu', name='inception_4e_pool_1_1')
	inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5,inception_4e_pool_1_1],axis=3, mode='concat')

	# Max Pool 
	pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')

	# Inception Module 5a
	inception_5a_1_1 = conv_2d_bn(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
	inception_5a_3_3_reduce = conv_2d_bn(pool4_3_3, 160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
	inception_5a_3_3 = conv_2d_bn(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_5a_3_3')
	inception_5a_5_5_reduce = conv_2d_bn(pool4_3_3, 32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
	inception_5a_5_5 = conv_2d_bn(inception_5a_5_5_reduce, 128, filter_size=5,  activation='relu', name='inception_5a_5_5')
	inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1,  name='inception_5a_pool')
	inception_5a_pool_1_1 = conv_2d_bn(inception_5a_pool, 128, filter_size=1,activation='relu', name='inception_5a_pool_1_1')
	inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3,mode='concat')

	# Inception Module 5b
	inception_5b_1_1 = conv_2d_bn(inception_5a_output, 384, filter_size=1,activation='relu', name='inception_5b_1_1')
	inception_5b_3_3_reduce = conv_2d_bn(inception_5a_output, 192, filter_size=1, activation='relu', name='inception_5b_3_3_reduce')
	inception_5b_3_3 = conv_2d_bn(inception_5b_3_3_reduce, 384,  filter_size=3,activation='relu', name='inception_5b_3_3')
	inception_5b_5_5_reduce = conv_2d_bn(inception_5a_output, 48, filter_size=1, activation='relu', name='inception_5b_5_5_reduce')
	inception_5b_5_5 = conv_2d_bn(inception_5b_5_5_reduce,128, filter_size=5,  activation='relu', name='inception_5b_5_5' )
	inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1,  name='inception_5b_pool')
	inception_5b_pool_1_1 = conv_2d_bn(inception_5b_pool, 128, filter_size=1, activation='relu', name='inception_5b_pool_1_1')
	inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3, mode='concat')

	# Average Pool 
	pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)

	# Dropout 
	# pool5_7_7 = dropout(pool5_7_7, cnn_keep_probability)

	# Loss Layer 
	loss_layer = fully_connected(pool5_7_7, num_output_classes, 
		regularizer=cnn_regularization_type, 
		weight_decay=cnn_regularization_weight_decay, 
		activation=cnn_loss_layer_activation)

	# Return net
	return loss_layer

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Function For GoogleNet Inception V3
def load_googlenet_v3(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation):
	# Input Layer 
	input_layer = input_data(shape=[None, cnn_image_shape[0], cnn_image_shape[1], cnn_image_shape[2]], data_preprocessing = cnn_img_prep, data_augmentation = cnn_img_aug)

	# Basic Conv Layers
	network = conv_2d(input_layer, 32, 3, strides=2)
	network = batch_normalization(network)
  	network = tflearn.activation(network, 'relu')
	network = conv_2d(network, 32, 3)
	network = batch_normalization(network)
  	network = tflearn.activation(network, 'relu')
	network = conv_2d(network, 64, 3)
	network = batch_normalization(network)
  	network = tflearn.activation(network, 'relu')
	network = max_pool_2d(network,3,strides=2)
	network = conv_2d(network, 80, 1)
	network = batch_normalization(network)
  	network = tflearn.activation(network, 'relu')
	network = conv_2d(network, 192, 3)
	network = batch_normalization(network)
  	network = tflearn.activation(network, 'relu')
	network = max_pool_2d(network,3,strides=2)
	
	# Inception Wide Module 1
	network_1 = conv_2d(network, 64, 1)
	network_1 = batch_normalization(network_1)
  	network_1 = tflearn.activation(network_1, 'relu')

	network_2 = conv_2d(network, 48, 1)
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')
	network_2 = conv_2d(network_2, 64, 5)
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')

	network_3 = conv_2d(network, 64, 1)
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 96, 3)
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 96, 3)
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')

	network_4 = avg_pool_2d(network, kernel_size=3, strides=1)
	network_4 = conv_2d(network_4, 32, 1)
	network_4 = batch_normalization(network_4)
  	network_4 = tflearn.activation(network_4, 'relu')
	network = merge([network_1, network_2, network_3, network_4], axis=3, mode='concat')
	del network_1, network_2, network_3, network_4

	# Inception Wide Module 2 
	network_1 = conv_2d(network, 64, 1)
	network_1 = batch_normalization(network_1)
  	network_1 = tflearn.activation(network_1, 'relu')

	network_2 = conv_2d(network, 48, 1)
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')
	network_2 = conv_2d(network_2, 64, 5)
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')

	network_3 = conv_2d(network, 64, 1)
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 96, 3)
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 96, 3)
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')

	network_4 = avg_pool_2d(network, kernel_size=3, strides=1)
	network_4 = conv_2d(network_4, 64, 1)
	network_4 = batch_normalization(network_4)
  	network_4 = tflearn.activation(network_4, 'relu')
	network = merge([network_1, network_2, network_3, network_4], axis=3, mode='concat')
	del network_1, network_2, network_3, network_4
	
	# Inception Wide Module 3 
	network_1 = conv_2d(network, 64, 1)
	network_1 = batch_normalization(network_1)
  	network_1 = tflearn.activation(network_1, 'relu')

	network_2 = conv_2d(network, 48, 1)
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')
	network_2 = conv_2d(network_2, 64, 5)
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')

	network_3 = conv_2d(network, 64, 1)
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 96, 3)
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 96, 3)
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')

	network_4 = avg_pool_2d(network, kernel_size=3, strides=1)
	network_4 = conv_2d(network_4, 64, 1)
	network_4 = batch_normalization(network_4)
  	network_4 = tflearn.activation(network_4, 'relu')
	network = merge([network_1, network_2, network_3, network_4], axis=3, mode='concat')
	del network_1, network_2, network_3, network_4
	
	# Inception Long Module 1
	network_1 = conv_2d(network, 384, 3, strides=2)
	network_1 = batch_normalization(network_1)
	network_1 = tflearn.activation(network_1, 'relu')

	network_2 = conv_2d(network, 64, 1)
	network_2 = batch_normalization(network_2)
	network_2 = tflearn.activation(network_2, 'relu')
	network_2 = conv_2d(network_2, 96, 3)
	network_2 = batch_normalization(network_2)
	network_2 = tflearn.activation(network_2, 'relu')
	network_2 = conv_2d(network_2, 96, 3, strides=2)
	network_2 = batch_normalization(network_2)
	network_2 = tflearn.activation(network_2, 'relu')

	network_3 = max_pool_2d(network,3,strides=2)
	network = merge([network_1, network_2, network_3], axis=3, mode='concat')
	del network_1, network_2, network_3

	# Inception Wide Long Module 1
	network_1 = conv_2d(network, 192, 1)
	network_1 = batch_normalization(network_1)
  	network_1 = tflearn.activation(network_1, 'relu')

	network_2 = conv_2d(network, 128, 1)
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')
	network_2 = conv_2d(network_2, 128, [1,7])
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')
	network_2 = conv_2d(network_2, 192, [7,1])
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')

	network_3 = conv_2d(network, 128, 1)
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 128, [7,1])
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 128, [1,7])
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 128, [7,1])
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 192, [1,7])
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')

	network_4 = avg_pool_2d(network, kernel_size=3, strides=1)
	network_4 = conv_2d(network_4, 192, 1)
	network_4 = batch_normalization(network_4)
  	network_4 = tflearn.activation(network_4, 'relu')
	network = merge([network_1, network_2, network_3, network_4], axis=3, mode='concat')
	del network_1, network_2, network_3, network_4

	# Inception Wide Long Module 2
	network_1 = conv_2d(network, 192, 1)
	network_1 = batch_normalization(network_1)
  	network_1 = tflearn.activation(network_1, 'relu')

	network_2 = conv_2d(network, 160, 1)
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')
	network_2 = conv_2d(network_2, 160, [1,7])
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')
	network_2 = conv_2d(network_2, 192, [7,1])
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')

	network_3 = conv_2d(network, 160, 1)
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 160, [7,1])
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 160, [1,7])
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 160, [7,1])
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 192, [1,7])
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')

	network_4 = avg_pool_2d(network, kernel_size=3, strides=1)
	network_4 = conv_2d(network_4, 192, 1)
	network_4 = batch_normalization(network_4)
  	network_4 = tflearn.activation(network_4, 'relu')
	network = merge([network_1, network_2, network_3, network_4], axis=3, mode='concat')
	del network_1, network_2, network_3, network_4

	# Inception Wide Long Module 3
	network_1 = conv_2d(network, 192, 1)
	network_1 = batch_normalization(network_1)
  	network_1 = tflearn.activation(network_1, 'relu')

	network_2 = conv_2d(network, 160, 1)
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')
	network_2 = conv_2d(network_2, 160, [1,7])
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')
	network_2 = conv_2d(network_2, 192, [7,1])
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')

	network_3 = conv_2d(network, 160, 1)
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 160, [7,1])
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 160, [1,7])
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 160, [7,1])
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 192, [1,7])
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')

	network_4 = avg_pool_2d(network, kernel_size=3, strides=1)
	network_4 = conv_2d(network_4, 192, 1)
	network_4 = batch_normalization(network_4)
  	network_4 = tflearn.activation(network_4, 'relu')
	network = merge([network_1, network_2, network_3, network_4], axis=3, mode='concat')
	del network_1, network_2, network_3, network_4

	# Inception Wide Long Module 4
	network_1 = conv_2d(network, 192, 1)
	network_1 = batch_normalization(network_1)
  	network_1 = tflearn.activation(network_1, 'relu')

	network_2 = conv_2d(network, 192, 1)
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')
	network_2 = conv_2d(network_2, 192, [1,7])
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')
	network_2 = conv_2d(network_2, 192, [7,1])
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')

	network_3 = conv_2d(network, 192, 1)
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 192, [7,1])
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 192, [1,7])
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 192, [7,1])
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network_3, 192, [1,7])
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')

	network_4 = avg_pool_2d(network, kernel_size=3, strides=1)
	network_4 = conv_2d(network_4, 192, 1)
	network_4 = batch_normalization(network_4)
  	network_4 = tflearn.activation(network_4, 'relu')
	network = merge([network_1, network_2, network_3, network_4], axis=3, mode='concat')
	del network_1, network_2, network_3, network_4

	# Inception Long Module
	network_1 = conv_2d(network, 192, 1)
	network_1 = batch_normalization(network_1)
  	network_1 = tflearn.activation(network_1, 'relu')
	network_1 = conv_2d(network_1, 320, 3,strides=2)
	network_1 = batch_normalization(network_1)
  	network_1 = tflearn.activation(network_1, 'relu')

	network_2 = conv_2d(network, 192, 1)
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')
	network_2 = conv_2d(network_2, 192, [1,7])
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')
	network_2 = conv_2d(network_2, 192, [7,1])
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')
	network_2 = conv_2d(network_2, 192, 3, strides=2)
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')

	network_3 = max_pool_2d(network,3,strides=2)
	network = merge([network_1, network_2, network_3], axis=3, mode='concat')
	del network_1, network_2, network_3

	# Inception Branch Module 1 
	network_1 = conv_2d(network, 320, 1)
	network_1 = batch_normalization(network_1)
  	network_1 = tflearn.activation(network_1, 'relu')

	network_2 = conv_2d(network, 384, 1)
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')
	network_2_1 = conv_2d(network_2, 384, [1,3])
	network_2_1 = batch_normalization(network_2_1)
  	network_2_1 = tflearn.activation(network_2_1, 'relu')
	network_2_2 = conv_2d(network_2, 384, [3,1])
	network_2_2 = batch_normalization(network_2_2)
  	network_2_2 = tflearn.activation(network_2_2, 'relu')

	network_3 = conv_2d(network, 448, 1)
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network, 384, 3)
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3_1 = conv_2d(network_3, 384, [1,3])
	network_3_1 = batch_normalization(network_3_1)
  	network_3_1 = tflearn.activation(network_3_1, 'relu')
	network_3_2 = conv_2d(network_3, 384, [3,1])
	network_3_2 = batch_normalization(network_3_2)
  	network_3_2 = tflearn.activation(network_3_2, 'relu')

	network_4 = avg_pool_2d(network, kernel_size=3, strides=1)
	network_4 = conv_2d(network_4, 192, 1)
	network_4 = batch_normalization(network_4)
  	network_4 = tflearn.activation(network_4, 'relu')
	network = merge([network_1, network_2_1, network_2_2, network_3_1, network_3_2, network_4], axis=3, mode='concat')
	del network_1, network_2, network_3, network_4
	
	# Inception Branch Module 2
	network_1 = conv_2d(network, 320, 1)
	network_1 = batch_normalization(network_1)
  	network_1 = tflearn.activation(network_1, 'relu')

	network_2 = conv_2d(network, 384, 1)
	network_2 = batch_normalization(network_2)
  	network_2 = tflearn.activation(network_2, 'relu')
	network_2_1 = conv_2d(network_2, 384, [1,3])
	network_2_1 = batch_normalization(network_2_1)
  	network_2_1 = tflearn.activation(network_2_1, 'relu')
	network_2_2 = conv_2d(network_2, 384, [3,1])
	network_2_2 = batch_normalization(network_2_2)
  	network_2_2 = tflearn.activation(network_2_2, 'relu')

	network_3 = conv_2d(network, 448, 1)
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3 = conv_2d(network, 384, 3)
	network_3 = batch_normalization(network_3)
  	network_3 = tflearn.activation(network_3, 'relu')
	network_3_1 = conv_2d(network_3, 384, [1,3])
	network_3_1 = batch_normalization(network_3_1)
  	network_3_1 = tflearn.activation(network_3_1, 'relu')
	network_3_2 = conv_2d(network_3, 384, [3,1])
	network_3_2 = batch_normalization(network_3_2)
  	network_3_2 = tflearn.activation(network_3_2, 'relu')

	network_4 = avg_pool_2d(network, kernel_size=3, strides=1)
	network_4 = conv_2d(network_4, 192, 1)
	network_4 = batch_normalization(network_4)
  	network_4 = tflearn.activation(network_4, 'relu')
	network = merge([network_1, network_2_1, network_2_2, network_3_1, network_3_2, network_4], axis=3, mode='concat')
	del network_1, network_2, network_3, network_4

	# Average Pool 
	network = avg_pool_2d(network, kernel_size=8, strides=1)

	# Flatten 
	network = flatten(network)

	# Fully Connected Layer
	network = fully_connected(network, 2048, activation='elu')

	# Loss Layer 
	loss_layer = fully_connected(network, num_output_classes, regularizer=cnn_regularization_type, weight_decay=cnn_regularization_weight_decay, activation=cnn_loss_layer_activation)

	# Return net
	return loss_layer


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Function For VGG 11 
def load_vgg_11(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation):
	input_layer = input_data(shape=[None, cnn_image_shape[0], cnn_image_shape[1], cnn_image_shape[2]], data_preprocessing = cnn_img_prep, data_augmentation = cnn_img_aug)
	network = conv_2d(input_layer, 64, 3, activation='relu')
	network = max_pool_2d(network, 2, strides=2)

	network = conv_2d(network, 128, 3, activation='relu')
	network = max_pool_2d(network, 2, strides=2)

	network = conv_2d(network, 256, 3, activation='relu')
	network = conv_2d(network, 256, 3, activation='relu')
	network = max_pool_2d(network, 2, strides=2)

	network = conv_2d(network, 512, 3, activation='relu')
	network = conv_2d(network, 512, 3, activation='relu')
	network = max_pool_2d(network, 2, strides=2)

	network = conv_2d(network, 512, 3, activation='relu')
	network = conv_2d(network, 512, 3, activation='relu')
	network = max_pool_2d(network, 2, strides=2)

	network = fully_connected(network, 4096, activation='relu')
	network = dropout(network, cnn_keep_probability)
	network = fully_connected(network, 4096, activation='relu')
	network = dropout(network, cnn_keep_probability)

	loss_layer = fully_connected(network, num_output_classes, regularizer=cnn_regularization_type, weight_decay=cnn_regularization_weight_decay, activation=cnn_loss_layer_activation)

	# Return net
	return loss_layer

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Function For VGG 16
def load_vgg_16(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation):
	input_layer = input_data(shape=[None, cnn_image_shape[0], cnn_image_shape[1], cnn_image_shape[2]], data_preprocessing = cnn_img_prep, data_augmentation = cnn_img_aug)
	network = conv_2d(input_layer, 64, 3, activation='relu')
	network = conv_2d(network, 64, 3, activation='relu')
	network = max_pool_2d(network, 2, strides=2)

	network = conv_2d(network, 128, 3, activation='relu')
	network = conv_2d(network, 128, 3, activation='relu')
	network = max_pool_2d(network, 2, strides=2)

	network = conv_2d(network, 256, 3, activation='relu')
	network = conv_2d(network, 256, 3, activation='relu')
	network = conv_2d(network, 256, 3, activation='relu')
	network = max_pool_2d(network, 2, strides=2)

	network = conv_2d(network, 512, 3, activation='relu')
	network = conv_2d(network, 512, 3, activation='relu')
	network = conv_2d(network, 512, 3, activation='relu')
	network = max_pool_2d(network, 2, strides=2)

	network = conv_2d(network, 512, 3, activation='relu')
	network = conv_2d(network, 512, 3, activation='relu')
	network = conv_2d(network, 512, 3, activation='relu')
	network = max_pool_2d(network, 2, strides=2)

	network = fully_connected(network, 4096, activation='relu')
	network = dropout(network, cnn_keep_probability)
	network = fully_connected(network, 4096, activation='relu')
	network = dropout(network, cnn_keep_probability)

	loss_layer = fully_connected(network, num_output_classes, regularizer=cnn_regularization_type, weight_decay=cnn_regularization_weight_decay, activation=cnn_loss_layer_activation)

	# Return net
	return loss_layer


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Function For VGG 19
def load_vgg_19(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation):
	input_layer = input_data(shape=[None, cnn_image_shape[0], cnn_image_shape[1], cnn_image_shape[2]], data_preprocessing = cnn_img_prep, data_augmentation = cnn_img_aug)
	network = conv_2d(input_layer, 64, 3, activation='relu')
	network = conv_2d(network, 64, 3, activation='relu')
	network = max_pool_2d(network, 2, strides=2)

	network = conv_2d(network, 128, 3, activation='relu')
	network = conv_2d(network, 128, 3, activation='relu')
	network = max_pool_2d(network, 2, strides=2)

	network = conv_2d(network, 256, 3, activation='relu')
	network = conv_2d(network, 256, 3, activation='relu')
	network = conv_2d(network, 256, 3, activation='relu')
	network = conv_2d(network, 256, 3, activation='relu')
	network = max_pool_2d(network, 2, strides=2)

	network = conv_2d(network, 512, 3, activation='relu')
	network = conv_2d(network, 512, 3, activation='relu')
	network = conv_2d(network, 512, 3, activation='relu')
	network = conv_2d(network, 512, 3, activation='relu')
	network = max_pool_2d(network, 2, strides=2)

	network = conv_2d(network, 512, 3, activation='relu')
	network = conv_2d(network, 512, 3, activation='relu')
	network = conv_2d(network, 512, 3, activation='relu')
	network = conv_2d(network, 512, 3, activation='relu')
	network = max_pool_2d(network, 2, strides=2)

	network = fully_connected(network, 4096, activation='relu')
	network = dropout(network, cnn_keep_probability)
	network = fully_connected(network, 4096, activation='relu')
	network = dropout(network, cnn_keep_probability)

	loss_layer = fully_connected(network, num_output_classes, regularizer=cnn_regularization_type, weight_decay=cnn_regularization_weight_decay, activation=cnn_loss_layer_activation)

	# Return net
	return loss_layer


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Function For ResNet 
def load_resnet(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation, n):
	input_layer = input_data(shape=[None, cnn_image_shape[0], cnn_image_shape[1], cnn_image_shape[2]], data_preprocessing = cnn_img_prep, data_augmentation = cnn_img_aug)
	network = tflearn.conv_2d(input_layer, 16, 3, regularizer=cnn_regularization_type, weight_decay=cnn_regularization_weight_decay)
	network = tflearn.residual_block(network, n, 16)
	network = tflearn.residual_block(network, 1, 32, downsample=True)
	network = tflearn.residual_block(network, n-1, 32)
	network = tflearn.residual_block(network, 1, 64, downsample=True)
	network = tflearn.residual_block(network, n-1, 64)
	network = tflearn.batch_normalization(network)
	network = tflearn.activation(network, 'relu')
	network = tflearn.global_avg_pool(network)

	# Loss Layer 
	loss_layer = fully_connected(network, num_output_classes, regularizer=cnn_regularization_type, weight_decay=cnn_regularization_weight_decay, activation=cnn_loss_layer_activation)

	# Return net
	return loss_layer

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Function For Wide ResNet 
def wide_residual_block(incoming, nb_blocks, out_channels, downsample=False,
                        downsample_strides=2, activ='relu', batch_norm=True,
                        apply_dropout=False, keep_prob=0.5,
                        bias=True, weights_init='variance_scaling',
                        bias_init='zeros', regularizer='L2', weight_decay=0.0001,
                        trainable=True, restore=True, reuse=False, scope=None,
                        name="WideResidualBlock"):
	resnet = incoming
 	in_channels = incoming.get_shape().as_list()[-1]

  	with tf.variable_scope(scope, name, values=[incoming], reuse=reuse) as scope:
    		name = scope.name 

	    	for i in range(nb_blocks):
	      		identity = resnet

		      	if not downsample:
			  	downsample_strides = 1

		      	if batch_norm:
				resnet = batch_normalization(resnet)
		      	resnet = tflearn.activation(resnet, activ)

		      	resnet = tflearn.conv_2d(resnet, out_channels, 3,
				       downsample_strides, 'same', 'linear',
				       bias, weights_init, bias_init,
				       regularizer, weight_decay, trainable,
				       restore)
				       
				       
		      	if apply_dropout:
				resnet = dropout(resnet, keep_prob)

		      	if batch_norm:
				resnet = batch_normalization(resnet)
		      	resnet = tflearn.activation(resnet, activ)

		      	resnet = tflearn.conv_2d(resnet, out_channels, 3, 1, 'same',
				       'linear', bias, weights_init,
				       bias_init, regularizer, weight_decay,
				       trainable, restore)

		      	# Downsampling
		      	if downsample_strides > 1:
				identity = tflearn.avg_pool_2d(identity, 1, downsample_strides)

		      	# Projection to new dimension
		      	if in_channels != out_channels:
				ch = (out_channels - in_channels)//2
		       		identity = tf.pad(identity,
				          [[0, 0], [0, 0], [0, 0], [ch, ch]])
				in_channels = out_channels

      		resnet = resnet + identity
  	return resnet

# ---------- Main Function ---------------
def load_wide_resnet(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation, n, k):
	input_layer = input_data(shape=[None, cnn_image_shape[0], cnn_image_shape[1], cnn_image_shape[2]], data_preprocessing = cnn_img_prep, data_augmentation = cnn_img_aug)  
  	network = tflearn.conv_2d(input_layer, 16, 3, regularizer=cnn_regularization_type, weight_decay=cnn_regularization_weight_decay)
  	network = wide_residual_block(network, n, 16 * k, apply_dropout=dropout, keep_prob=cnn_keep_probability)
  	network = wide_residual_block(network, 1, 32 * k, downsample=True, apply_dropout=dropout, keep_prob=cnn_keep_probability)
  	network = wide_residual_block(network, n - 1, 32 * k, apply_dropout=dropout, keep_prob=cnn_keep_probability)
  	network = wide_residual_block(network, 1, 64 * k, downsample=True, apply_dropout=dropout, keep_prob=cnn_keep_probability)
  	network = wide_residual_block(network, n - 1, 64 * k, apply_dropout=dropout, keep_prob=cnn_keep_probability)
  	network = batch_normalization(network)
  	network = tflearn.activation(network, 'relu')
  	network = tflearn.global_avg_pool(network)

	# Loss Layer 
	loss_layer = fully_connected(network, num_output_classes, regularizer=cnn_regularization_type, weight_decay=cnn_regularization_weight_decay, activation=cnn_loss_layer_activation)

	# Return net
	return loss_layer

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Function For Dense Net  
def weight_variable_densenet(shape):
	initial = tf.truncated_normal(shape, stddev=0.01)
  	return tf.Variable(initial)

def bias_variable_densenet(shape):
  	initial = tf.constant(0.01, shape=shape)
  	return tf.Variable(initial)

def conv2d_densenet(input, in_features, out_features, kernel_size, with_bias=False):
  	W = weight_variable_densenet([ kernel_size, kernel_size, in_features, out_features ])
  	conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
  	if with_bias:
    		return conv + bias_variable_densenet([ out_features ])
  	return conv

def batch_activ_conv_densenet(network, in_features, out_features, kernel_size, is_training, keep_prob):
  	network = tf.contrib.layers.batch_norm(network, scale=True, is_training=is_training, updates_collections=None)
  	network = tf.nn.relu(network)
  	network = conv2d_densenet(network, in_features, out_features, kernel_size)
  	network = tf.nn.dropout(network, keep_prob)
  	return network

def block_densenet(input, layers, in_features, growth, is_training, keep_prob):
  	network = input
  	features = in_features
  	for idx in xrange(layers):
    		tmp = batch_activ_conv_densenet(network, features, growth, 3, is_training, keep_prob)
    		network = tf.concat(3, (network, tmp))
    		features += growth
  	return network, features

def avg_pool_densenet(input, s):
  	return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')

# ---------- Main Function -------------
def load_densenet(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation, depth, growth_ratio):

	# Variables 
	layers = int((depth - 4) / 3)
	is_training = True

	input_layer = input_data(shape=[None, cnn_image_shape[0], cnn_image_shape[1], cnn_image_shape[2]], data_preprocessing = cnn_img_prep, data_augmentation = cnn_img_aug)
  	network = conv2d_densenet(input_layer, 3, 16, 3)
  	network, features = block_densenet(network, layers, 16, growth_ratio, is_training, cnn_keep_probability)
  	network = batch_activ_conv_densenet(network, features, features, 1, is_training, cnn_keep_probability)
  	network = avg_pool_densenet(network, 2)
  	network, features = block_densenet(network, layers, features, growth_ratio, is_training, cnn_keep_probability)
  	network = batch_activ_conv_densenet(network, features, features, 1, is_training, cnn_keep_probability)
  	network = avg_pool_densenet(network, 2)
  	network, features = block_densenet(network, layers, features, growth_ratio, is_training, cnn_keep_probability)

  	network = tf.contrib.layers.batch_norm(network, scale=True, is_training=is_training, updates_collections=None)
 	network = tf.nn.relu(network)
  	network = avg_pool_densenet(network, 8)

	# Loss Layer 
	loss_layer = fully_connected(network, num_output_classes, regularizer=cnn_regularization_type, weight_decay=cnn_regularization_weight_decay, activation=cnn_loss_layer_activation)

	# Return net
	return loss_layer

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Function for Convolutional Highway Networks
def load_highway(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation, block_depth): 

	network = input_data(shape=[None, cnn_image_shape[0], cnn_image_shape[1], cnn_image_shape[2]], data_preprocessing = cnn_img_prep, data_augmentation = cnn_img_aug)
	# Highway convolutions with pooling and dropout
	for i in range(block_depth):
		for j in [3, 2, 1]: 
			network = highway_conv_2d(network, 16, j, activation='elu')
	    	network = max_pool_2d(network, 2)
	    	network = batch_normalization(network)
	    
	network = fully_connected(network, 128, activation='elu')
	network = fully_connected(network, 256, activation='elu')

	# Loss Layer 
	loss_layer = fully_connected(network, num_output_classes, regularizer=cnn_regularization_type, weight_decay=cnn_regularization_weight_decay, activation=cnn_loss_layer_activation)

	# Return net
	return loss_layer

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Functions for Inception Resnet V4
def repeat(inputs, repetitions, layer, *args, **kwargs):
    outputs = inputs
    for i in range(repetitions):
        outputs = layer(outputs, *args, **kwargs)
    return outputs

def block35(net, scale=1.0, activation="relu"):
    tower_conv = relu(batch_normalization(conv_2d(net, 32, 1, bias=False, activation='relu', name='Conv2d_1x1')))
    tower_conv1_0 = relu(batch_normalization(conv_2d(net, 32, 1, bias=False, activation='relu',name='Conv2d_0a_1x1')))
    tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1_0, 32, 3, bias=False, activation='relu',name='Conv2d_0b_3x3')))
    tower_conv2_0 = relu(batch_normalization(conv_2d(net, 32, 1, bias=False, activation='relu', name='Conv2d_0a_1x1')))
    tower_conv2_1 = relu(batch_normalization(conv_2d(tower_conv2_0, 48,3, bias=False, activation='relu', name='Conv2d_0b_3x3')))
    tower_conv2_2 = relu(batch_normalization(conv_2d(tower_conv2_1, 64,3, bias=False, activation='relu', name='Conv2d_0c_3x3')))
    tower_mixed = merge([tower_conv, tower_conv1_1, tower_conv2_2], mode='concat', axis=3)
    tower_out = relu(batch_normalization(conv_2d(tower_mixed, net.get_shape()[3], 1, bias=False, activation='relu', name='Conv2d_1x1')))
    net += scale * tower_out
    if activation:
        if isinstance(activation, str):
            net = activations.get(activation)(net)
        elif hasattr(activation, '__call__'):
            net = activation(net)
        else:
            raise ValueError("Invalid Activation.")
    return net

def block17(net, scale=1.0, activation="relu"):
    tower_conv = relu(batch_normalization(conv_2d(net, 192, 1, bias=False, activation='relu', name='Conv2d_1x1')))
    tower_conv_1_0 = relu(batch_normalization(conv_2d(net, 128, 1, bias=False, activation='relu', name='Conv2d_0a_1x1')))
    tower_conv_1_1 = relu(batch_normalization(conv_2d(tower_conv_1_0, 160,[1,7], bias=False, activation='relu',name='Conv2d_0b_1x7')))
    tower_conv_1_2 = relu(batch_normalization(conv_2d(tower_conv_1_1, 192, [7,1], bias=False, activation='relu',name='Conv2d_0c_7x1')))
    tower_mixed = merge([tower_conv,tower_conv_1_2], mode='concat', axis=3)
    tower_out = relu(batch_normalization(conv_2d(tower_mixed, net.get_shape()[3], 1, bias=False, activation='relu', name='Conv2d_1x1')))
    net += scale * tower_out
    if activation:
        if isinstance(activation, str):
            net = activations.get(activation)(net)
        elif hasattr(activation, '__call__'):
            net = activation(net)
        else:
            raise ValueError("Invalid Activation.")
    return net


def block8(net, scale=1.0, activation="relu"):
    tower_conv = relu(batch_normalization(conv_2d(net, 192, 1, bias=False, activation='relu', name='Conv2d_1x1')))
    tower_conv1_0 = relu(batch_normalization(conv_2d(net, 192, 1, bias=False, activation='relu', name='Conv2d_0a_1x1')))
    tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1_0, 224, [1,3], bias=False, activation='relu', name='Conv2d_0b_1x3')))
    tower_conv1_2 = relu(batch_normalization(conv_2d(tower_conv1_1, 256, [3,1], bias=False, name='Conv2d_0c_3x1')))
    tower_mixed = merge([tower_conv,tower_conv1_2], mode='concat', axis=3)
    tower_out = relu(batch_normalization(conv_2d(tower_mixed, net.get_shape()[3], 1, bias=False, activation='relu', name='Conv2d_1x1')))
    net += scale * tower_out
    if activation:
        if isinstance(activation, str):
            net = activations.get(activation)(net)
        elif hasattr(activation, '__call__'):
            net = activation(net)
        else:
            raise ValueError("Invalid Activation.")
    return net 

# ---------- Main Function -------------
def load_inception_resnet_v4(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation): 

	network = input_data(shape=[None, cnn_image_shape[0], cnn_image_shape[1], cnn_image_shape[2]], data_preprocessing = cnn_img_prep, data_augmentation = cnn_img_aug)
	conv1a_3_3 = relu(batch_normalization(conv_2d(network, 32, 3, strides=2, bias=False, padding='VALID',activation='relu',name='Conv2d_1a_3x3')))
	conv2a_3_3 = relu(batch_normalization(conv_2d(conv1a_3_3, 32, 3, bias=False, padding='VALID',activation='relu', name='Conv2d_2a_3x3')))
	conv2b_3_3 = relu(batch_normalization(conv_2d(conv2a_3_3, 64, 3, bias=False, activation='relu', name='Conv2d_2b_3x3')))
	maxpool3a_3_3 = max_pool_2d(conv2b_3_3, 3, strides=2, padding='VALID', name='MaxPool_3a_3x3')
	conv3b_1_1 = relu(batch_normalization(conv_2d(maxpool3a_3_3, 80, 1, bias=False, padding='VALID',activation='relu', name='Conv2d_3b_1x1')))
	conv4a_3_3 = relu(batch_normalization(conv_2d(conv3b_1_1, 192, 3, bias=False, padding='VALID',activation='relu', name='Conv2d_4a_3x3')))
	maxpool5a_3_3 = max_pool_2d(conv4a_3_3, 3, strides=2, padding='VALID', name='MaxPool_5a_3x3')

	tower_conv = relu(batch_normalization(conv_2d(maxpool5a_3_3, 96, 1, bias=False, activation='relu', name='Conv2d_5b_b0_1x1')))
	tower_conv1_0 = relu(batch_normalization(conv_2d(maxpool5a_3_3, 48, 1, bias=False, activation='relu', name='Conv2d_5b_b1_0a_1x1')))
	tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1_0, 64, 5, bias=False, activation='relu', name='Conv2d_5b_b1_0b_5x5')))
	tower_conv2_0 = relu(batch_normalization(conv_2d(maxpool5a_3_3, 64, 1, bias=False, activation='relu', name='Conv2d_5b_b2_0a_1x1')))
	tower_conv2_1 = relu(batch_normalization(conv_2d(tower_conv2_0, 96, 3, bias=False, activation='relu', name='Conv2d_5b_b2_0b_3x3')))
	tower_conv2_2 = relu(batch_normalization(conv_2d(tower_conv2_1, 96, 3, bias=False, activation='relu',name='Conv2d_5b_b2_0c_3x3')))
	tower_pool3_0 = avg_pool_2d(maxpool5a_3_3, 3, strides=1, padding='same', name='AvgPool_5b_b3_0a_3x3')
	tower_conv3_1 = relu(batch_normalization(conv_2d(tower_pool3_0, 64, 1, bias=False, activation='relu',name='Conv2d_5b_b3_0b_1x1')))
	tower_5b_out = merge([tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_1], mode='concat', axis=3)
	network = repeat(tower_5b_out, 10, block35, scale=0.17)

	tower_conv = relu(batch_normalization(conv_2d(network, 384, 3, bias=False, strides=2,activation='relu', padding='VALID', name='Conv2d_6a_b0_0a_3x3')))
	tower_conv1_0 = relu(batch_normalization(conv_2d(network, 256, 1, bias=False, activation='relu', name='Conv2d_6a_b1_0a_1x1')))
	tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1_0, 256, 3, bias=False, activation='relu', name='Conv2d_6a_b1_0b_3x3')))
	tower_conv1_2 = relu(batch_normalization(conv_2d(tower_conv1_1, 384, 3, bias=False, strides=2, padding='VALID', activation='relu',name='Conv2d_6a_b1_0c_3x3')))
	tower_pool = max_pool_2d(network, 3, strides=2, padding='VALID',name='MaxPool_1a_3x3')
	network = merge([tower_conv, tower_conv1_2, tower_pool], mode='concat', axis=3)
	network = repeat(network, 20, block17, scale=0.1)

	tower_conv = relu(batch_normalization(conv_2d(network, 256, 1, bias=False, activation='relu', name='Conv2d_0a_1x1')))
	tower_conv0_1 = relu(batch_normalization(conv_2d(tower_conv, 384, 3, bias=False, strides=2, padding='VALID', activation='relu',name='Conv2d_0a_1x1')))

	tower_conv1 = relu(batch_normalization(conv_2d(network, 256, 1, bias=False, padding='VALID', activation='relu',name='Conv2d_0a_1x1')))
	tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1,288,3, bias=False, strides=2, padding='VALID',activation='relu', name='COnv2d_1a_3x3')))

	tower_conv2 = relu(batch_normalization(conv_2d(network, 256,1, bias=False, activation='relu',name='Conv2d_0a_1x1')))
	tower_conv2_1 = relu(batch_normalization(conv_2d(tower_conv2, 288,3, bias=False, name='Conv2d_0b_3x3',activation='relu')))
	tower_conv2_2 = relu(batch_normalization(conv_2d(tower_conv2_1, 320, 3, bias=False, strides=2, padding='VALID',activation='relu', name='Conv2d_1a_3x3')))

	tower_pool = max_pool_2d(network, 3, strides=2, padding='VALID', name='MaxPool_1a_3x3')
	network = merge([tower_conv0_1, tower_conv1_1,tower_conv2_2, tower_pool], mode='concat', axis=3)

	network = repeat(network, 9, block8, scale=0.2)
	network = block8(network, activation='relu')

	network = relu(batch_normalization(conv_2d(network, 1536, 1, bias=False, activation='relu', name='Conv2d_7b_1x1')))
	network = avg_pool_2d(network, network.get_shape().as_list()[1:3],strides=2, padding='VALID', name='AvgPool_1a_8x8')
	network = flatten(network)
	network = dropout(network, cnn_keep_probability)
	

	# Loss Layer 
	loss_layer = fully_connected(network, num_output_classes, regularizer=cnn_regularization_type, weight_decay=cnn_regularization_weight_decay, activation=cnn_loss_layer_activation)

	# Return net
	return loss_layer






