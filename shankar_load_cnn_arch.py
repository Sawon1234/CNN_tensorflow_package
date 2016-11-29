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
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d, highway_conv_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import *
from tflearn.optimizers import Momentum
from tflearn.layers.normalization import batch_normalization

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
def load_googlenet_v3(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation):
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



def conv_2d_bn(incoming,nb_filter, filter_size, strides=1, padding='same', activation='linear', bias=True, weights_init='uniform_scaling', bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True, restore=True, reuse=False, scope=None, name='Conv2D_BN'):
	conv_bn = conv_2d(incoming, nb_filter, filter_size, strides=strides, name=name)
	conv_bn = batch_normalization(conv_bn)
	conv_bn = activation(bnlayer,'relu')
	return conv_bn

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# GoogleLeNet wih Batch Normalization
def load_googlenet_bn(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation):
	# Input Layer 
	input_layer = input_data(shape=[None, cnn_image_shape[0], cnn_image_shape[1], cnn_image_shape[2]], data_preprocessing = cnn_img_prep, data_augmentation = cnn_img_aug)

	# Basic Conv Layers
	conv1_7_7 = conv_2d_bn(input_layer, 64, 7, strides=2, activation='relu', n,'relu'ame = 'conv1_7_7_s2')
	pool1_3_3 = max_pool_2d(conv1_7_7, 3,strides=2)
	# pool1_3_3 = local_response_normalization(pool1_3_3)
	conv2_3_3_reduce = conv_2d_bn(pool1_3_3, 64,1, activation='relu',na,'relu'me = 'conv2_3_3_reduce')
	conv2_3_3 = conv_2d_bn(conv2_3_3_reduce, 192,3, activation='relu', n,'relu'ame='conv2_3_3')
	conv2_3_3 = local_response_normalization(conv2_3_3)
	pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')

	# Inception Module 3a
	inception_3a_1_1 = conv_2d_bn(pool2_3_3, 64, 1, activation='relu', n,'relu'ame='inception_3a_1_1')
	inception_3a_3_3_reduce = conv_2d_bn(pool2_3_3, 96,1, activation='relu', n,'relu'ame='inception_3a_3_3_reduce')
	inception_3a_3_3 = conv_2d_bn(inception_3a_3_3_reduce, 128,filter_size=3,  activation='relu', n,'relu'ame = 'inception_3a_3_3')
	inception_3a_5_5_reduce = conv_2d_bn(pool2_3_3,16, filter_size=1,activation='relu', n,'relu'ame ='inception_3a_5_5_reduce' )
	inception_3a_5_5 = conv_2d_bn(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', n,'relu'ame= 'inception_3a_5_5')
	inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
	inception_3a_pool_1_1 = conv_2d_bn(inception_3a_pool, 32, filter_size=1, activation='relu', n,'relu'ame='inception_3a_pool_1_1')
	inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

	# Inception Module 3b
	inception_3b_1_1 = conv_2d_bn(inception_3a_output, 128,filter_size=1,activation='relu', n,'relu'ame= 'inception_3b_1_1' )
	inception_3b_3_3_reduce = conv_2d_bn(inception_3a_output, 128, filter_size=1, activation='relu', n,'relu'ame='inception_3b_3_3_reduce')
	inception_3b_3_3 = conv_2d_bn(inception_3b_3_3_reduce, 192, filter_size=3,  activation='relu',na,'relu'me='inception_3b_3_3')
	inception_3b_5_5_reduce = conv_2d_bn(inception_3a_output, 32, filter_size=1, activation='relu', n,'relu'ame = 'inception_3b_5_5_reduce')
	inception_3b_5_5 = conv_2d_bn(inception_3b_5_5_reduce, 96, filter_size=5,  name = 'inception_3b_5_5')
	inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
	inception_3b_pool_1_1 = conv_2d_bn(inception_3b_pool, 64, filter_size=1,activation='relu', n,'relu'ame='inception_3b_pool_1_1')
	inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

	# Inception Module 4a
	pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
	inception_4a_1_1 = conv_2d_bn(pool3_3_3, 192, filter_size=1, activation='relu', n,'relu'ame='inception_4a_1_1')
	inception_4a_3_3_reduce = conv_2d_bn(pool3_3_3, 96, filter_size=1, activation='relu', n,'relu'ame='inception_4a_3_3_reduce')
	inception_4a_3_3 = conv_2d_bn(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', n,'relu'ame='inception_4a_3_3')
	inception_4a_5_5_reduce = conv_2d_bn(pool3_3_3, 16, filter_size=1, activation='relu', n,'relu'ame='inception_4a_5_5_reduce')
	inception_4a_5_5 = conv_2d_bn(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', n,'relu'ame='inception_4a_5_5')
	inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
	inception_4a_pool_1_1 = conv_2d_bn(inception_4a_pool, 64, filter_size=1, activation='relu', n,'relu'ame='inception_4a_pool_1_1')
	inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')

	# Inception Module 4b
	inception_4b_1_1 = conv_2d_bn(inception_4a_output, 160, filter_size=1, activation='relu', n,'relu'ame='inception_4a_1_1')
	inception_4b_3_3_reduce = conv_2d_bn(inception_4a_output, 112, filter_size=1, activation='relu', n,'relu'ame='inception_4b_3_3_reduce')
	inception_4b_3_3 = conv_2d_bn(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', n,'relu'ame='inception_4b_3_3')
	inception_4b_5_5_reduce = conv_2d_bn(inception_4a_output, 24, filter_size=1, activation='relu', n,'relu'ame='inception_4b_5_5_reduce')
	inception_4b_5_5 = conv_2d_bn(inception_4b_5_5_reduce, 64, filter_size=5,  activation='relu', n,'relu'ame='inception_4b_5_5')
	inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1,  name='inception_4b_pool')
	inception_4b_pool_1_1 = conv_2d_bn(inception_4b_pool, 64, filter_size=1, activation='relu', n,'relu'ame='inception_4b_pool_1_1')
	inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')

	# Inception Module 4c
	inception_4c_1_1 = conv_2d_bn(inception_4b_output, 128, filter_size=1, activation='relu',na,'relu'me='inception_4c_1_1')
	inception_4c_3_3_reduce = conv_2d_bn(inception_4b_output, 128, filter_size=1, activation='relu', n,'relu'ame='inception_4c_3_3_reduce')
	inception_4c_3_3 = conv_2d_bn(inception_4c_3_3_reduce, 256,  filter_size=3, activation='relu', n,'relu'ame='inception_4c_3_3')
	inception_4c_5_5_reduce = conv_2d_bn(inception_4b_output, 24, filter_size=1, activation='relu', n,'relu'ame='inception_4c_5_5_reduce')
	inception_4c_5_5 = conv_2d_bn(inception_4c_5_5_reduce, 64,  filter_size=5, activation='relu', n,'relu'ame='inception_4c_5_5')
	inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
	inception_4c_pool_1_1 = conv_2d_bn(inception_4c_pool, 64, filter_size=1, activation='relu', n,'relu'ame='inception_4c_pool_1_1')
	inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3,name='inception_4c_output')

	# Inception Module 4d
	inception_4d_1_1 = conv_2d_bn(inception_4c_output, 112, filter_size=1, activation='relu', n,'relu'ame='inception_4d_1_1')
	inception_4d_3_3_reduce = conv_2d_bn(inception_4c_output, 144, filter_size=1, activation='relu', n,'relu'ame='inception_4d_3_3_reduce')
	inception_4d_3_3 = conv_2d_bn(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', n,'relu'ame='inception_4d_3_3')
	inception_4d_5_5_reduce = conv_2d_bn(inception_4c_output, 32, filter_size=1, activation='relu', n,'relu'ame='inception_4d_5_5_reduce')
	inception_4d_5_5 = conv_2d_bn(inception_4d_5_5_reduce, 64, filter_size=5,  activation='relu', n,'relu'ame='inception_4d_5_5')
	inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1,  name='inception_4d_pool')
	inception_4d_pool_1_1 = conv_2d_bn(inception_4d_pool, 64, filter_size=1, activation='relu', n,'relu'ame='inception_4d_pool_1_1')
	inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], mode='concat', axis=3, name='inception_4d_output')

	# Inception Module 4e
	inception_4e_1_1 = conv_2d_bn(inception_4d_output, 256, filter_size=1, activation='relu', n,'relu'ame='inception_4e_1_1')
	inception_4e_3_3_reduce = conv_2d_bn(inception_4d_output, 160, filter_size=1, activation='relu', n,'relu'ame='inception_4e_3_3_reduce')
	inception_4e_3_3 = conv_2d_bn(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', n,'relu'ame='inception_4e_3_3')
	inception_4e_5_5_reduce = conv_2d_bn(inception_4d_output, 32, filter_size=1, activation='relu', n,'relu'ame='inception_4e_5_5_reduce')
	inception_4e_5_5 = conv_2d_bn(inception_4e_5_5_reduce, 128,  filter_size=5, activation='relu', n,'relu'ame='inception_4e_5_5')
	inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1,  name='inception_4e_pool')
	inception_4e_pool_1_1 = conv_2d_bn(inception_4e_pool, 128, filter_size=1, activation='relu', n,'relu'ame='inception_4e_pool_1_1')
	inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5,inception_4e_pool_1_1],axis=3, mode='concat')

	# Max Pool 
	pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')

	# Inception Module 5a
	inception_5a_1_1 = conv_2d_bn(pool4_3_3, 256, filter_size=1, activation='relu', n,'relu'ame='inception_5a_1_1')
	inception_5a_3_3_reduce = conv_2d_bn(pool4_3_3, 160, filter_size=1, activation='relu', n,'relu'ame='inception_5a_3_3_reduce')
	inception_5a_3_3 = conv_2d_bn(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', n,'relu'ame='inception_5a_3_3')
	inception_5a_5_5_reduce = conv_2d_bn(pool4_3_3, 32, filter_size=1, activation='relu', n,'relu'ame='inception_5a_5_5_reduce')
	inception_5a_5_5 = conv_2d_bn(inception_5a_5_5_reduce, 128, filter_size=5,  activation='relu', n,'relu'ame='inception_5a_5_5')
	inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1,  name='inception_5a_pool')
	inception_5a_pool_1_1 = conv_2d_bn(inception_5a_pool, 128, filter_size=1,activation='relu', n,'relu'ame='inception_5a_pool_1_1')
	inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3,mode='concat')

	# Inception Module 5b
	inception_5b_1_1 = conv_2d_bn(inception_5a_output, 384, filter_size=1,activation='relu', n,'relu'ame='inception_5b_1_1')
	inception_5b_3_3_reduce = conv_2d_bn(inception_5a_output, 192, filter_size=1, activation='relu', n,'relu'ame='inception_5b_3_3_reduce')
	inception_5b_3_3 = conv_2d_bn(inception_5b_3_3_reduce, 384,  filter_size=3,activation='relu', n,'relu'ame='inception_5b_3_3')
	inception_5b_5_5_reduce = conv_2d_bn(inception_5a_output, 48, filter_size=1, activation='relu', n,'relu'ame='inception_5b_5_5_reduce')
	inception_5b_5_5 = conv_2d_bn(inception_5b_5_5_reduce,128, filter_size=5,  activation='relu', n,'relu'ame='inception_5b_5_5' )
	inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1,  name='inception_5b_pool')
	inception_5b_pool_1_1 = conv_2d_bn(inception_5b_pool, 128, filter_size=1, activation='relu', n,'relu'ame='inception_5b_pool_1_1')
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
# Function for Highway Convolutional Networks
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







