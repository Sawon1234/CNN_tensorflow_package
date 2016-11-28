# ------------------------------------------------------------------------
# Module of functions for loading data to train and test with TFLearn
# Author : Sukrit Shankar
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
# Necessary Imports
from __future__ import division, absolute_import

import os
import numpy as np
import scipy 
import scipy.io

import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import *
from tflearn.optimizers import Momentum

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Function to load single label data
def load_single_label_data(file_name, root_image_path, image_shape, normalize, grayscale):
	categorical_labels = False
	files_extension = None
	filter_channel = False
	with open(file_name, 'r') as f:
		images, labels = [], []
	    	for l in f.readlines():
			l = l.strip('\n').split()
		        if not files_extension or any(flag in l[0] for flag in files_extension):
		            if filter_channel:
		                if get_img_channel(l[0]) != 3:
		                    continue
		            images.append(root_image_path + l[0])
		            labels.append(int(l[1]))

	n_classes = np.max(labels) + 1
	labels_cat = to_categorical(labels,None)  # From List to List 
	X = ImagePreloader(images, image_shape, normalize, grayscale)
	Y = LabelPreloader(labels_cat, n_classes, categorical_labels)
	
	# Return X, Y and number of output classes 
	return X, Y, n_classes


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Function to load multi label data
def load_multi_label_data(file_name, root_image_path, image_shape, normalize, grayscale):
	categorical_labels = False
	with open(file_name, 'r') as f:
		count = 1
		for l in f.readlines():
			if (count == 1):
				l = l.strip('\n').split()
				n_classes = len(l) - 1 
			count = count + 1

	n_entries = count - 1

	with open(file_name, 'r') as f:
		images = []
		labels = np.zeros((n_entries,n_classes))
		count = 0
	    	for l in f.readlines():
			l = l.strip('\n').split() 
		    	images.append(root_image_path + l[0])

			for i in range(1,len(l)):
				labels[count,i-1] = int(l[i])
			count = count + 1

	X = ImagePreloader(images, image_shape, normalize, grayscale)
	Y = LabelPreloader(labels, n_classes, categorical_labels)
	
	# Return X, Y and number of output classes 
	return X, Y, n_classes



