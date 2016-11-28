# ----------------------------------------------------------------------
# Train various CNN Architectures in TensorFlow (using TFlearn)
# Author : Sukrit Shankar
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
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

import shankar_load_data as  ssld
import shankar_load_cnn_arch as sslca
# import shankar_load_rnn_arch as sslra

####################################################################################################################
# Configuration Settings
####################################################################################################################

# ---------------------------------------------------------------
# -------------- Set Parameters for Running ---------------------
# Image Shape - Constant here - Prefers to be 3D 
cnn_image_shape_before_crop = [256,256,3]   # Shape before cropping
cnn_image_shape = [224, 224, 3] # Shape after cropping 

# Initial Learning Rate - Lower down incase loss keeps increasing (diverging trend)
cnn_initial_learning_rate = 0.1

# Decay Step Iteration
cnn_decay_step_size = 100

# Decay Multiplication Factor - Lower means faster decay 
cnn_decay_mult_factor = 0.96 

# Dropout (Keep) Probability in (0,1] [Note its the keep probability --- 1 => no dropout, ~0 => all dropout]
cnn_keep_probability = 0.5

# Regularization - L1 / L2 
cnn_regularization_type = 'L2'

# Regularization Weight Decay
cnn_regularization_weight_decay = 0.0001

# Batch Size - can lower upto 8 in case out of GPU memory (Sometimes with ResNets, batch size of 1 also works)
cnn_batch_size = 32

# Max Number of Epochs
cnn_max_num_epochs = 40  

# topK Accuracy Mode (1 to M-1) [M = number of labels per instance]
cnn_top_k_value = 1

# Strings with which the run will be identified
cnn_run_id_string = 'camtail'
cnn_checkpoint_path_string = 'model_googlenet' 

# -------------------------------------------------------------
# -----------------Set Parameters for Data Loading ------------
# Single Label case or multi label case (binary labels in both cases)
comm_multi_label_scenario = 0 # 0 = Single label 

# Train Val and Test File Paths
# For Single Label, the file format should be as follows:
# --- Each row contains information about one data instance, <relative_image_path_with_extension> <label>
# Root image path is prefixed to the image paths in the train and val txt files 
comm_train_text_file_SL = 'datasets/fashion_categories_camtail/train.txt'
comm_val_text_file_SL = 'datasets/fashion_categories_camtail/val.txt'
comm_root_image_path_SL = '/home/sukrit/Desktop/DEEP_LEARNING/CODES_OURS/tf_package_ver_1/datasets/fashion_categories_camtail/images_resized_nochange_ar/'

# For Multiple Labels, the file format should be as follows:
# --- Each row contains information about one data instance, <relative_image_path_with_extension> <label/value>  <label/value> ... 
# Root image path is prefixed to the image paths in the train and val txt files
comm_train_text_file_ML = 'datasets/face_attributes_celebA/train.txt'
comm_val_text_file_ML = 'datasets/face_attributes_celebA/val.txt'
comm_root_image_path_ML = '/home/sukrit/Desktop/DEEP_LEARNING/CODES_OURS/tf_package_ver_1/datasets/face_attributes_celebA/images/' #With / at the end 

# -------------------------------------------------------------
# -------------- Set Parameters for Data Preprocessing  -------
cnn_compute_mean_std = 0 # 1 = Yes, 0 = No 
cnn_mean_vector_precomputed = [0.70786846,  0.67048967,  0.65235734] # Used when cnn_compute_mean_std = 0
cnn_std_vector_precomputed = [0.23995902,  0.25372425,  0.25706375] # Used when cnn_compute_mean_std = 0

# -------------------------------------------------------------
# -------------- Set Parameters for Architecture Choice  ------
# CNN Architecture Choices 
# 0 = AlexNet
# 1 = GoogleNet (Inception-v3)
# 2 = VGG-11
# 3 = VGG-16 
# 4 = VGG-19
# 5 = ResNet (configurable)
# 6 = Wide ResNet (configurable)
# 7 = Dense Net (configurable)
# 8 = Highway Convolutional Networks (configurable)
cnn_arch_choice = 8
cnn_resNet_n = 3 # Only for ResNet (n = 5 => 32 layer network, n = 9 => 56 layers and so on) 
cnn_wideResNet_n = 3 # Number of Blocks for Wide ResNet 
cnn_wideResNet_k = 1 # Widening Ratio for Wide ResNet 
cnn_denseNet_depth = 5 # Depth for Dense Net (min 5)
cnn_denseNet_growth_ratio = 1 # Growth Ratio for Dense Net 
cnn_highway_block_depth = 3 # Block Depth for Highway Convolutional Networks

# RNN Architecture Choices 
# 0 = RNN
# 1 = LSTM 
# 2 = GRU 
# 3 = Bidirectional RNN
# 4 = Spatial LSTM 
# 5 = Spatial LSTM + RNN over the hidden states 
# 6 = Spatial LSTM + Input features from CNN(s)
# 7 = Static Graph Spatial LSTM 
# 8 = Spatial LSTM + Soft Attention over the input features 
rnn_arch_choice = 0

# Set static vs dynamic run mode in RNN architectures 
# 0 = Simple , 1 = Dynamic 
rnn_run_mode = 0 

# Set the simple MLP choice over vectors or vectorized images
mlp_choice = 1 # 0 = no MLP

# -------------------------------------------------------------
# -------------- Set Parameters for Loss Types  ---------------
# 0 = Softmax (Single label classification / implicit ranking of labels for a given data instance)
# 1 = Sigmoid Cross Entropy (Multi label classification / implicit ranking of data instances for a given label)
# 2 = Magnetic Loss (Metric Learning with Adaptive Density Discrimination)
# 3 = Triplet Loss  
# 4 = Contrastive Loss (for Siamese networks)
# 5 = Pairwise Ranking Loss 
# 6 = Euclidean Loss 
# 7 = Hinge Loss 
comm_loss_type = 0

# -------------------------------------------------------------
# -------------- Set Parameters for CNN Transfer Learning -----
cnn_transfer_learn = 1  # 0 = Means no transfer learn 
cnn_base_model_path = ''

# -------------------------------------------------------------
# -------------- Set Parameters for Testing & Visualization ----
# CNN Visualization
# 0 = Saliency Map (with Global Max Pool) 
# 1 = Average Spatial Response of a layer for an image 
# 2 = Correlation matrices for all classes at a given layer (class mean of average spatial responses)
# 3 = Spatial Responses at various layers for a given image
# 4 = Input reconstruction from certain layers of a trained net 
cnn_test = 0 # 1 = Test by dumping output probabilities for a test set 
cnn_vis = 0  # 1 = Visualize
cnn_vis_pretrained_net_path = ''

# RNN Visualization & Test 
rnn_test = 0 # 1 = Test 
rnn_vis = 0 # 1 = Visualize 
rnn_vis_pretrained_net_path = '' 

####################################################################################################################
# Data loading and preprocessing
####################################################################################################################

# Load the data manually 
if (comm_multi_label_scenario == 0):
	# Load Train and Validation Data 
	XTrain, YTrain, num_output_classes = ssld.load_single_label_data(comm_train_text_file_SL, comm_root_image_path_SL, (cnn_image_shape_before_crop[0], cnn_image_shape_before_crop[1]), normalize=True, grayscale=False)
	XVal, YVal, num_output_classes = ssld.load_single_label_data(comm_val_text_file_SL, comm_root_image_path_SL, (cnn_image_shape_before_crop[0], cnn_image_shape_before_crop[1]), normalize=True, grayscale=False)

if (comm_multi_label_scenario == 1):
	XTrain, YTrain, num_output_classes = ssld.load_multi_label_data(comm_train_text_file_ML, comm_root_image_path_ML, (cnn_image_shape_before_crop[0], cnn_image_shape_before_crop[1]), normalize=True, grayscale=False)
	XVal, YVal, num_output_classes = ssld.load_multi_label_data(comm_val_text_file_ML, comm_root_image_path_ML, (cnn_image_shape_before_crop[0], cnn_image_shape_before_crop[1]), normalize=True, grayscale=False)

# Image Preprocessing Methods Instantiation
cnn_img_prep = ImagePreprocessing()

if (cnn_compute_mean_std == 1):
	cnn_img_prep.add_featurewise_zero_center() # Zero Center (With mean computed over the whole dataset)
	cnn_img_prep.add_featurewise_stdnorm() # STD Normalization (With std computed over the whole dataset)
if (cnn_compute_mean_std == 0):
	cnn_img_prep.add_featurewise_zero_center(per_channel=True, mean=cnn_mean_vector_precomputed )
	cnn_img_prep.add_featurewise_stdnorm(per_channel=True,std=cnn_std_vector_precomputed)

cnn_img_aug = tflearn.ImageAugmentation() # Real-time data augmentation
cnn_img_aug.add_random_flip_leftright() # Random flip an image
cnn_img_aug.add_random_crop([cnn_image_shape[0], cnn_image_shape[1]])

####################################################################################################################
# Call the network architecture 
####################################################################################################################
if (comm_loss_type == 0):
	cnn_loss_layer_activation = 'softmax'
if (comm_loss_type == 1):
	cnn_loss_layer_activation = 'sigmoid'

# Call the network architectures 
if (cnn_arch_choice == 0):
	net = sslca.load_alexnet(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation) 
if (cnn_arch_choice == 1):
	net = sslca.load_googlenet_v3(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation) 
if (cnn_arch_choice == 2):
	net = sslca.load_vgg_11(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation) 
if (cnn_arch_choice == 3):
	net = sslca.load_vgg_16(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation)
if (cnn_arch_choice == 4):
	net = sslca.load_vgg_19(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation)
if (cnn_arch_choice == 5):
	net = sslca.load_resnet(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation, cnn_resNet_n)
if (cnn_arch_choice == 6):
	net = sslca.load_wide_resnet(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation, cnn_wideResNet_n, cnn_wideResNet_k)
if (cnn_arch_choice == 7):
	net = sslca.load_densenet(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation, cnn_denseNet_depth, cnn_denseNet_growth_ratio)
if (cnn_arch_choice == 8):
	net = sslca.load_highway(cnn_image_shape, cnn_img_prep, cnn_img_aug, cnn_keep_probability, num_output_classes, cnn_regularization_type, cnn_regularization_weight_decay, cnn_loss_layer_activation, cnn_highway_block_depth)


####################################################################################################################
# Train the network
####################################################################################################################

# Fill the training params 
momentum = Momentum(learning_rate=cnn_initial_learning_rate, lr_decay=cnn_decay_mult_factor, decay_step=cnn_decay_step_size)

# Set the loss 
if (comm_loss_type == 0):
	loss_string = 'categorical_crossentropy'
if (comm_loss_type == 1):
	loss_string = 'categorical_crossentropy'

network = regression(net, optimizer=momentum,
                     loss=loss_string,
                     learning_rate=cnn_initial_learning_rate)                     
                
# Set the network      
model = tflearn.DNN(network, checkpoint_path=cnn_checkpoint_path_string,
                    max_checkpoints=1, tensorboard_verbose=2)

# Train the network  
model.fit(XTrain, YTrain, n_epoch=cnn_max_num_epochs, validation_set=(XVal, YVal), shuffle=True,
          show_metric=True, batch_size=cnn_batch_size,
          snapshot_epoch=True, run_id=cnn_run_id_string)


####################################################################################################################
# Test and Visualize 
####################################################################################################################











