# ECC_tensorflow_package
Comprehensive Tensorflow Package for various CNN variants, RNN variants, visualization and advanced losses. 

Flow starts from main.py. One needs to just set the configuration parameters, and leave the rest to the package. Specifically, we aim to support the following:

----------------------------
CNN Architectures
----------------------------
(0) AlexNet <br />
(1) GoogleNet (Inception-v4) <br />
(2) VGG-11, VGG-16, VGG-19 <br />
(3) ResNet (configurable) <br />
(4) Wide ResNet (configurable) <br />
(5) Dense Net (configurable)
(6) Highway Convolutional Networks (configurable)


----------------------------
Loss Objectives  
----------------------------
(0) L1 / L2 regularization (supported for all losses below)
(1) Softmax (Single label classification / implicit ranking of labels for a given data instance)
(2) Cross Entropy (Multi label classification / implicit ranking of data instances for a given label)
(3) Magnetic Loss (Metric Learning with Adaptive Density Discrimination)
(4) Triplet Loss  
(5) Contrastive Loss (for Siamese networks)
(6) Pairwise Ranking Loss 
(7) Euclidean Loss 
(8) Hinge Loss 

---------------------
Visualization  
---------------------
(0) Net Architecture 
(1) Saliency Map (with Global Max Pool) 
(2) Average Spatial Response of a layer for an image 
(3) Correlation matrices for all classes at a given layer (class mean of average spatial responses)
(4) Spatial Responses at various layers for a given image
(5) Input reconstruction from certain layers of a pretrained net 

---------------------------
Transfer Learning
---------------------------
(1) Finetuning last layer for a new dataset 
(2) Finetuning multiple new layers at the end 
(3) Finetuning multiple layers choosing from anywhere in the pretrained net 

------------------------------------------------
Time Series Modelling (Recurrent Architectures)   
------------------------------------------------
(0) Testing and Visualization
(1) RNNs
(2) LSTMs
(3) GRUs
(4) Bidirectional RNNs
(5) RNN Pixels
(6) Spatial LSTM 
(7) Spatial LSTM + RNN over the hidden states 
(8) Spatial LSTM + Soft Attention over the input features 
(9) Spatial LSTM + Input features from CNN(s)
(10) Static Graph Spatial LSTM 

-------------
INSTALLATION
-------------
(1) Read detailed instructions at https://docs.google.com/document/d/1y57HmFuxD64CQxuQB-equP9Q0BcxvxrBboDqeGwv5S0/edit?usp=sharing. We have personally found out that Tensorflow Device Creation works much faster with Cuda 7.5 in comparison to Cuda 8.0. Also Ubuntu 14.04 seems more stable than 16.04 in general; however, after following the installation instructions for either of them, things should be fine. In all cases Tensorflow version installed will be >= 0.11.0 

(2) Install TFLearn simply using "pip install tflearn". This requires for a Tensorflow version >= 0.9.0, which will be automatically satisfied with (1) above. 

------------------------------
NOTES ON RUNNING WITh TFLEARN 
------------------------------
(1) For some reason, TFLearn occupies maximum amount of GPU memory available, even when it needs less. We are still debugging into its causes, and potential fallacies in TFLearn. 

(2) In case one needs to stop the execution in between, use Ctrl-C and not Ctrl-Z. The former clears up all memory occupied in the GPU, while the latter does not. In case memory is held up, use nvidia-smi to list the process Ids and note the one consuming the undesired memory, and then kill that process Id by using kill -9 <pid>.
