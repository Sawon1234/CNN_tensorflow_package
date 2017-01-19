# CNN_tensorflow_package
Tensorflow Package for various CNN variants. 

Flow starts from main.py. One needs to just set the configuration parameters, and leave the rest to the package. Specifically, following are supported, and we will be constantly updaitng this space with more variants:

----------------------------
Annotations
----------------------------
(0) Single Label / Class Scenario <br/>
(1) Multi Label / Class Scenario 

----------------------------
CNN Architectures
----------------------------
(0) AlexNet <br />
(1) GoogleNet (Inception-v3) <br />
(2) VGG-11, VGG-16, VGG-19 <br />
(3) ResNet (configurable) <br />
(4) Wide ResNet (configurable) <br />
(5) Dense Net (configurable)<br />
(6) Highway Convolutional Networks (configurable)<br />


----------------------------
Loss Objectives  
----------------------------
(0) L1 / L2 regularization (supported for all losses below)<br />
(1) Softmax (Single label classification / implicit ranking of labels for a given data instance)<br />
(2) Cross Entropy (Multi label classification / implicit ranking of data instances for a given label)<br />

-------------
INSTALLATION
-------------
(1) Read detailed instructions at https://docs.google.com/document/d/1y57HmFuxD64CQxuQB-equP9Q0BcxvxrBboDqeGwv5S0/edit?usp=sharing. We have personally found out that Tensorflow Device Creation works much faster with Cuda 7.5 in comparison to Cuda 8.0. Also Ubuntu 14.04 seems more stable than 16.04 in general; however, after following the installation instructions for either of them, things should be fine. In all cases Tensorflow version installed will be >= 0.11.0 

(2) Install TFLearn simply using "pip install tflearn". This requires for a Tensorflow version >= 0.9.0, which will be automatically satisfied with (1) above. 

------------------------------
NOTES ON RUNNING WITH TFLEARN 
------------------------------
(1) For some reason, TFLearn occupies maximum amount of GPU memory available, even when it needs less. We are still debugging into its causes, and potential fallacies in TFLearn. Latest versions of TFLearn are beginning to fix this issue however. 

(2) In case one needs to stop the execution in between, use Ctrl-C and not Ctrl-Z. The former clears up all memory occupied in the GPU, while the latter does not. In case memory is held up, use nvidia-smi to list the process Ids and note the one consuming the undesired memory, and then kill that process Id by using kill -9 <pid>.
