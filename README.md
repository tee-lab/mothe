## MOTHe
Mothe is a PYPI library to detect and track multiple animals in a heterogeneous environment. MOTHe is a python based repository and it uses Convolutional Neural Network (CNN) architecture for the object detection task. It takes a digital image as an input and reads its features to assign a category. These algorithms are learning algorithms which means that they extract features from the images by using huge amounts of labeled training data. Once the CNN models are trained, these models can be used to classify novel data (images). MOTHe is designed to be generic which empowers the user to track objects of interest even in a natural setting.

This repository is the backend library of our GUI based app - [MOTHe-GUI](https://github.com/tee-lab/MOTHe-GUI) 


## PIPELINE DESCRIPTION:

MOTHe can automate all the tasks associated with object classification and is divided into 5 methods dedicated to the following tasks:

1. __System configuration__: The system configuration is used to setup MOTHe on the users system. Basic details such as the path to the local repository, path to the video to be processed, the size of the individial to be cropped, number of frames to skip while running detection or tracking (to reduce compute time/to run a test case) and the size of the bounding box to be drawn during the detection phase.

2. __Dataset generation__: The dataset generation is a crucial step towards object detection and tracking. The manual effort required to generate the required amount of training data is huge. The data generation class and executable highly automates the process by allowing the user to crop the region of interest by simple clicks over a GUI and automatically saves the images in the appropriate folders.

3. __Training the convolutional neural networktrain_model__: After generating sufficient number of training example, the data is used to train the neural network. The neural network produces a classifier as the output. The accuracy of the classifier is dependent on how well the network is trainied, which in turn depends on the quality and quantity of training data (See section __How much training data do I need?__). The various tuning parameters of the network are fixed to render the process easy for the users. This network works well for binary classification - object of interest (animals) and background. Multi-class classification is not supported on this pipeline.

4. __Object detection__: This method performs two key tasks - it first identifies the regions in the image which can potentially have animals, this is called localisation; then it performs classification on the cropped regions. This classification is done using a small CNN (6 convolutional layers). Output is in the form of *.csv* files which contains the locations of the identified animals in each frame.

5. __Object tracking__: Object tracking is the final goal of the MOTHe. This module assigns unique IDs to the detected individuals and generates their trajectories. We have separated detection and tracking modules, so that it can also be used by someone interested only in the count data (eg. surveys). This modularisation also provides flexibility of using more sophisticated tracking algorithms to the experienced programmers. We use an existing code for the tracking task (from the Github page of ref). This algorithm uses Kalman filters and Hungarian algorithm. This script can be run once the detections are generated in the previous step. Output is a \text{.csv} file which contains individual IDs and locations for each frame. A video output with the unique IDs on each individual is also generated.


### IMPORTANT NOTE

MOTHe is a python package which uses several other python libraries which might have been updated. Therefore, it is important to be aware of the versions that we download/install. The recommended python versions are the python3.6 to python3.7 stable releases (The latest LTS versions of linux (ex: Ubuntu 20.04 Focal Fossa) are installed with a stock python3.8 which is not compatible with MOTHe). Python3.8 does not support Tensorflow versions below the 2.2 releases which are required by MOTHe to work. Please note the versions of some libraries that are modified rather quickly and are used to test MOTHe very recently:

1. Tensorflow: 2.1.0
2. Keras: 2.3.1
3. sklearn: 0.23.1
4. opencv-python: 3.4.0
