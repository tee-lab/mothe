## MOTHe
Mothe is a pipeline developed to detect and track multiple animals in a heterogeneous environment. MOTHe is a python based repository and it uses Convolutional Neural Network (CNN) architecture for the object detection task. It takes a digital image as an input and reads its features to assign a category. These algorithms are learning algorithms which means that they extract features from the images by using huge amounts of labeled training data. Once the CNN models are trained, these models can be used to classify novel data (images). MOTHe is designed to be generic which empowers the user to track objects of interest even in a natural setting.

## WARNING

*__WARNING: MOTHe uses several methods which have either been moved or changed. It is important to be aware of the versions that we download/install. The recommended python versions are the python3.6 to python3.7 stable releases (The latest LTS versions of linux (ex: Ubuntu 20.04 Focal Fossa) are installed with a stock python3.8 which is not compatible with MOTHe). Python3.8 does not support Tensorflow versions below the 2.2 releases which are required by MOTHe to work. Please note the versions of some libraries that are modified rather quickly and are used to test MOTHe very recently.__*

1. Tensorflow: 2.1.0
2. Keras: 2.3.1
3. sklearn: 0.23.1

*__If the environment has the wrong versions installed, just reinstall the package using pip3 and specifying the correct versions as shown below.__*

*__pip3 install tensorflow==2.1.0__*

## VIRTUAL ENVIRONMENT SETUP

*__QUICK TIP: Consider setting up a virtual environment which makes handling the required packages easy. Consider the case of having a fresh Ubuntu 20.04 inatall which boasts a python 3.8 integration by default. Installing a stable version of python is nessasary. Virtual environments help us to maintain multiple environments on the same system and find the setup that works best. Follow the instructions below to setup a virtual environment.__*

1. Install the python3-dev and the python3-tk modules required for mothe using the following commands.

*__sudo apt-get install python3-dev__*

*__sudo apt-get install python3-tk__*

*__If a new version of python is installed, ex: python3.6 along with the existing python3.8 stock install, the packages should be installed as follows__*

*__sudo apt-get install python3.6-dev__*

*__sudo apt-get install python3.6-tk__*

2. Install virtualenv and virtualenvwrapper using the pip3 package manager

*__sudo pip3 install virtualenv virtualenvwrapper__*

3. Execute the following FOUR commands one after the other in the terminal to update the .bashrc file

*__echo -e "\n# virtualenv and virtualenvwrapper" >> ~/.bashrc__*

*__echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc__*

*__echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.bashrc__*

*__echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc__*

4. After updating the .bashrc file, we need to source it to apply the changes using the following command

*__source ~/.bashrc__*

5. After setting up the virtualenv and virtualenvwrapper, create a virtualenv using the following command. Be aware of the python version used for creating this environment. Ex: If we have installed python3.6, the version should be pointed towards python3.6

*__mkvirtualenv mothe -p python3.6__*

6. After creating the virtual environment, we need to activate it before working on it. Activate the environment using the following command

*__workon mothe__*

7. Once we are in the mothe virtual environment, install mothe using the pip package manager as shown below

*__pip install mothe__*

## PIPELINE DESCRIPTION:

The 'mothe' library includes 5 methods that provide an end to end solution for tracking multiple objects in a heterogeneous environment. It includes methods to setup configuration, dataset generation, training the CNN, multiple object detection and object tracking.

1. __set_config__: The system configuration is used to setup MOTHe on the users system. Basic details such as the path to the local repository, path to the video to be processed, the size of the individial to be cropped, number of frames to skip while running detection or tracking (to reduce compute time/to run a test case) and the size of the bounding box to be drawn during the detection phase.

2. __generate_dataset__: The dataset generation is a crucial step towards object detection and tracking. The manual effort required to generate the required amount of training data is huge. The data generation class and executable highly automates the process by allowing the user to crop the region of interest by simple clicks over a GUI and automatically saves the images in appropriate folders.

3. __train_model__: After generating sufficient number of training example, the data is used to train the neural network. The neural network produces a classifier as the output. The accuracy of the classifier is dependent on how well the network is trainied, which in turn depends on the quality and quantity of training data (See section "How much training data do I need?"). The various tuning parameters of the network are fixed to render the process easy for the users. This network works well for binary classification - object of interest (animals) and background. Multi-class classification is not supported on this pipeline.

4. __detection__: This is the most crucial module in the repository. It performs two key tasks - it first identifies the regions in the image which can potentially have animals, this is called localisation; then it performs classification on the cropped regions. This classification is done using a small CNN (6 convolutional layers). Output is in the form of .csv files which contains the locations of the identified animals in each frame.

5. __Object tracking__: Object tracking is the final goal of the MOTHe. This module assigns unique IDs to the detected individuals and generates their trajectories. We have separated detection and tracking modules, so that it can also be used by someone interested only in the count data (eg. surveys). This modularisation also provides flexibility of using more sophisticated tracking algorithms to the experienced programmers. We use an existing code for the tracking task (from the Github page of ref). This algorithm uses Kalman filters and Hungarian algorithm. This script can be run once the detections are generated in the previous step. Output is a \text{.csv} file which contains individual IDs and locations for each frame. A video output with the unique IDs on each individual is also 

# MOTHE IMPLEMENTATION

**_THE MOUSE CALLBACK FUNCTIONS OF OPEN-CV DEPENDS ON Tkinter MODULE. PLEASE INSTALL IT BY USING THE FOLLOWING COMMAND IN THE TERMINAL_**

**_sudo apt-get install python3-tk python3-dev_**

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/1_create_mothe_directory.png">
<br>

Open the terminal and navigate to the desktop. Create a folder named "mothe" and navigate into this folder. Execute the following commands.

**_cd Desktop_**

**_mkdir mothe_**

**_cd mothe_**

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/2_copy_video_to_mothe.png">
<br>

Copy your test video to the mothe folder to configure mothe settings. Also copy all the videos (multiple videos required for generating dataset, detection and tracking) to the mothe folder. This tutorial shows the working of the mothe library with just one test video.

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/3_pip3_install_mothe.png">
<br>

Install the mothe library using the pip3 package manager. In case you do not have pip3 installed, Execute the following command.

**_sudo apt-get install python3-pip_**

Forego the previous step if you have pip3 installed. Execute the following command to install mothe.

**_pip3 install mothe_**

Incase you are facing trouble installing mothe or you get errors while using mothe, it is most likely attributed to the descripency of version of the modules used in mothe and modules instaled on your system. You can choose to setup a virtual environment at this point and only install mothe in this environment.

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/4_post_successful_mothe_installation.png">
<br>

You will see the "successfully installed mothe" message if you have installed mothe with pip3 succesfully. Some warnings may appear for various reasons during this step. But as long as the mothe module can be imported in python, it is not a problem.

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/5_run_python3.png">
<br>

Run the python3 command in your terminal at this point to open the python shell. If you are in a new terminal, make sure you have navigated into the mothe folder before doing this.

**_python3_**

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/6_import_mothe.png">
<br>

In the python shell, import the mothe module by executing the following command. Some warnings may be printed after the import (generally something related to tensorflow and its compatibility with the GPU on the specific system). However, these warnings may be conviniently ignored.

**_from mothe.pipe import mothe_**

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/7_create_mothe_instance.png">
<br>

Create a mothe instance/object by initializing mothe using the following command. The min and max threshold values depends on the specific case study and maytake a few tril and error attempts to get right. For the black buck videos, we have chosen 0 and 150 as the min and max threshold values and 150 and 250 for the wasp videos.

**_mothe = mothe("path/to/the/project/folder", min_threshold, max_threshold, step_size_for_detection_and_tracking)_**

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/8_set_configuration.png">
<br>

Set the configuration for mothe using the following command. Assuming that we are in the folder which we have named mothe (choice of the user), config.yml is generated in this directory.

**_mothe.set_config("path/to/video")_**

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/9_window_appears.png">
<br>

A window appears during the configuration process which is a frame of the test video you have chosen. This step is to determine the size of the animal to be detected and tracked. This value also helps in the dataset generation phase. Make sure to choose the most accomodating animal on the screen to avoid occlusions and missed detections later.

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/10_drag_across_animal.png">
<br>

Click and drag across the animal to set the animal size fro the configuration.


<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/11_press_c_twice.png">
<br>

Press the 'c' button ones to view the cropped animal. If satisfied with the click and drag process, proceed to press the 'c' button again to confirm and end the configuration process.


<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/12_creates_config_file.png">
<br>

You can view and change the config.yml file created in the mothe folder later.


<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/13_generate_dataset.png">
<br>

After the configuration process, initiate the data generation process by following the following command. Make sure to use multiple videos for data generation to accomodate variations and to produce enough examples. Mothe supports only binary classification. Therefore name the classes 'yes' for positive examples and 'no' for background examples. The data generation method takes a step size argument as well which helps the user to keep the number of examples per video in check. (Ex: a higher step size limits the number of frames per video. if a video is very long, one can set a higher step size to skip through unwated and consecutive frames.) 

**_mothe.generate_dataset("path/to/video", "class_name")_**

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/14_window_appears.png">
<br>

A window appears which is a frame of the video you have chosen. Start generating data at this point.

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/15_click_and_press_a.png">
<br>

Click at the center of the animal once. The algorithm calculates the size of the bounding box based on the config file entry. Press the 's' button ones to crop and store the animals once we have selected all the animals in the frame. Then it will take us to the next frame automatically. Press the 'n' button to proceed to the next frame if the current frame is not worth collecting data from. Any selected animals are not cropped and stored if 'n' key is pressed. It just takes us to the next frame. Press the 'u' button if you want to undo a perticular selection that you have made.

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/16_creates_class_folder.png">
<br>

At this point, a class folder is created in the mothe folder which stores all animal examples.

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/17_starts_storing_data.png">
<br>

Data starts to get stored in the class folder


<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/18_select_all_individuals.png">
<br>

Select all animals in every frame. Repeat this process for the 'no' class too. Select all background examples in this case. At this point you will have two class folder with many examples to train the neural network. The 'esc' key maybe used to end the process at any stage.


<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/19_train_model.png">
<br>

Start training the neural network using the generated data by executing the following command.

**_mothe.train_model()_**

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/20_post_training_graphs.png">
<br>

After successfully training the model, two graphs appear on the screen. The loss graph starts at a higher point and if the correct learning rate is applied, it takes a drastic decline and starts to plateau out as it reaches near zero. If a very high learning rate is applied, the graph starts travelling upwards instead of downwards. If a slightly higher learning rate is applied, it will not reack a closer point towards the zero line. The accuracy curve should travel upwards sharply and plateau out. It is important to avoid over fitting of data. This can be done by using adequate variance in the examples we generate during data generation. It is also important not to have too much variance since the accuracy may go down even though the network can generalize fairly well. For this stage, please use the link provided below to use the already generated data to train the network. 

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/21_stores_model.png">
<br>

After training, the model gets stored in the mothe directory as 'mothe_model.h5py'. This model will be used to detect and track animals in the test videos.


<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/22_start_detection.png">
<br>

Initiate the detection process by using the following command. Make sure to enter the name of a test video and the correct model name as arguments. Please use the link provided below to use an already trained model. Rename the model and provide the same in the argument section of the detection method.

**_mothe.detection("path/to/the/video/file", "path/to/the/trained/model")_**


<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/23_stores_video_csv.png">
<br>

After the successful detection, a detection video and csv are generated in the mothe folder.

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/24_start_tracking.png">
<br>

Initiate the tracking process by using the following command. Make sure to enter the name of a test video and the correct model name as arguments. The model used here is the same model that is used by the detection step.

**_mothe.tracking("path/to/the/video/file", "path/to/the/trained/model")_**


<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/25_stores_video_csv.png">
<br>

After the successful tracking, a tracking video and csv are generated in the mothe folder.

## HOW MUCH TRAINING DATA DO I NEED?

MOTHe uses a CNN which uses a set of labelled examples to learn the features of the objects. Neural Networks generally work well with huge number of training samples. We recommend using at least 8-10k image examples for the animal category. This number may need to be increased if the animal of interest shows a lot of variation in morphology. For example, if males and females are of different colors, it is important to include sufficient examples for both of them. Similarly, if the background is too heterogeneous then you may need more training data (around 1500-2000 samples for different types of variations in the background).
For example to train the MOTHe on our blackbuck videos, we used 9800 cropped samples for blackbuck (including males and females) and 19000 samples for the background because background included grass, soil, rocks, bushes, water etc.


## CHOOSING COLOR THRESHOLDS

The object detection steps requires user to enter threshold values in the config files. Object detection in MOTHe works in two steps, it first uses a color filter to identify the regions in the image on which to run the classification. We use color threshold to select these regions. You can see the values of thresholds for blackbuck and wasp videos in the *config.yml* file.
If you are running MOTHe on a new dataset, there are two ways to select appropriate threshold values:

1. You may open some frames from different videos in an interactive viewer (for example MATLAB image display), you can then click on the pixels on animal and check the RGB values (take the avergae of all 3 channels). Looking at these values in multiple instances will give you an idea to choose a starting minimum and maximum threshold. 
Run the detection with these thresholds and you can improve the detection by hit and trial method to tweak the threshold.

2. You can compare your videos to wasp and blackbuck videos and start with threshold values to which your data is more similar. For example, if your animal looks more similar to blackbuck in color and lighting conditions, you may start with default thresholds and improve the detection by changing lower and upper threshold by little amount at a time.


## USING ON WINDOWS
Using windows to implement MOTHe is easier than the linux counterpart. There are TWO options for implementing with windows.

1. Using an environment such as anaconda.
Anaconda helps in installing packages from the Pypi repository easily and also contains useful tools such as spyder(Text editor for python projects) and jupyter notebooks for experimenting with the Mothe library. Use the following link to download anaconda and get it up and running. It provides clear documentation on how to install pypi packages in the environment.
*__https://docs.anaconda.com/anaconda/install/windows/_**

2. Installing python3 and pip package manager.
Downloading and installing python3 and the pip package manager is the __recommended option__ since it is quick and easy for controlling the versions. Use the following links to download and install python3 and pip package manager.
*__https://www.python.org/downloads/windows/__*
*__https://www.liquidweb.com/kb/install-pip-windows/__*

3. Install mothe using the pip package manager
Use the following command to install mothe
*__pip3 install mothe__* / *__pip install mothe__*
(Depending on whether we are in a virtualenv or on system)

*__WARNING: MOTHe uses several methods which have either been moved or changed. It is important to be aware of the versions that we download/install. The recommended python versions are the python3.6 to python3.7 stable releases. Python3.8 does not support Tensorflow versions below the 2.2 releases which are required by MOTHe to work. Please note the versions of some libraries that are modified rather quickly and are used to test MOTHe very recently.__*

1. Tensorflow: 2.1.0
2. Keras: 2.3.1
3. sklearn: 0.23.1
*__If the environment has the wrong versions installed, just reinstall the package using pip3 and specifying the correct versions as shown below.__*

*__pip3 install tensorflow==2.1.0__*

