## MOTHe
Mothe is a pipeline developed to detect and track multiple animals in a heterogeneous environment. MOTHe is a python based repository and it uses Convolutional Neural Network (CNN) architecture for the object detection task. It takes a digital image as an input and reads its features to assign a category. These algorithms are learning algorithms which means that they extract features from the images by using huge amounts of labeled training data. Once the CNN models are trained, these models can be used to classify novel data (images). MOTHe is designed to be generic which empowers the user to track objects of interest even in a natural setting.

__MOTHe can be used on both Linux and windows operating systems. We have provided instructions for the windows system separately at the end of this page.__

## PIPELINE DESCRIPTION:

MOTHe can automate all the tasks associated with object classification and is divided into 5 methods (one command line for each method) dedicated to the following tasks:

1. __System configuration__: The system configuration is used to setup MOTHe on the users system. Basic details such as the path to the local repository, path to the video to be processed, the size of the individial to be cropped, number of frames to skip while running detection or tracking (to reduce compute time/to run a test case) and the size of the bounding box to be drawn during the detection phase.

2. __Dataset generation__: The dataset generation is a crucial step towards object detection and tracking. The manual effort required to generate the required amount of training data is huge. The data generation class and executable highly automates the process by allowing the user to crop the region of interest by simple clicks over a GUI and automatically saves the images in the appropriate folders.

3. __Training the convolutional neural networktrain_model__: After generating sufficient number of training example, the data is used to train the neural network. The neural network produces a classifier as the output. The accuracy of the classifier is dependent on how well the network is trainied, which in turn depends on the quality and quantity of training data (See section __How much training data do I need?__). The various tuning parameters of the network are fixed to render the process easy for the users. This network works well for binary classification - object of interest (animals) and background. Multi-class classification is not supported on this pipeline.

4. __Object detection__: This method performs two key tasks - it first identifies the regions in the image which can potentially have animals, this is called localisation; then it performs classification on the cropped regions. This classification is done using a small CNN (6 convolutional layers). Output is in the form of *.csv* files which contains the locations of the identified animals in each frame.

5. __Object tracking__: Object tracking is the final goal of the MOTHe. This module assigns unique IDs to the detected individuals and generates their trajectories. We have separated detection and tracking modules, so that it can also be used by someone interested only in the count data (eg. surveys). This modularisation also provides flexibility of using more sophisticated tracking algorithms to the experienced programmers. We use an existing code for the tracking task (from the Github page of ref). This algorithm uses Kalman filters and Hungarian algorithm. This script can be run once the detections are generated in the previous step. Output is a \text{.csv} file which contains individual IDs and locations for each frame. A video output with the unique IDs on each individual is also generated.

## Setting-up MOTHe on Linux

### IMPORTANT NOTE

MOTHe is a python package which uses several other python libraries which might have been updated. Therefore, it is important to be aware of the versions that we download/install. The recommended python versions are the python3.6 to python3.7 stable releases (The latest LTS versions of linux (ex: Ubuntu 20.04 Focal Fossa) are installed with a stock python3.8 which is not compatible with MOTHe). Python3.8 does not support Tensorflow versions below the 2.2 releases which are required by MOTHe to work. Please note the versions of some libraries that are modified rather quickly and are used to test MOTHe very recently:

1. Tensorflow: 2.1.0
2. Keras: 2.3.1
3. sklearn: 0.23.1

__If the environment has the wrong versions installed, just reinstall the package using pip3 and specifying the correct versions as shown below.__

`$ pip3 install tensorflow==2.1.0`

### VIRTUAL ENVIRONMENT SETUP

__Setting up a virtual environment is not mandatory but its prefereed as it makes the handling of required packages easy. Consider the case of having a fresh Ubuntu 20.04 install which boasts a python 3.8 integration by default. Installing a stable version of python is necessary. Virtual environments help us to maintain multiple environments on the same system and find the setup that works best. Follow the instructions below to setup a virtual environment:__

1. Install python3-dev and python3-tk modules required for mothe using the following commands-

`$ sudo apt-get install python3-dev`

`$ sudo apt-get install python3-tk`

*__If a new version of python is installed, ex: python3.6 along with the existing python3.8 stock install, the packages should be installed as follows__*

`$ sudo apt-get install python3.6-dev`

`$ sudo apt-get install python3.6-tk`

2. Install virtualenv and virtualenvwrapper using the pip3 package manager

`$ sudo pip3 install virtualenv virtualenvwrapper`

3. Execute the following FOUR commands one after the other in the terminal to update the `.bashrc` file

`$ echo -e "\n# virtualenv and virtualenvwrapper" >> ~/.bashrc`
`$ echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc`
`$ echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.bashrc`
`$ echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc`

4. After updating the `.bashrc` file, we need to source it to apply the changes using the following command

`$ source ~/.bashrc`

5. After setting up the virtualenv and virtualenvwrapper, create a virtualenv using the following command. Be aware of the python version used for creating this environment. Ex: If we have installed python3.6, copy the following command, replace version number with the python version on your system

`$ mkvirtualenv mothe -p python3.6`

6. After creating the virtual environment, we need to activate it before working on it. Activate the environment using the following command

`$ workon mothe`

7. Once we are in the mothe virtual environment, install mothe using the pip package manager as shown below

`$ pip install mothe`



### MOTHe implementation 

Users can run MOTHe to detect and track single or multiple individuals in the videos (or images). In this section, we describe the step-by-step procedure to run/test MOTHe. If you are interested in learning the procedure first by running it on our videos (and data), follow the guidelines under subsection **"Testing"** in each step.

__The mouse callback functions of OPENCV depend on Tkinter module. Please install it by using the follwing command in the terminal__

`$ sudo apt-get install python3-tk python3-dev`

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/1_create_mothe_directory.png">
<br>

Open the terminal and navigate to the desktop. Create a folder named "mothe" and navigate into this folder. Execute the following commands:

`$ cd Desktop`

`$ mkdir mothe`

`$ cd mothe`

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/2_copy_video_to_mothe.png">
<br>

Copy your test video to the "mothe" folder to configure settings. Also, copy all the videos (multiple videos required for generating dataset, detection and tracking) to the mothe folder. This tutorial shows the working of the mothe library with just one test video. It is preferable to use a subset of videos covering a wide-variety of scenario (habita, lighting, groups etc.) for the training purpose.

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/3_pip3_install_mothe.png">
<br>

Install the MOTHe library using the pip3 package manager (if not done already). In case you do not have pip3 installed, Execute the following command.

`$ sudo apt-get install python3-pip`

Forego the previous step if you have pip3 installed. Execute the following command to install mothe.

`$ pip3 install mothe`

If you are facing trouble installing MOTHe or you get errors while using MOTHe, it is most likely attributed to the descripency of version of the modules used in MOTHe and modules instaled on your system. You can choose to setup a virtual environment at this point and only install mothe in this environment (explained in the previous section).

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/4_post_successful_mothe_installation.png">
<br>

You will see the "successfully installed mothe" message if you have installed mothe with pip3 successfully. Some warnings may appear for various reasons during this step. But as long as the mothe module can be imported in python, it is not a problem.

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/5_run_python3.png">
<br>

Run the python3 command in your terminal at this point to open the python shell. If you are in a new terminal, make sure you have navigated into the mothe folder and activated the virtual environment before doing this.

`$ python3`

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/6_import_mothe.png">
<br>

In the python shell, import the MOTHe module by executing the following command. Some warnings may be printed after the import (generally something related to tensorflow and its compatibility with the GPU on the specific system). However, these warnings may be ignored.

`$ from mothe.pipe import mothe`

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/7_create_mothe_instance.png">
<br>

Now MOTHe library is successfully imported in our system and we can proceed with the methods associated with our object detection tasks.

__Step 1: System configuration__

This step is used to set parameters of MOTHe. All the parameters are saved in *config.yml*.
Parameters to be set in this step - home directory, cropping size of animals in the videos, path to video files etc. 

Create a mothe instance/object by initializing mothe using the following command. The min and max threshold values depends on the specific case study (contrast between animal and background) and may take a few trial and error attempts to get it right (Read section **Choosing color threshold** for more details). For the blackbuck videos, we have chosen 0 and 150 as the min and max threshold values and 150 and 250 for the wasp videos. You will also specify a step size (no. of frames to skip for detection and tracking task). If for any reason, you want to run the detection for every n frames instead of all the frames (it can spped up the detection task significantly).

`$ mothe = mothe("path/to/the/project/folder", min_threshold, max_threshold, step_size_for_detection_and_tracking)`

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/8_set_configuration.png">
<br>

Set the configuration for mothe using the following command. Assuming that we are in the folder which we have named mothe (choice of the user), *config.yml* is generated in this directory.

`$ mothe.set_config("path/to/video")`

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/9_window_appears.png">
<br>

A window appears during the configuration process which is a frame of the test video you have chosen. This step is to determine the size of the animal to be detected and tracked. This value also helps in the dataset generation phase. Make sure to choose the most accomodating animal on the screen to avoid occlusions and missed detections later.

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/10_drag_across_animal.png">
<br>

Click and drag across the animal to set the animal size for the configuration.


<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/11_press_c_twice.png">
<br>

Press the **c** key once to view the cropped animal. If satisfied with the click and drag process, proceed to press the **c** key **again** to confirm and end the configuration process.


<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/12_creates_config_file.png">
<br>

You can view and change the *config.yml* file created in the mothe folder.


<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/13_generate_dataset.png">
<br>

After the configuration process, initiate the data generation process using next step

__Step 2: Data Generation__


This program will pick frames from the videos, user can click on animals or background in these images to create samples for both categories (animal and background). Examples of animal of ineterst will be saved in the folder **yes** and background in the folder **no**.
User needs to generate at least 8k-10k samples for each category (see section **How much training data do I need?** for detailed guidelines). One must ensure to take a wide representation of forms in which animals appears in the videos and same for the background variation.

Mothe supports only binary classification. Therefore name the classes 'yes' for positive examples and 'no' for background examples. The data generation method takes a **step size** argument as well which helps the user to keep the number of examples per video in check. (Ex: a higher step size limits the number of frames per video. if a video is very long, one can set a higher step size to skip through unwated and consecutive frames.) 

`$ mothe.generate_dataset("path/to/video", "class_name")`

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/14_window_appears.png">
<br>

A window appears which is a frame of the video you have chosen. Start generating data at this point.

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/15_click_and_press_a.png">
<br>

Click at the center of the animal once. The algorithm calculates the size of the bounding box based on the config file entry. Press the **s** key once to crop and store the animals once we have selected all the animals in the frame. Then it will take us to the next frame automatically. Press the **n** key to proceed to the next frame if the current frame is not worth collecting data from. Any selected animals are not cropped and stored if **n** key is pressed. It just takes us to the next frame. Press the **u** key if you want to undo a perticular selection that you have made. Once you are done collectiong samples from a video, press **esc** key to complete the process for this video. You shall repeat this process for multiple videos to sample training examples as widely as possible.

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/16_creates_class_folder.png">
<br>

At this point, a class folder is created in the mothe folder which stores all animal examples.

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/17_starts_storing_data.png">
<br>

Data starts to get stored in the class folder.


<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/18_select_all_individuals.png">
<br>

Repeat this process for the 'no' class too. Select all background examples in this case. At this point you will have two class folder with many examples to train the neural network.


<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/19_train_model.png">
<br>

**For testing:**
If you wish to test (learn how to run) this module, download our video clips from [here](https://figshare.com/s/82661a4fd39008fae445). You can then generate samples by choosing any of these videos. If you directly want to proceed to next steps, download our training data from the same drive.


**Step 3: Training the CNN**

 To train the neural network, run the following command in the python shell-
 
`$ mothe.train_model()`

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/20_post_training_graphs.png">
<br>

After successfully training the model, two graphs appear on the screen. The loss graph starts at a higher point and if the correct learning rate is applied, it takes a drastic decline and starts to plateau out as it reaches near zero. If a very high learning rate is applied, the graph starts travelling upwards instead of downwards. If a slightly higher learning rate is applied, it will not reack a closer point towards the zero line. The accuracy curve should travel upwards sharply and plateau out. It is important to avoid over fitting of data. This can be done by using adequate variance in the examples we generate during data generation. It is also important not to have too much variance since the accuracy may go down even though the network can generalize fairly well. For this stage, please use the link provided below to use the already generated data to train the network. 

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/21_stores_model.png">
<br>

After training, the model gets stored in the mothe directory as *mothe_model.h5py*. This model will be used to detect and track animals in the test videos.


<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/22_start_detection.png">
<br>

**For testing -**
You can completely skip this step if you want to run MOTHe on wasp or blackbuck videos. For these videos, trained models are already saved in the repository.

**Step 4: Object detection**

This module will detect the animals (object of interest) in the video frames. As mentioned earlier, this process is done in two steps - first the code predicts the areas in which animal may be potentially present (localisation) and then these areas are passes to the network for classification task. For localisation, we need thrsholding approach which gives us regions which have animals as well as background noise.
Initiate the detection process by using the following command. Make sure to enter the name of a test video and the correct model name as arguments. 
You can use the already trained model availab;le in MOTHe Github repository to run detection on blackbuck or wasp videos. Rename the model and provide the same in the argument section of the detection method.

`$ mothe.detection("path/to/the/video/file", "path/to/the/trained/model")`


<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/23_stores_video_csv.png">
<br>

After the successful detection, a detection video and *.csv* are generated in the mothe folder.

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/24_start_tracking.png">
<br>

**STep 5: Object tracking**
This step is used to ascribe unique IDs to the detected animals and it gives us thetrajectoris of the animals. 
It will use the detections from previous step. Hence, the input for this step would be original video clip and *.csv* generated in the previous step.
Initiate the tracking process by using the following command. Make sure to enter the name of a test video and the correct model name as arguments. The model used here is the same model that is used by the detection step.

`$ mothe.tracking("path/to/the/video/file", "path/to/the/trained/model")`


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


## Instrictions for the implementation in WINDOWS

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

