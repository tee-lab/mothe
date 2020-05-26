# MOTHe
'mothe' is a pipeline developed to detect and track multiple objects in a heterogeneous environment. MOTHe is based on a simple
Convolutional neural network and allows users to generate training data using a simple GUI tool, automatically        detects
classified objects and tracks them. MOTHe is designed to be generic which empoweres the user to track      any
objects of interest in a natural setting.

__MOTHe has been developed and tested on Ubuntu 16.04 and above using python 3.6.9__
# PIPELINE DESCRIPTION:

The 'mothe' library includes 5 methods that provide an end to end solution for tracking multiple objects in a heterogeneous environment. It includes methods to setup configuration, dataset generation, training the CNN, multiple object detection and object tracking.

1. __set_config()__: The 'set_config()' method is used to setup MOTHe on the users system. Basic details such as minimum and maximum threshold values
   and the size of the individial to be cropped and the size of the bounding
   box to be drawn during the detection and tracking phase.  

2. __generate_dataset()__: The dataset generation is a crucial step towards object detection and tracking. The manual effort required
   to generate the required amount of training data is huge. The data generation  method highly automates the process
   by allowing the user to crop the region of interest by simple clicks over a GUI and automatically saves the images in appropriate
   folders. It is important however to crop the images consistantly since the accuracy of the classifier plays a huge role in the
   overall efficiency of the pipeline.

3. __train_model()__: After generating sufficient number of training example, the data is used to train the
   neural network. The neural network produces a classifier as the output. The accuracy of the classifier is dependent on how well
   the network is trained which in turn depends on the quality of data generation. The various tuning parameters of the network is
   fixed to render the process easy for the users. This network has proven its efficiency when it comes to binary classification
   (object of interest and background). Multi-class classification is not supported with this pipeline.

4. __detection()__: The weights produced by the trained network is used to classify various regions of the test frame. In the event
   of a positive classification, a square bounding box is drawn around that region annotating the object. Using methods like sliding
   window helps to cover an entire frame for region proposal but is computationally expensive. MOTHe employs the thresholding technique
   to identify the potential object of interest to be classified. This method allows us to utilize finite number of points around which
   the region is classified making it an efficient and fast process. The object detection phase of MOTHe provides the user with a csv
   file containing the coordinates of all positive classifications and a video file with the detected objects with bounding boxes.

5. __Object tracking__: Object tracking is the final goal of MOTHe. Unique ids are generated for all the detected objects. The ids are tracked through frames of the test video
   based on kalman filter. Few objects that go undetected for a few frames are reassigned with a new
   id. Additionnal individuals entering the frame are assigned a new id after detection.
   
**_NOTE: MOTHE DEPENDS ON VARIOUS OTHER MODULES FOR COMPUTATION AND IS SUBJECT TO MAINTENANCE OF THESE MODULES. THEREFORE, MAKE SURE TO KEEP YOUR OS UPDATED ALONG WITH THE PYTHON VERSION. FOR THIS REASON, IT IS ADVISED TO SET UP A PYTHON VIRTUAL ENVIRONMENT EXCLUSIVELY FOR MOTHE TO MAINTAIN A RUN ENVIRONMENT AND THE RIGHT MODULE VERSIONS. MOTHE HAS BEEN TESTED AND WORKS PERFECTLY ON UBUNTU 18.04 AND ABOVE WITH PYTHON 3.6 AND ABOVE.  _**


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

You will see the "successfully installed mothe" message if you have installed mothe with pip3 succesfully.

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/5_run_python3.png">
<br>

Run the python3 command in your terminal at this point to open the python shell. If you are in a new terminal, make sure you have navigated into the mothe folder before doing this.

**_python3_**

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/6_import_mothe.png">
<br>

In the python shell, import the mothe module by executing the following command.

**_from mothe.pipe import mothe_**

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/7_create_mothe_instance.png">
<br>

Create a mothe instance/object by initializing mothe using the following command. The min and max threshold values depends on the specific case study and maytake a few tril and error attempts to get right. For the black buck videos, we have chosen 0 and 150 as the min and max threshold values and 150 and 250 for the wasp videos.

**_mothe = mothe("path/to/the/project/folder", min_threshold, max_threshold)_**

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/8_set_configuration.png">
<br>

Set the configuration for mothe using the following command. This command also stores a config.yml file in the mothe directory.

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

After the configuration process, initiate the data generation process by following the following command. Make sure to use multiple videos for data generation to accomodate variations and to produce enough examples. Mothe supports only binary classification. Therefore name the classes 'yes' for positive examples and 'no' for background examples. 

**_mothe.generate_dataset("path/to/video", "class_name")_**

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/14_window_appears.png">
<br>

A window appears which is a frame of the video you have chosen. Start generating data at this point.

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/15_click_and_press_a.png">
<br>

Click at the center of the animal once. The algorithm calculates the size of the bounding box based on the config file entry. The press the 'a' button ones to crop and store the animal. Then click on the next animal. A green marker appears on the previous animal to suggest it has already been cropped. Repeat the process to store all the animal examples.

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

Select all animals in every frame. Repeat this process for the 'no' class too. Select all background examples in this case. At this point you will have two class folder with many examples to train the neural network.


<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/19_train_model.png">
<br>

Start training the neural network using the generated data by executing the following command.

**_mothe.train_model()_**

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/20_post_training_graphs.png">
<br>

After successfully training the model, two graphs appear on the screen.

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/21_stores_model.png">
<br>

After training, the model gets stored in the mothe directory as 'mothe_model.h5py'. This model will be used to detect and track animals in the test videos.


<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/22_start_detection.png">
<br>

Initiate the detection process by using the following command. Make sure to enter the name of a test video and the correct model name as arguments.

**_mothe.detection("path/to/the/video/file", "path/to/the/trained/model")_**


<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/23_stores_video_csv.png">
<br>

After the successful detection, a detection video and csv are generated in the mothe folder.

<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/24_start_tracking.png">
<br>

Initiate the tracking process by using the following command. Make sure to enter the name of a test video and the correct model name as arguments.

**_mothe.tracking("path/to/the/video/file", "path/to/the/trained/model")_**


<br>
<img height="350" src="https://github.com/tee-lab/mothe/blob/master/mothe_screenshots/25_stores_video_csv.png">
<br>

After the successful tracking, a tracking video and csv are generated in the mothe folder.
