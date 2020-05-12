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


# MOTHE IMPLEMENTATION

 __Users should first setup their system with mothe. To do this you simply have to use the pip package manager. Use the following command to install mothe.__

 **_pip install mothe_**

 1. __Step 1__: Set a variable to contain the mothe object

    To do this, call the mothe object and pass the 3 arguments
    a. Path to the project folder
    b. minimum threshold value (experimental. Suggested starting value is 50)
    c. maximum threshold value (experimental. Suggested starting value is 200)

    **_Mothe = mothe("path/to/the/project/folder", min_threshold, max_threshold)_**

 2. __step 2__: Generate the configuration file by using the 'set_config()' method

    Initiate the configuration by executing the following code snippet. The 'set_config()' method accepts 1 argument
    a. path to the video file
    A video file is required to determine the size of the bounding box. Ones the command is executed, a window with a frame from the selected video appears. The user is required to click and drag across the object of interest to set the size of the bounding box. On drawing the bounding box satisfactorily, press the 'c' button twice to save a configuration file in teh yaml format containing the root_path, min_threshold value, max_threshold value and size of the bounding box.

    **_Mothe.set_config("path/to/video")_**

 3. __step 3__: Generate the dataset using the 'generate_dataset()' method

    To generate the dataset, execute the following code. The 'generate_dataset()' method takes 2 Arguments
    a. path to the video
    b. class name ("yes" or "no")
    When the code is executed, a window appears with a frame of the video selected. to generate data, single click at the centre of the object of interest and press the 'a' button. The object gets cropped and stored in the folder bearing the class name provided as an argument to the method. A marker also appears on the selected object for confirmation on the pressing of the 'a' button. When the whole frame is covered, press the 'esc' key to move to the next frame. Generate data this way for both the classes.

    **_Mothe.generate_dataset("path/to/video", "class_name")_**

 4. __step 4__: Train the neural network using the 'train_model()' method

    Use the following code to train the neural network using the generated dataset. A trained model is produced and stored in the project folder which is used to detect and track the objects of interest. This method needs no argument.

    **_Mothe.train_model()_**

 5. __step 5__: Track multiple objects with unique ids using the 'tracking()' method

    Execute the following code to initiate the tracking process. The tracking method takes 2 Arguments
    a. video name
    b. model name
    The video name or the video path corresponds to the video in which you would like to track the objects. This method produces a csv file with framewise data of the individuals position and associated unique ids. It also produces a video file containing the tracks.

    **_Mothe.tracking("path/to/the/video/file", "path/to/the/trained/model")_**

6. __step 6__: Detect multiple objects using the 'detection()' method

   Execute the following code to initiate the detection process. The detection method takes 2 Arguments
   a. video name
   b. model name
   The video name or the video path corresponds to the video in which you would like to detect the objects. This method produces a csv file with framewise data of the individuals position. It also produces a video file containing the detections.

   **_Mothe.detection("path/to/the/video/file", "path/to/the/trained/model")_**
