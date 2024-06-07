# FUSION RADAR CAMERA BASED ON NEURAL NETWORK

This repository contains the software to perform sensor fusion by using data from images and radar data, trained using the Nuscenes library in two steps:

1. Monocular depth estimation: using existing algorithms for dense depth prediction from a single image, a relative depthmap is created, which is the first input to our algorithm. The employed algorithm here is *Depth anything*

2. Using the relative depthmap and the radar measurements, we turn the relative depthmap to an absolute depthmap using a lightweight neural network consisting of a U-net for the image and a PointNet for the radar point to extract their features and fusing them to perform dense depth estimation.

# REQUIREMENTS

We uploaded .yaml files of dependencies for Windows and Nvidia Jetson AGX Orin, using Python 3.8. If you prefer to download the dependencies independently, the main libraries are *nuscenes-devkit*, *torch*, *OpenCV*. *Depth-Anything* also requires its own libraries (https://github.com/LiheYoung/Depth-Anything.git).

Also, the Nuscenes dataset is needed (https://www.nuscenes.org/).

# STEPS

1. Execute, in order, the files inside the folder *prepare_dataset* modifying the variables *DIR_NUSCENES* (path of the Nuscenes dataset), *VERSION* (Nuscene's version, either the mini or the complete), and *DIR_DATA* (data where the new dataset including the relative depth will be stored). We generate in this step also the relative depthmaps using the last file, and for that the *Depth-Anything* library must be installed from its Github (https://github.com/LiheYoung/Depth-Anything.git).

2. Inside the *algorithm* folder, the *train_nuscenes.ipynb* and *test_nuscenes.ipynb* files perform training and test. If you change the architecture of the model (the variables *RADAR_CHANNELS_ENCODER*, *UNET_CHANNELS_IMG* and *UNET_CHANNELS_RADAR*) please note that there is a restriction in *RADAR_CHANNELS_ENCODER*, and the last number of the list hast o match the one obtained by the command *model.required_radar_size(depth_data)*, where *depth_data* is any vector of size (BATCH_SIZE, 1, HEIGHT_IMAGE, WIDTH_IMAGE). In *model.py*, there is an example that can be run with *python model.py*

