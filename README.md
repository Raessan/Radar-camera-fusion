# FUSION RADAR CAMERA BASED ON NEURAL NETWORK

This repository contains the software to perform sensor fusion by using data from images and radar data, trained using the Nuscenes library in two steps:

1. Monocular depth estimation: using existing algorithms for dense depth prediction from a single image, a relative depthmap is created, which is the first input to our algorithm. The employed algorithm here is *Depth anything*

2. Using the relative depthmap and the radar measurements, we turn the relative depthmap to an absolute depthmap using a lightweight neural network consisting of a U-net for the image and a PointNet for the radar point to extract their features and fusing them to perform dense depth estimation.

# REQUIREMENTS

We uploaded .yaml files of dependencies for Windows and Nvidia Jetson AGX Orin, using Python 3.8. If you prefer to download the dependencies independently, the main libraries are *nuscenes-devkit*, *torch*, *scikit-image*, *Depth-Anything* which also requires its own libraries (https://github.com/LiheYoung/Depth-Anything.git).

Also, the Nuscenes dataset is needed (https://www.nuscenes.org/).

# STEPS FOR TRAINING AND TESTING

1. Execute, in order, the files inside the folder *prepare_dataset* modifying the variables *DIR_NUSCENES* (path of the Nuscenes dataset), *VERSION* (Nuscene's version, either the mini or the complete), and *DIR_DATA* (data where the new dataset including the relative depth will be stored). We generate in this step also the relative depthmaps using the last file, and for that the *Depth-Anything* library must be installed from its Github (https://github.com/LiheYoung/Depth-Anything.git).

2. Inside the *algorithm* folder, the *train_nuscenes.ipynb* and *test_nuscenes.ipynb* files perform training and test. If you change the architecture of the model (the variables *RADAR_CHANNELS_ENCODER*, *UNET_CHANNELS_IMG* and *UNET_CHANNELS_RADAR*) please note that there is a restriction in *RADAR_CHANNELS_ENCODER*, and the last number of the list hast o match the one obtained by the command *model.required_radar_size(depth_data)*, where *depth_data* is any vector of size (BATCH_SIZE, 1, HEIGHT_IMAGE, WIDTH_IMAGE). In *model.py*, there is an example that can be run with *python model.py*

# STEPS FOR INFERENCE

An inference folder that employs TensorRT in C++ is also present. It has been tested in Nvidia ORIN, which supports TensorRT through its Jetpack 5.1. You need to add your Depth Anything model in .onnx format inside the *models* folder. In order to build the package and execute, go to the *inference* folder and type:

```bash
mkdir build
cd build
cmake ..
make
./inference_tensorrt
```

Before building, you may need to adapt the parameters inside the `main.cpp`, which are self-explanatory. To read the samples, you can use those added to the folder *test_data* or use your own. Be aware that using data of different nature from the Nuscenes dataset may cause worse performance. In that case, you should either train the algorithm with your own dataset or perform some augmentation to the Nuscenes dataset to make data more diverse and similar to yours. If `main.cpp` executed correctly, the program will output the inference time in milliseconds, and also display an image with the result, where the lidar pointcloud is overlapped.

The TensorRT library and NN handler are within the *libs* folder. Nothing should be changed here. However, TensorRT needs C++17 to run right now because it uses the `filesystem` library. If you need to downgrade to C++14, you'll have to remove the usage of `filesystem` and manage the file reading in other way.

