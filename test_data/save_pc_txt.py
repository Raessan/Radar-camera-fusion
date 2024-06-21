import numpy as np
import os

# Location of the original pointcloud
POINTCLOUD_DATASET_PATH = "/ssd/Datasets_and_code/nuscenes_depth_estimation/dataset/dataset_radar_cam/relative_depth/00000_rel_depth.npy"

# We steal the name of the file but create it as .txt
file_name_with_extension = os.path.basename(POINTCLOUD_DATASET_PATH)
pointcloud_test_path = os.path.splitext(file_name_with_extension)[0] + ".txt"

# We load the pointcloud from the original dataset folder
pc = np.load(POINTCLOUD_DATASET_PATH)

# We save the pointcloud in .txt format
np.savetxt(pointcloud_test_path, pc, delimiter= " ")