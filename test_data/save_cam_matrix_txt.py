import numpy as np
import os

# Location of the original cam matrix
CAM_MATRIX_DATASET_PATH = "/ssd/Datasets_and_code/nuscenes_depth_estimation/dataset/dataset_radar_cam/cam_matrix/00400_cam_matrix.npz"

# We steal the name of the file but create it as .txt
file_name_with_extension = os.path.basename(CAM_MATRIX_DATASET_PATH)
cam_matrix_test_path = os.path.splitext(file_name_with_extension)[0] + ".txt"

# We load the cam matrix from the original dataset folder
cam_matrix = np.load(CAM_MATRIX_DATASET_PATH)['K']

# We save the cam matrix in .txt format
np.savetxt(cam_matrix_test_path, cam_matrix, delimiter= " ")