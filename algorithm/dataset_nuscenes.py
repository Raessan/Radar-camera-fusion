import torch
from torch.utils.data import Dataset
import numpy as np
import os
import skimage.io as io
from skimage.transform import resize
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from tqdm import tqdm

# self.dir_data = 'D:/Datasets_and_code/datasets/nuscenes_depth/prepared_data' # Root to the folder with the prepared data
# DIR_NUSCENES = "D:/Datasets_and_code/datasets/nuscenes"
# VERSION = "v1.0-trainval"
# self.w_resized = 800 # If 0, the image is not resized in width
# self.h_resized = 450 # If 0, the image is not resized in height
# self.min_dist = 1 # Threshold minimum depth to discard points
# self.max_dist = 50 # Threshold maximum depth to discard points
# self.scale_factor_pc = 10 # This is a scale factor for radar's disparity to make learning easier. 
#                         # This factor has to be applied also during testing and production
# self.number_radar_points = 98 # Number of radar points to use. If the number of points in a sample is less than this, they are 
#                           # resampled randomly to create an array of fixed number of points. This shouldn't hurt the performance

class DepthDatasetNuscenes(Dataset):
    def __init__(self, mode, dir_data, w_resized, h_resized, min_dist, max_dist, number_radar_points, augment_dist=False, augment_dist_max_subtract=4.0, augment_dist_min_scale=0.8):
        self.mode = mode
        self.dir_data = dir_data
        self.w_resized = w_resized
        self.h_resized = h_resized
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.number_radar_points = number_radar_points
        self.augment_dist = augment_dist
        self.augment_dist_max_subtract = augment_dist_max_subtract
        self.augment_dist_min_scale = augment_dist_min_scale

        if self.mode == "train":
            self.sample_indices = torch.load(os.path.join(self.dir_data,'data_split.tar'))['train_sample_indices']
        elif self.mode == "val":
            self.sample_indices = torch.load(os.path.join(self.dir_data,'data_split.tar'))['val_sample_indices']
        elif self.mode == "test":
            self.sample_indices = torch.load(os.path.join(self.dir_data,'data_split.tar'))['test_sample_indices']
        elif self.mode == "trainval":
            train_sample_indices = torch.load(os.path.join(self.dir_data,'data_split.tar'))['train_sample_indices']
            val_sample_indices = torch.load(os.path.join(self.dir_data,'data_split.tar'))['val_sample_indices']
            self.sample_indices = train_sample_indices + val_sample_indices
        else:
             raise Exception("The mode is invalid. Possible values are train, val, test, trainval")
             
    def __len__(self):
        return len(self.sample_indices)
    
    def augment_pointcloud(self, points, subtract_value, scale_value):
        """
        Augment the radar/lidar points by subtracting a random value from the distances.
        """
        augmented_points = points.copy()
        augmented_points[2, :] = np.maximum(augmented_points[2, :] - subtract_value, 0)
        augmented_points[2, :] = augmented_points[2, :] * scale_value
        return augmented_points
    
    def __getitem__(self,idx):
        rel_depth = np.load(os.path.join(self.dir_data, "relative_depth", '%05d_rel_depth.npy' % self.sample_indices[idx]))
        matrix = np.load(os.path.join(self.dir_data, "cam_matrix", '%05d_cam_matrix.npz' % self.sample_indices[idx]))
        radar = np.load(os.path.join(self.dir_data, "radar", '%05d_radar_pc.npy' % self.sample_indices[idx]))
        lidar = np.load(os.path.join(self.dir_data, "lidar", '%05d_lidar_pc.npy' % self.sample_indices[idx]))

        h_original = rel_depth.shape[0]
        w_original = rel_depth.shape[1]

        self.w_nn = self.w_resized
        self.h_nn = self.h_resized

        # Update image
        if self.w_nn == 0:
            self.w_nn = w_original
        if self.h_nn == 0:
            self.h_nn = h_original
        if self.w_nn is not w_original or self.h_nn is not h_original:
            rel_depth = resize(rel_depth, (self.h_nn, self.w_nn), order=1, preserve_range=True, anti_aliasing=False)
        # Revert rel_depth so it represents depth instead of inverses
        rel_depth = 1-rel_depth
        
        # Update matrix
        K = matrix['K']
        scale_factor_w = self.w_nn/w_original
        scale_factor_h = self.h_nn/h_original
        K[0][0] *= scale_factor_w
        K[1][1] *= scale_factor_h
        K[0][2] *= scale_factor_w
        K[1][2] *= scale_factor_h

        # Boolean to know if it is radar or lidar
        is_radar = True
        radar_adapted = None
        lidar_adapted = None
        # Subtract value in case we perform augmentation
        subtract_value = np.random.uniform(0, self.augment_dist_max_subtract)
        scale_value = np.random.uniform(self.augment_dist_min_scale, 1.0)
        for pc in [radar, lidar]:
            # Grab the depths (camera frame z axis points away from the camera).
            depths = pc[2, :]
            # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
            points = view_points(pc, K, normalize=True)
            # We change the third dimension (which is 1 after renormalization) to the depth
            points[2, :] = depths

            # Augment distance values
            if self.augment_dist and self.mode == 'train':
                points = self.augment_pointcloud(points, subtract_value, scale_value)
                depths = points[2, :]

            # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
            # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
            # casing for non-keyframes which are slightly out of sync.
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > self.min_dist)
            mask = np.logical_and(mask, depths < self.max_dist)
            mask = np.logical_and(mask, points[0, :] > 1)
            mask = np.logical_and(mask, points[0, :] < self.w_nn - 1)
            mask = np.logical_and(mask, points[1, :] > 1)
            mask = np.logical_and(mask, points[1, :] < self.h_nn - 1)
            points = points[:, mask]

            # Clip values of points between min_dist and max_dist
            #points[2, :] = np.clip(points[2, :], self.min_dist, self.max_dist)
            
            # Finally, we normalize. We have to apply this normalization in test!
            points[2, :] = (points[2, :] - self.min_dist) / (self.max_dist - self.min_dist)
            
            
            # Append radar or lidar pointcloud
            if is_radar:
                # print(points)
                radar_adapted = points
                is_radar=False
            else:
                lidar_adapted = points
        
        # PREPARE RADAR POINTCLOUD
        
        # We normalize the radar points to 0-1
        radar_adapted[0,:] /= self.w_nn
        radar_adapted[1,:] /= self.h_nn
        
        # Now we resample the radar array to have fixed self.number_radar_points
        num_points = radar_adapted.shape[1]
        if num_points >= self.number_radar_points:
            # If the initial pointcloud has more points, select target_size points without replacement
            selected_indices = np.random.choice(num_points, size=self.number_radar_points, replace=False)
            radar_adapted = radar_adapted[:, selected_indices]
        else:
            # If the initial pointcloud has less points, randomly sample with replacement until filling target_size
            selected_indices = np.random.choice(num_points, size=self.number_radar_points - num_points, replace=True)
            radar_adapted = np.hstack([radar_adapted, radar_adapted[:, selected_indices]])
        
        # PREPARE LIDAR POINTCLOUD
        lidar_matrix = np.zeros((self.h_nn, self.w_nn))
        lidar_depths = lidar_adapted[2,:]

        # Get indices for matrix
        pixel_indices = (lidar_adapted[:2, :]+0.5).astype(int)
        lidar_matrix[pixel_indices[1, :], pixel_indices[0, :]] = lidar_depths
                
        # Return the IDX, the relative depth map, the radar PC and the lidar PC
        return self.sample_indices[idx], np.expand_dims(rel_depth, axis=0), radar_adapted, np.expand_dims(lidar_matrix, axis=0)