import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

# Function that draws a sample
def draw_sample(sample, min_dist, max_dist, dot_size=5):
    
    def get_points_from_image(image):
        points = []
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] != 0:
                    y = i
                    x = j
                    z = image[i, j]
                    points.append([x, y, z])
        return np.array(points)
    
    
    sample_token, rel_depth, radar, lidar = sample
    rel_depth = np.squeeze(rel_depth)
    lidar = np.squeeze(lidar)
    
    # Process radar
    radar[0,:] *= rel_depth.shape[1]
    radar[1,:] *= rel_depth.shape[0]
    radar[2,:] = radar[2,:] * (max_dist - min_dist) + min_dist
    
    # Process lidar
    lidar_points = get_points_from_image(lidar)
    lidar_points[:,2] = lidar_points[:,2] * (max_dist - min_dist) + min_dist

    
    # Init axes.
    fig, ax = plt.subplots(1, 2, figsize=(18, 32))
    # Radar plot
    ax[0].set_title('Radar with token ' + str(sample_token))
    radar_picture = ax[0].scatter(radar[0,:], radar[1,:], c=radar[2,:], cmap='viridis', s=dot_size)
    fig.colorbar(radar_picture, ax=ax[0], fraction=0.026, pad=0.01)
    ax[0].imshow(rel_depth, alpha=0.4, cmap='viridis')
    ax[0].axis('off')
    #ax[0].figure.colorbar(rel_depth, ax=ax[0], label='Depth')
    
    
    # Lidar plot
    ax[1].set_title('Lidar with token ' + str(sample_token))
    lidar_picture = ax[1].scatter(lidar_points[:,0], lidar_points[:,1], c=lidar_points[:,2], cmap='viridis', s=dot_size)
    fig.colorbar(lidar_picture, ax=ax[1], fraction=0.026, pad=0.01)
    ax[1].imshow(rel_depth, alpha=0.4, cmap='viridis')
    ax[1].axis('off')
    

def draw_result(rel_depth, output, lidar_matrix, min_dist=1, max_dist=50, dot_size=5):

    n_samples = rel_depth.shape[0]

    # Init axes.
    fig, ax = plt.subplots(n_samples, 3, figsize=(9,16), squeeze=False)
    
    for k in range(n_samples):
        rel_depth_sample = rel_depth[k,0,:,:]
        output_sample = output[k,0,:,:]
        lidar_matrix_sample = lidar_matrix[k,0,:,:]

        # Process lidar
        lidar = []
        for i in range(lidar_matrix_sample.shape[0]):
            for j in range(lidar_matrix_sample.shape[1]):
                if lidar_matrix_sample[i, j] != 0:
                    y = i
                    x = j
                    z = lidar_matrix_sample[i, j]
                    lidar.append([x, y, z])
        lidar = np.array(lidar)

        #print("Max output: " + str(torch.max(output)))
        #print("Min output: " + str(torch.min(output)))
        output_sample_real = output_sample * (max_dist - min_dist) + min_dist
        lidar_depths_real = lidar[:,2] * (max_dist - min_dist) + min_dist


        # Output plot
        ax[k][0].set_title('Relative depth input')
        im0 = ax[k][0].imshow(rel_depth_sample, alpha=1.0, cmap='viridis')
        fig.colorbar(im0, ax=ax[k][0], fraction=0.026, pad=0.01)
        ax[k][0].axis('off')
        
        ax[k][1].set_title('Output')
        im1 = ax[k][1].imshow(output_sample_real, alpha=1.0, cmap='viridis')
        fig.colorbar(im1, ax=ax[k][1], fraction=0.026, pad=0.01)
        ax[k][1].axis('off')
        
        ax[k][2].set_title('Lidar GT')
        lidar_picture = ax[k][2].scatter(lidar[:,0], lidar[:,1], c=lidar_depths_real, cmap='viridis', s=dot_size)
        fig.colorbar(lidar_picture, ax=ax[k][2], fraction=0.026, pad=0.01)
        ax[k][2].imshow(output_sample_real, alpha=0.0)
        ax[k][2].axis('off')

    plt.show()

# Function that draws a sample
def draw_masked_depth(rel_depth, threshold = 0.95):
    rel_depth_colored = cv2.applyColorMap((np.squeeze(rel_depth)*255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
    rel_depth_masked = np.copy(rel_depth_colored)
    mask = (rel_depth>threshold)
    n_pixels_masked = np.sum(mask)
    mask = mask.reshape(rel_depth.shape[1], rel_depth.shape[2], 1)
    mask = np.repeat(mask, 3, axis=2)
    #rel_depth_masked = rel_depth_masked * mask
    rel_depth_masked[mask] = 255.0

    rel_depth_colored = cv2.cvtColor(rel_depth_colored, cv2.COLOR_BGR2RGB)
    rel_depth_masked = cv2.cvtColor(rel_depth_masked, cv2.COLOR_BGR2RGB)
    
    
    # Init axes.
    fig, ax = plt.subplots(1, 2, figsize=(18, 32))
    # Original depthmap plot
    ax[0].set_title('Original depthmap')
    ax[0].imshow(rel_depth_colored)
    ax[0].axis('off')
    
    # Masked depthmap
    ax[1].set_title('Masking ' + str(float(n_pixels_masked)/(rel_depth.shape[1]*rel_depth.shape[2])*100.0) + ' % of the image')
    ax[1].imshow(rel_depth_masked)
    ax[1].axis('off')
    
    plt.show()

def draw_error_map(error_maps, clip_error=0, dot_size=5, names = None):
    n_maps = len(error_maps)
    n_samples = error_maps[0].shape[0]

    # Init axes.
    fig, ax = plt.subplots(n_samples, n_maps, figsize=(9,16), squeeze=False)

    for i in range(n_samples):
        for j in range(n_maps):
            error_map = error_maps[j][i,0,:,:]
            if clip_error > 0:
                error_map = torch.clamp(error_map,0,clip_error)
            error_map_pc = []
            for k in range(error_map.shape[0]):
                for l in range(error_map.shape[1]):
                    if error_map[k, l] != 0:
                        y = k
                        x = l
                        z = error_map[k, l]
                        error_map_pc.append([x, y, z])
            error_map_pc = np.array(error_map_pc)
            
            # Output plot
            if names != None and len(names) == len(error_maps):
                ax[i][j].set_title(names[j])
            lidar_picture = ax[i][j].scatter(error_map_pc[:,0], error_map_pc[:,1], c=error_map_pc[:,2], cmap='viridis', s=dot_size)
            fig.colorbar(lidar_picture, ax=ax[i][j], fraction=0.026, pad=0.01)
            # ax[i][j].set_ylim(ax[i][j].get_ylim()[::-1])
            ax[i][j].imshow(error_map, alpha=0.0)
            ax[i][j].axis('off')

    plt.show()
