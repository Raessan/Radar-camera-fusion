import torch
import torch.nn.functional as F
import copy
import numpy as np
import time


def custom_loss(output, lidar, relative_depth_input, l2_lidar=False, alpha=1.0, beta=1.0, gamma=1.0):

    # Lidar loss
    lidar_loss, _ = depth_loss(output, lidar, l2_lidar)

    # Smoothness loss
    if beta>0:
        smoothness_loss = calculate_structural_loss(output, relative_depth_input)
    else:
        smoothness_loss = torch.tensor(0.0)

    if gamma>0:
        consistency_loss = calculate_consistency_loss(output, relative_depth_input)
    else:
        consistency_loss = torch.tensor(0.0)

    # Total loss
    total_loss = alpha * lidar_loss + beta * smoothness_loss + gamma * consistency_loss
    
    
    return total_loss, lidar_loss, smoothness_loss, consistency_loss

def compute_local_neighborhood_diffs(depth_map, kernel_size=3):
    """
    Compute the differences between each pixel and its local neighbors.
    
    Parameters:
    depth_map (torch.Tensor): 4D tensor of shape (batch_size, 1, height, width)
    kernel_size (int): Size of the local neighborhood to consider.
    
    Returns:
    torch.Tensor: Differences between each pixel and its local neighbors.
    """
    padding = kernel_size // 2
    depth_padded = F.pad(depth_map, (padding, padding, padding, padding), mode='replicate')
    diffs = []
    for i in range(kernel_size):
        for j in range(kernel_size):
            if i == padding and j == padding:
                continue  # Skip the center pixel
            diffs.append(depth_map - depth_padded[:, :, i:i+depth_map.shape[2], j:j+depth_map.shape[3]])
    return torch.stack(diffs, dim=1)

def calculate_structural_loss(output_depth_batch, input_depth_batch):
    """
    Compute the structural consistency loss for a batch of input relative depth maps and output absolute depth maps based on local neighborhoods.
    
    Parameters:
    input_depth_batch (torch.Tensor): 4D tensor of shape (batch_size, 1, height, width) representing the batch of input relative depth maps.
    output_depth_batch (torch.Tensor): 4D tensor of shape (batch_size, 1, height, width) representing the batch of output absolute depth maps.
    kernel_size (int): Size of the local neighborhood to consider.
    
    Returns:
    float: The average structural consistency loss over the batch.
    """
    kernel_size=3
    # Compute local neighborhood differences for input and output depth maps
    input_diffs = compute_local_neighborhood_diffs(input_depth_batch, kernel_size)
    output_diffs = compute_local_neighborhood_diffs(output_depth_batch, kernel_size)

    # Determine where the relative ordering is violated
    violation_mask = (input_diffs * output_diffs) < 0

    # Calculate the loss
    loss = torch.abs(output_diffs) * violation_mask.float()
    loss_sum = loss.sum(dim=[1, 2, 3, 4])  # Sum for each sample in the batch
    count = violation_mask.sum(dim=[1, 2, 3, 4])  # Count of violations for each sample in the batch

    # Calculate average loss per sample, avoid division by zero
    avg_loss = torch.where(count > 0, loss_sum / count.float(), torch.zeros_like(loss_sum))

    # Return the average loss over the batch
    return avg_loss.mean()

def calculate_consistency_loss(output, relative_depth_input):

    consistency_loss = torch.mean(torch.abs(torch.exp(output)-torch.exp(relative_depth_input)))
    return consistency_loss

def depth_loss(output, lidar_gt, l2_lidar = False):
    
    lidar_gt_masked = lidar_gt.clone()
    # Get the lidar GT mask
    lidar_mask = lidar_gt_masked != 0
    
    if l2_lidar:
        lidar_loss_map = torch.square(output*lidar_mask - lidar_gt_masked*lidar_mask)
    else:
        lidar_loss_map = torch.abs(output*lidar_mask - lidar_gt_masked*lidar_mask)

    lidar_loss = torch.sum(lidar_loss_map)/(torch.sum(lidar_mask))

    return lidar_loss, lidar_loss_map

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = (torch.rand((1,1,450,800))*6.0).to(device)
    lidar_gt = (torch.rand((1,1,450,800))*6.0).to(device)
    lidar_gt[lidar_gt<3.0] = 0
    lidar_gt[lidar_gt!=0] = lidar_gt[lidar_gt!=0]-3.0
    relative_depth_input = torch.rand((1,1,450,800)).to(device)

    loss = custom_loss(output, lidar_gt, relative_depth_input, l2_lidar=True, alpha=1.0, beta=1.0, gamma=1.0)
    print("Total loss: ", loss[0])
    print("Lidar loss: ", loss[1])
    print("Smoothness loss: ", loss[2])
    print("Consistency loss: ", loss[3])

    output = torch.Tensor([[1, 1, 1, 1], [1, 1, 1, 1]]).to(device)
    lidar = torch.Tensor([[0, 1, 2, 3], [4, 5, 51, 52]]).to(device)
    lidar[lidar>0] = 1/lidar[lidar>0]
    depth_loss, depth_loss_map = depth_loss(output, lidar, l2_lidar=False)
    print("depth loss: ", depth_loss)
    print("Depth loss map: ", depth_loss_map)