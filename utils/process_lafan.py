import sys
sys.path.append("../")

import os
import argparse
import numpy as np
import os.path as osp
import json 
from tqdm import tqdm
import torchgeometry as tgm
import torch 
import torch.nn as nn
import torch.nn.functional as F

from fk_layer import ForwardKinematicsLayer

from lafan_extract import *

dict_keys = ['betas', 'dmpls', 'gender', 'mocap_framerate', 'poses', 'trans']

# extract SMPL joints from SMPL-H model
joints_to_use = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 37
])
joints_to_use = np.arange(0, 156).reshape((-1, 3))[joints_to_use].reshape(-1)

def aa2matrot(pose):
    '''
    :param bs X 1 X num_joints X 3
    :return: pose_matrot: bs X num_joints X 3 X 3
    '''
    batch_size = pose.size(0)
    pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous().view(batch_size, 1, -1, 9)
    # bs X 1 X n_joints X 9
    pose_body_matrot = pose_body_matrot.view(batch_size, 1, -1, 3, 3) # bs X 1 X n_joints X 3 X 3
    pose_body_matrot = pose_body_matrot.squeeze(1) # bs X n_joints X 3 X 3
    return pose_body_matrot

def matrot2aa(pose_matrot):
    '''
    :param pose_matrot: bs x 1 x num_joints x 9/ bs X 1 X num_joints X 3 X 3
    :return: bs x 1 x num_joints x 3
    '''
    batch_size = pose_matrot.size(0)
    homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0, 1])
    pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(batch_size, 1, -1, 3).contiguous()
    return pose

def read_data(folder, dest_folder, fps=None):
    fk_layer = ForwardKinematicsLayer(device=torch.device("cpu"))
    
    thetas, vid_names = read_single_sequence(folder, fk_layer, dest_folder, fps)

def read_single_sequence(folder, fk_layer, dest_folder, fps=None):
    actions = os.listdir(folder)
    # subjects = salsa_id # subject ids for sub-folder

    thetas = []
    vid_names = []

    for action in actions:
        fname = osp.join(folder, action)
        
        if fname.endswith('shape.npz'):
            continue
    
        data = read_bvh(fname)
            
       #ori_pose = data.pos[:, joints_to_use] # N X 72
        T = data.pos.shape[0]
        ori_pose = data.quats.reshape(T, -1) # N X (T x 66)
        ori_translation = data.pos[:,0:1].squeeze() # N X 3

        if fps is not None:
            sampling_freq = 30 // fps
        else:
            sampling_freq = 1

        pose = ori_pose[0::sampling_freq, :] # N X 66
        translation = ori_translation[0::sampling_freq, :] # N X 3
        # pose = ori_pose[:, :]
        # translation = ori_translation[:, :]

        if pose.shape[0] < 30: # Remove very short sequence 
            continue

        #shape = np.repeat(data['betas'][:10][np.newaxis], T, axis=0) # N X 10
        
        vid_name = np.array([f'{action[:-4]}']*pose.shape[0])
        vid_names.append(vid_name)
    
        # axis-angle to rotation matrix 
        batch_size = 256 # In case sequence is too long to be computed directly 
        timesteps = pose.shape[0]
        rot_list = torch.zeros(timesteps, 22, 3, 3)
        rot6d_list = torch.zeros(timesteps, 22, 6)
        coord_list = torch.zeros(timesteps, 22, 3)
        
        pose = torch.from_numpy(pose).float() # N X 72
        translation = torch.from_numpy(translation).float() # N X 3
        for t_idx in range(0, timesteps, batch_size):
            aa_data = pose[t_idx:t_idx+batch_size, :66][:, None, :] # bs X 1 X 66
            aa_data = aa_data.view(-1, 1, 22, 3) # bs X 1 X 66 X 3
            rotMatrices = aa2matrot(aa_data) # bs X 66 X 3 X 3
            # Convert rotation matrix to 6D representation
            cont6DRep = torch.stack((rotMatrices[:, :, :, 0], rotMatrices[:, :, :, 1]), dim=-2) # bs X 24 X 2 X 3
            cont6DRep = cont6DRep.view(rotMatrices.size()[0], rotMatrices.size()[1], 6) # bs X 24 X 6
            # Convert rotation matrix to 3D joint coordinates representation
            coordRep = fk_layer(rotMatrices) # bs X 24 X 3

            rot_list[t_idx:t_idx+batch_size, :, :, :] = rotMatrices
            rot6d_list[t_idx:t_idx+batch_size, :, :] = cont6DRep
            coord_list[t_idx:t_idx+batch_size, :, :] = coordRep

        # Calculate linear velocity
        minus_coord_data = coord_list[:-1, :, :] # (T-1) X 24 X 3
        minus_coord_data = torch.cat((coord_list[0, :, :][None, :, :], minus_coord_data), dim=0) # T X 24 X 3
        linear_v = coord_list - minus_coord_data # T X 24 X 3, the first timestep is zero, all root are zeros 

        # Calculate root translation
        minus_trans_data = translation[:-1, :] # (T-1) X 3
        minus_trans_data = torch.cat((translation[0, :][None, :], minus_trans_data), dim=0) # T X 3
        root_v = translation - minus_trans_data # T X 3, first timestep translation is zero

        theta = torch.cat((rot6d_list.view(timesteps, -1), rot_list.view(timesteps, -1), coord_list.view(timesteps, -1), \
            linear_v.view(timesteps, -1), linear_v.view(timesteps, -1), root_v), dim=1) # T X n_dim
        # n_dim = 24*6(rot6d) + 24*3*3(rot matrix) + 24*3(joint coord) + 24*3(linear v) + 24*3(angular v, not used, use linear v) + 3
        # = 144 + 216 + 72 + 72 + 72 + 3 = 579

        theta = theta.data.cpu().numpy()
        thetas.append(theta)

        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        dest_npy_path = os.path.join(dest_folder, vid_name[0]+".npy")
        np.save(dest_npy_path, theta)
    
    return np.concatenate(thetas, axis=0), np.concatenate(vid_names, axis=0)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='/home/dataset/ubisoft-laforge-animation-dataset/output/BVH')
    # parser.add_argument('--dest-folder', type=str, help='processed data directory', default='/orion/u/jiamanli/datasets/amass_for_hm_vae')
    parser.add_argument('--dest-folder', type=str, help='processed data directory', default='/home/dataset/lafan_for_hm_vae_fps30')
    args = parser.parse_args()

    lafan_path = "/home/dataset/ubisoft-laforge-animation-dataset/output/BVH"

    lafan_bvhs_path = [lafan_path +"/"+ i for i in os.listdir(lafan_path)]



    #read_data(args.dir, all_sequences, args.dest_folder)
    read_data(args.dir, args.dest_folder, fps=30)