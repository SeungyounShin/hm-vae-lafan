from fnmatch import translate
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

from fk_layer_lafan import *
from lafan_extract import *
from pytorch3d.transforms import *

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
    
    thetas, vid_names = read_single_sequence(folder, dest_folder, fps)

def read_single_sequence(folder, dest_folder, fps=None):
    actions = os.listdir(folder)
    # subjects = salsa_id # subject ids for sub-folder

    # skeleton define
    lafan_skeleton = Skeleton(
        offsets = sk_offsets,
        parents = sk_parents)
    lafan_skeleton.remove_joints(sk_joints_to_remove)   

    thetas = []
    vid_names = []

    for action in actions:
        fname = osp.join(folder, action)
        
        if fname.endswith('shape.npz'):
            continue
        
        data = read_bvh(fname)
        pose_axis_angle  = data.quats
        translation = torch.tensor(data.pos[:,0,:])        # [T x 3]
        pose_quat_angle = axis_angle_to_quaternion(torch.tensor(pose_axis_angle.copy())) # [T x 22 x 4]
        rot_list = quaternion_to_matrix(pose_quat_angle)    # [T x 22 x 3 x 3]
        rot6d_list = matrix_to_rotation_6d(rot_list)        # [T x 22 x 6]
        T = int(rot6d_list.shape[0])

        # calculate gloabl coordinates  [T x 22 x 3] 
        coord_list = lafan_skeleton.forward_kinematics( 
            rotations = pose_quat_angle.unsqueeze(0).double(),
            root_positions = translation.unsqueeze(0)).contiguous()

            
        # ori_pose = data.pos[:, joints_to_use] # N X 72
        if T < 30: # Remove very short sequence 
            continue

        if fps is not None:
            sampling_freq = 30 // fps
        else:
            sampling_freq = 1

        # shape = np.repeat(data['betas'][:10][np.newaxis], T, axis=0) # N X 10
        
        vid_name = np.array([f'{action[:-4]}']*T)
        vid_names.append(vid_name)        

        # Calculate linear velocity
        minus_coord_data = coord_list[:-1, :, :] # (T-1) X 24 X 3
        minus_coord_data = torch.cat((coord_list[0, :, :][None, :, :], minus_coord_data), dim=0) # T X 24 X 3
        linear_v = coord_list - minus_coord_data # T X 24 X 3, the first timestep is zero, all root are zeros 

        # Calculate root translation
        minus_trans_data = translation[:-1, :] # (T-1) X 3
        minus_trans_data = torch.cat((translation[0, :][None, :], minus_trans_data), dim=0) # T X 3
        root_v = translation - minus_trans_data # T X 3, first timestep translation is zero

        theta = torch.cat((rot6d_list.view(T, -1), rot_list.view(T, -1), coord_list.view(T, -1), \
            linear_v.view(T, -1), linear_v.view(T, -1), root_v), dim=1) # T X n_dim
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