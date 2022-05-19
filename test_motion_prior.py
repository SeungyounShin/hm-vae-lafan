from sqlite3 import DatabaseError
import torch
import os
import sys
import argparse
import shutil
import numpy as np 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

from utils_common import get_config, make_result_folders
from utils_motion_vae import get_train_loaders_all_data_seq
from trainer_motion_vae import Trainer

import torch

if __name__=="__main__":
    ckpt_path = "/home/hm-vae-lafan/outputs/len64_no_aug_hm_vae/checkpoints/gen_00240000.pt"

    config = get_config("/home/hm-vae-lafan/configs/len64_no_aug_hm_vae.yaml")

    trainer = Trainer(config)
    trainer.load_ckpt(ckpt_path)
    trainer.cuda()
    trainer.model.cuda().eval()

    data_loaders = get_train_loaders_all_data_seq(config)

    train_loader = data_loaders[0]
    val_loader = data_loaders[1]
    test_loader = data_loaders[2]

    data = iter(test_loader).next()

    hp = dict()
    hp['random_root_rot_flag'] = False

    latent_dims = [trainer.model.shallow_latent_d,trainer.model.latent_d,trainer.model.latent_d,trainer.model.latent_d]
    skeleton_dims = [12, 7 ,7 ,7]
    sampled_z_list = list()
    for ld, sd in zip(latent_dims,skeleton_dims):
        sampled_z_list.append(torch.randn(8,sd,ld).cuda())

    sampled_out_cont6d, sampled_out_rotation_matrix, sampled_out_pose_pos, \
                sampled_out_root_v, _, _, _ = trainer.model._decode(sampled_z_list, torch.zeros(8,64,22).cuda()) 

    sampled_out_pose_pos = sampled_out_pose_pos.detach().cpu()
    print(sampled_out_pose_pos.shape) # [B x T x 22 x 3]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    pad = 200

    edges = [
        (0, 1), (0, 5), (0, 9),
        (1 ,2), (2, 3), (3, 4),
        (5 ,6), (6, 7), (7, 8),
        (9,10), (10,11),(11,12),
        (12, 13), (11, 14), (14, 15), (15, 16), (16, 17), (11, 18), (18, 19), (19, 20), (20, 21)
    ]

    root_mean = torch.zeros(3)

    def update(frame):
        ax.cla()
        sample_pos  = sampled_out_pose_pos[0,int(frame),:,:]
        for e in edges:
            ax.plot(
                [sample_pos[:, 0][e[0]], sample_pos[:, 0][e[1]]],
                [sample_pos[:, 2][e[0]], sample_pos[:, 2][e[1]]],
                [sample_pos[:, 1][e[0]], sample_pos[:, 1][e[1]]], color='royalblue',
                linewidth=2.0)
            
            ax.set_xlim([root_mean[0]-pad,root_mean[0]+pad])
            ax.set_ylim([root_mean[2]-pad,root_mean[2]+pad]) 
            ax.set_zlim([-200,+200]) 

    animm = FuncAnimation(fig, update, frames=[i for i in range(0,64,1)])

    animm.save('./sampled_result.gif', writer='imagemagick')
