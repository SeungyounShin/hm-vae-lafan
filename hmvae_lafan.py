import copy
import numpy as np
import json 
import os 
import pickle as pkl 
import datetime 
import random 

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import lr_scheduler

import torchgeometry as tgm

import my_tools
from fk_layer import ForwardKinematicsLayer
from skeleton import SkeletonUnpool, SkeletonPool, SkeletonConv, find_neighbor, SkeletonLinear, get_edges

from torch.distributions import Categorical
import torch.distributions.multivariate_normal as dist_mn

from utils_common import show3Dpose_animation, show3Dpose_animation_multiple, show3Dpose_animation_with_mask
from torch.utils.tensorboard import SummaryWriter

class Encoder(nn.Module):
    def __init__(self, args, topology):
        super(Encoder, self).__init__()
        self.topologies = [topology]
      
        self.latent_d = args['latent_d']
        self.shallow_latent_d = args['shallow_latent_d']

        self.channel_base = [6] # 6D representation 
        self.timestep_list = [args['train_seq_len']]
    
        self.channel_list = []
        self.edge_num = [len(topology)]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.latent_enc_layers = nn.ModuleList() # Hierarchical latent vectors from different depth of layers 
        self.args = args
        self.convs = []

        kernel_size = args['kernel_size']
        padding = (kernel_size - 1) // 2
        bias = True

        for i in range(args['num_layers']):
            self.channel_base.append(self.channel_base[-1] * 2) # 6, 12, 24,(48, 96)
          
            if args['train_seq_len'] == 8:
                if i == 0 or i == args['num_layers'] - 1:
                    self.timestep_list.append(self.timestep_list[-1]) # 8, 8, 4, 2, 2
                else:
                    self.timestep_list.append(self.timestep_list[-1]//2)
            elif args['train_seq_len'] == 16:
                if i == 0: # For len = 16
                    self.timestep_list.append(self.timestep_list[-1]) # 16, 16, 8, 4, 2
                else:
                    self.timestep_list.append(self.timestep_list[-1]//2) 
            else:
                self.timestep_list.append(self.timestep_list[-1]//2) # 64, 32, 16, 8, 4
            # print("timestep list:{0}".format(self.timestep_list))

        for i in range(args['num_layers']):
            seq = []
            neighbor_list = find_neighbor(self.topologies[i], args['skeleton_dist']) # 24, 14, 9, 7
            in_channels = self.channel_base[i] * self.edge_num[i] # 6 * 24, 12 * 14, 24 * 9, 48 * 7,
            out_channels = self.channel_base[i+1] * self.edge_num[i] # 12 * 24, 24 * 14, 48 * 9,  96 * 7
         
            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels) # 6*24, 12*14, 24*9, 48*7, 96*7

            for _ in range(args['extra_conv']):
                seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
                                        joint_num=self.edge_num[i], kernel_size=kernel_size, stride=1,
                                        padding=padding, padding_mode=args['padding_mode'], bias=bias))
          
            if args['train_seq_len'] == 8:
                if i == 0 or i == args['num_layers'] - 1:
                    curr_stride = 1
                else:
                    curr_stride = 2
            elif args['train_seq_len'] == 16:
                if i == 0:
                    curr_stride = 1
                else:
                    curr_stride = 2 
            else:
                curr_stride = 2

            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=self.edge_num[i], kernel_size=kernel_size, stride=curr_stride,
                                    padding=padding, padding_mode=args['padding_mode'], bias=bias, add_offset=False,
                                    in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]))
            self.convs.append(seq[-1])
            last_pool = True if i == args['num_layers'] - 1 else False
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode=args['skeleton_pool'],
                                channels_per_edge=out_channels // len(neighbor_list), last_pool=last_pool)
            seq.append(pool)
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*seq))
    
            if i == 0:
                latent_encode_linear = nn.Linear(self.channel_base[i+1]*self.timestep_list[i+1], self.shallow_latent_d*2)
            else:
                latent_encode_linear = nn.Linear(self.channel_base[i+1]*self.timestep_list[i+1], self.latent_d*2)
            self.latent_enc_layers.append(latent_encode_linear)

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]))

    def forward(self, input, offset=None):
        # train_hier_level: 1, 2, 3, 4 (deep to shallow)
        z_vector_list = []
        for i, layer in enumerate(self.layers):
            # print("i:{0}".format(i))
            # print("layer:{0}".format(layer))
            # print("layer input:{0}".format(input.size()))
            input = layer(input)
            # print("layer output:{0}".format(input.size()))
         
            # latent: bs X (k_edges*d) X (T//2^n_layers)
            bs, _, compressed_t = input.size()
            # print("input shape[1]:{0}".format(input.shape[1]))
            # print("channel:{0}".format(self.channel_base[i+1]))
            k_edges = input.shape[1] // self.channel_base[i+1]
            # print("k_edges:{0}".format(k_edges))
            
            encoder_map_input = input.view(bs, k_edges, -1)
            # print("encoder_map_input:{0}".format(encoder_map_input.size()))

            curr_z_vector = self.latent_enc_layers[i](encoder_map_input)
            # print("curr_z_vector:{0}".format(curr_z_vector.size()))
            z_vector_list.append(curr_z_vector)
           
        return input, z_vector_list 
        # each z_vector is bs X k_edges X (2*latent_d)


if __name__=='__main__':
    enc = Encoder()