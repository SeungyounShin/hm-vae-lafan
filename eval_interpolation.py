import torch
import os
import sys
import argparse
import shutil
import numpy as np 
import copy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

from torch.utils.tensorboard import SummaryWriter
from pytorch3d.transforms import *

from utils_motion_vae import get_train_loaders_all_data_seq, MotionSeqDataWithWindow
from utils_common import get_config, make_result_folders
from utils_common import write_loss, show3Dpose_animation

from trainer_motion_vae import Trainer

import torch.backends.cudnn as cudnn
# Enable auto-tuner to find the best algorithm to use for your hardware.
cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=str,
                    default='/home/hm-vae-lafan/configs/len64_no_aug_hm_vae.yaml',
                    help='configuration file for training and testing')
parser.add_argument('--output_path',
                    type=str,
                    default='./',
                    help="outputs path")
parser.add_argument('--test_batch_size',
                    type=int,
                    default=10) 
parser.add_argument('--multigpus',
                    action="store_true")
parser.add_argument("--resume",
                    action="store_true")
parser.add_argument('--test_model',
                    type=str,
                    default='/home/hm-vae-lafan/outputs/len64_no_aug_hm_vae/interpolation_ckpt/gen_00240000.pt',
                    help="trained model for evaluation")
opts = parser.parse_args()

# Load experiment setting
config = get_config(opts.config)
edges = [
                        (0, 1), (0, 5), (0, 9),
                        (1 ,2), (2, 3), (3, 4),
                        (5 ,6), (6, 7), (7, 8),
                        (9,10), (10,11),(11,12),
                        (12, 13), (11, 14), (14, 15), (15, 16), (16, 17), (11, 18), (18, 19), (19, 20), (20, 21)
                    ]

#max_iter = config['max_iter']
max_iter = 1

trainer = Trainer(config)
trainer.cuda()
trainer.model = trainer.model.cuda()
if opts.multigpus:
    ngpus = torch.cuda.device_count()
    config['gpus'] = ngpus
    print("Number of GPUs: %d" % ngpus)
    trainer.model = torch.nn.DataParallel(trainer.model, device_ids=range(ngpus))
else:
    print("Using a single GPU")
    config['gpus'] = 1

rot_npy_folder = "/home/dataset/lafan_for_hm_vae_fps30"
val_json_file = "/home/hm-vae-lafan/utils/data/for_all_data_motion_model/val_all_lafan_motion_data.json"
mean_std_path = "/home/hm-vae-lafan/utils/data/for_all_data_motion_model/all_amass_data_mean_std.npy"
test_dataset = MotionSeqDataWithWindow(rot_npy_folder, val_json_file, mean_std_path, config, \
    fps_aug_flag=config['fps_aug_flag'], random_root_rot_flag=config['random_root_rot_flag'])

test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]

output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = make_result_folders(output_directory)
shutil.copyfile(opts.config, os.path.join(output_directory, 'config.yaml'))

if opts.test_model:
    trainer.load_ckpt(opts.test_model)
trainer.model.eval()

print("Test len :: ", len(test_loader))

window_size = 40

quat_gt_list = list()
quat_pred_list = list()

for it, input_data in enumerate(test_loader):

    gt_rot_6d, gt_rot_mat, _, gt_seq, _, _, _ = input_data
    B = gt_seq.shape[0]
    ori_T = gt_seq.shape[1]

    # calculate gloabl quat gt
    rotmat_to_local_quat = matrix_to_quaternion(gt_rot_mat.view(B,ori_T,22,3,3)).view(B,-1,22,4) # [B*T x 22 x 4]
    fk_gt_pos, gt_global_quat = trainer.model.skeleton.forward_kinematics_with_rotation(
                                    rotations = rotmat_to_local_quat.double().cuda(),
                                    root_positions = gt_seq[:,:,:3].cuda())

    pos,quat = trainer.model.interpolate(input_data, window_size = window_size)
    # global pos  : [B x T x 22 x 3]
    # global quat : [B x T x 22 x 4]

    pred_global_quat = quat[:,10:10+window_size,:,:].cpu()
    gt_global_quat   = gt_global_quat[:,10:10+window_size,:,:].cpu()

    quat_gt_list.append(gt_global_quat)
    quat_pred_list.append(pred_global_quat)

     # Visulization
    """
    with torch.no_grad():
        gt_seq = gt_seq[:,:10+window_size+1,:]
        gt_seq = gt_seq.view(B, 10+window_size+1, 22, 3)

        gt_seq_for_vis = gt_seq[0, :, :, :].cpu() # T X 22 X 3
        mean_out_for_vis = pos[0, :, :, :].cpu() # T X 22 X 3

        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(projection='3d')
        pad = 100

        root_mean = gt_seq_for_vis[:,:].mean(dim=[-3,-2])

        def update(frame):
            ax.cla()
            sample_pos  = mean_out_for_vis[int(frame),:,:]
            gt_pos  = gt_seq_for_vis[int(frame),:,:]
            #given_indexs = [0,1,2,3,4,5,6,7,8,9,gt_seq_for_vis.shape[0]-1]
            given_indexs = [0,3,6,9]
            alphas = np.arange(0,len(given_indexs))/(len(given_indexs)-1)

            for e in edges:
                
                #given index
                for ii,givenIdx in enumerate(given_indexs):
                    if givenIdx > frame:
                        break
                    gt_pos  = gt_seq_for_vis[givenIdx,:,:]
                    ax.plot(
                    [gt_pos[:, 0][e[0]], gt_pos[:, 0][e[1]]],
                    [gt_pos[:, 2][e[0]], gt_pos[:, 2][e[1]]],
                    [gt_pos[:, 1][e[0]], gt_pos[:, 1][e[1]]], color='red',
                    linewidth=2.0, alpha=alphas[ii])
                        
                #target index
                gt_pos  = gt_seq_for_vis[gt_seq_for_vis.shape[0]-1,:,:]
                ax.plot(
                [gt_pos[:, 0][e[0]], gt_pos[:, 0][e[1]]],
                [gt_pos[:, 2][e[0]], gt_pos[:, 2][e[1]]],
                [gt_pos[:, 1][e[0]], gt_pos[:, 1][e[1]]], color='green',
                linewidth=2.0, alpha=1)
                
                if frame >= 10:
                    ax.plot(
                        [sample_pos[:, 0][e[0]], sample_pos[:, 0][e[1]]],
                        [sample_pos[:, 2][e[0]], sample_pos[:, 2][e[1]]],
                        [sample_pos[:, 1][e[0]], sample_pos[:, 1][e[1]]], color='royalblue',
                        linewidth=2.0, alpha=1)
                        
                        #ax.set_xlim([root_mean[0]-pad,root_mean[0]+pad])
                        #ax.set_ylim([root_mean[2]-pad,root_mean[2]+pad]) 
                        #ax.set_zlim([-200,+200]) 

                ax.set_xlim([ gt_seq_for_vis[:,:,0].min()-pad, gt_seq_for_vis[:,:,0].max()+pad])
                ax.set_ylim([ gt_seq_for_vis[:,:,2].min()-pad, gt_seq_for_vis[:,:,2].max()+pad]) 
                ax.set_zlim([-150,150]) 

        animm = FuncAnimation(fig, update, frames=[i for i in range(0,gt_seq_for_vis.shape[0],1)])
    """
    
quat_gt_list = torch.cat(quat_gt_list, dim=0)
quat_pred_list = torch.cat(quat_pred_list, dim=0)

l2q = (quat_pred_list - quat_gt_list)**2
l2q = torch.sqrt(l2q.sum([-1,-2])) #[Bs, T]
B_all , T_all = l2q.shape
l2q = l2q.sum(1).sum(0)
l2q = l2q/T_all/B_all

print(l2q)

print("==Finish Evaluation==")
