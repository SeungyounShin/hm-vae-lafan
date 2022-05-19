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

from utils_motion_vae import get_train_loaders_all_data_seq
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
                    default='/home/hm-vae-lafan/outputs/len64_no_aug_hm_vae/checkpoints/prior_model.pt',
                    help="trained model for evaluation")
opts = parser.parse_args()

# Load experiment setting
config = get_config(opts.config)

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

data_loaders = get_train_loaders_all_data_seq(config)

train_loader = data_loaders[0]
val_loader = data_loaders[1]
test_loader = data_loaders[2]

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]

output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = make_result_folders(output_directory)
shutil.copyfile(opts.config, os.path.join(output_directory, 'config.yaml'))
train_writer = SummaryWriter(
    os.path.join(output_directory, "logs"))

iterations = 0

if opts.test_model:
    trainer.load_ckpt(opts.test_model)

# freeze decoder
for param in trainer.model.dec.parameters():
    param.requires_grad = False

max_iter_encoder, max_iter_decoder = 2500, 250000
epoch = 0

while True:
    epoch += 1
    train_dataset = train_loader 
    val_dataset = val_loader
    test_dataset = test_loader
    for it, input_data in enumerate(train_dataset):

        loss_all, loss_kl, loss_6d_rec, loss_rot_rec, loss_pose_rec, \
        loss_joint_pos_rec, loss_root_v_rec, loss_linear_v_rec, loss_angular_v_rec = \
        trainer.interpolation_encoder_update(input_data, config, iterations, opts.multigpus)
    
        if it % 50 == 0:
            print('%d::%d Training: Total loss: %.4f, KL loss: %.8f, Rec 6D loss: %.4f, Rec Rot loss: %.4f, Rec Pose loss: %.4f, \
                Rec joint pos loss: %.4f, Rec root v loss: %.4f, Rec linear v loss: %.4f, Rec angular v loss: %.4f' % \
                (epoch, iterations, loss_all, loss_kl, loss_6d_rec, loss_rot_rec, loss_pose_rec, loss_joint_pos_rec, \
                loss_root_v_rec, loss_linear_v_rec, loss_angular_v_rec))
        # torch.cuda.synchronize()
        
        # Check loss in validation set
        if (iterations + 1) % config['validation_iter'] == 0:
            with torch.no_grad():
                for val_it, val_input_data in enumerate(val_dataset):
                    if val_it >= 50:
                        break;
                    val_loss_all, val_loss_kl, val_loss_6d_rec, val_loss_rot_rec, val_loss_pose_rec, \
                    val_loss_joint_pos_rec, val_loss_root_v, val_loss_linear_v, val_loss_angular_v = trainer.gen_update(val_input_data, \
                                                        config, iterations, opts.multigpus, validation_flag=True)
                    print("*********************************************************************************************")
                    print('Val Total loss: %.4f, Val KL loss: %.8f, Val Rec 6D loss: %.4f, Val Rec Rot loss: %.4f, Val Rec Pose loss: %.4f, \
                    Val Rec joint pos loss: %.4f, Val Rec root v loss: %.4f, Val Rec linear v loss: %.4f, Val Rec angular v loss: %.4f' % \
                        (val_loss_all, val_loss_kl, val_loss_6d_rec, val_loss_rot_rec, val_loss_pose_rec, \
                        val_loss_joint_pos_rec, val_loss_root_v, val_loss_linear_v, val_loss_angular_v))

        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        """
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations, opts.multigpus)
            print('Saved model at iteration %d' % (iterations + 1))
        """
        iterations += 1

    if iterations > max_iter_encoder:
        break

print("==Phase 1 end==")

# freeze encoder
decoder_w = list()
for param in trainer.model.dec.parameters():
    decoder_w.append(param.clone().detach())
    param.requires_grad = True

for param in trainer.model.enc.parameters():
    param.requires_grad = False 

iterations = 0
epoch = 0
while True:
    epoch += 1
    train_dataset = train_loader 
    val_dataset = val_loader
    test_dataset = test_loader
    for it, input_data in enumerate(train_dataset):

        loss_all, loss_kl, loss_6d_rec, loss_rot_rec, loss_pose_rec, \
        loss_joint_pos_rec, loss_root_v_rec, loss_linear_v_rec, loss_angular_v_rec = \
            trainer.interpolation_decoder_update(input_data ,decoder_w,config, iterations, opts.multigpus)
    
        if it % 50 == 0:
            print('%d:%d Training: Total loss: %.4f, KL loss: %.8f, Rec 6D loss: %.4f, Rec Rot loss: %.4f, Rec Pose loss: %.4f, \
                Rec joint pos loss: %.4f, Rec root v loss: %.4f, Rec linear v loss: %.4f, Rec angular v loss: %.4f' % \
                (epoch,iterations, loss_all, loss_kl, loss_6d_rec, loss_rot_rec, loss_pose_rec, loss_joint_pos_rec, \
                loss_root_v_rec, loss_linear_v_rec, loss_angular_v_rec))
        # torch.cuda.synchronize()
        

        # Check loss in validation set
        if (iterations + 1) % config['validation_iter'] == 0:
            with torch.no_grad():
                for val_it, val_input_data in enumerate(val_dataset):
                    if val_it >= 50:
                        break;
                    val_loss_all, val_loss_kl, val_loss_6d_rec, val_loss_rot_rec, val_loss_pose_rec, \
                    val_loss_joint_pos_rec, val_loss_root_v, val_loss_linear_v, val_loss_angular_v = trainer.gen_update(val_input_data, \
                                                        config, iterations, opts.multigpus, validation_flag=True)
                    print("*********************************************************************************************")
                    print('Val Total loss: %.4f, Val KL loss: %.8f, Val Rec 6D loss: %.4f, Val Rec Rot loss: %.4f, Val Rec Pose loss: %.4f, \
                    Val Rec joint pos loss: %.4f, Val Rec root v loss: %.4f, Val Rec linear v loss: %.4f, Val Rec angular v loss: %.4f' % \
                        (val_loss_all, val_loss_kl, val_loss_6d_rec, val_loss_rot_rec, val_loss_pose_rec, \
                        val_loss_joint_pos_rec, val_loss_root_v, val_loss_linear_v, val_loss_angular_v))

        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save("/home/hm-vae-lafan/outputs/len64_no_aug_hm_vae/interpolation_ckpt", iterations, opts.multigpus)
            print('Saved model at iteration %d' % (iterations + 1))

        iterations += 1
    
     # Visulization
    if epoch%100 == 0:
        print(f"Test len :: {len(test_dataset)}")
        with torch.no_grad():
            for test_it, test_input_data in enumerate(test_dataset):
            
                if test_it >= opts.test_batch_size:
                    break;
                print(f"Test {test_it} saving results on {image_directory}")
                # Generate long sequences

                _, _, _, gt_seq, _, _, _ = test_input_data
                B, T, _ = gt_seq.shape
                gt_seq = gt_seq.view(B, T, 22, 3)

                _, mean_seq_out_pos, sampled_seq_out_pos, \
                    zero_seq_out_pos \
                    = trainer.gen_seq(test_input_data, config, iterations, interpolation =True)
                # T X bs X 24 X 3, T X bs X 24 X 3, T X bs X 24 X 3, T X bs X 24 X 3
                for bs_idx in range(0, 1): # test data loader set bs to 1
                    gt_seq_for_vis = gt_seq[bs_idx, :, :, :].cpu() # T X 22 X 3
                    mean_out_for_vis = mean_seq_out_pos[:, bs_idx, :, :].cpu() # T X 22 X 3
                    sampled_out_for_vis = sampled_seq_out_pos[:, bs_idx, :, :].cpu() # T X 24 X 3

                    fig = plt.figure(figsize=(15,15))
                    ax = fig.add_subplot(projection='3d')
                    pad = 200

                    edges = [
                        (0, 1), (0, 5), (0, 9),
                        (1 ,2), (2, 3), (3, 4),
                        (5 ,6), (6, 7), (7, 8),
                        (9,10), (10,11),(11,12),
                        (12, 13), (11, 14), (14, 15), (15, 16), (16, 17), (11, 18), (18, 19), (19, 20), (20, 21)
                    ]

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
                                    linewidth=2.0)
                            
                            #ax.set_xlim([root_mean[0]-pad,root_mean[0]+pad])
                            #ax.set_ylim([root_mean[2]-pad,root_mean[2]+pad]) 
                            #ax.set_zlim([-200,+200]) 

                            ax.set_xlim([ gt_seq_for_vis[:,:,0].min()-pad, gt_seq_for_vis[:,:,0].max()+pad])
                            ax.set_ylim([ gt_seq_for_vis[:,:,2].min()-pad, gt_seq_for_vis[:,:,2].max()+pad]) 
                            ax.set_zlim([-200,+200]) 


                    animm = FuncAnimation(fig, update, frames=[i for i in range(0,gt_seq_for_vis.shape[0],1)])

                    print(image_directory)
                    animm.save('./interpolation_result.gif', writer='imagemagick')

    if iterations >= max_iter_decoder:
        print("Finish Training")
        break

print("==Phase 2 end==")
print("==Finish Training==")
