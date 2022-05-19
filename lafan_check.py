import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import torch
from utils_common import vis_multiple_frames

edges = [
    (0, 1), (1, 2), (2, 3), (3, 4), 
    (0, 5), (5, 6), (6, 7), (7, 8), 
    #(0, 9), (9, 10), (10, 11), (11, 12), 
    #(12, 13), (11, 14), (14, 15), (15, 16), (16, 17), (11, 18), (18, 19), (19, 20), (20, 21)
]

example_path = "/home/dataset/lafan_for_hm_vae_fps30/aiming1_subject1.npy"
loadn = np.load(example_path)

print(loadn.shape)

coord = loadn[:, 22*3 + 22*3*3 : 22*3 + 22*3*3 +22*3] # [T x 66]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(coord[0, 2::3], coord[0, 1::3], coord[0, ::3])

for e in edges:
    ax.plot(
        [coord[0, 2::3][e[0]], coord[0, 2::3][e[1]]],
        [coord[0, 1::3][e[0]], coord[0, 1::3][e[1]]],
        [coord[0,  ::3][e[0]], coord[0,  ::3][e[1]]])

plt.savefig("./ch.png")

vis_multiple_frames(coord.reshape(-1,22,3)[100:100+100] , dest_img_path = "./ch.jpg")