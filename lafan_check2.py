import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import torch
from utils_common import vis_multiple_frames
from fk_layer import ForwardKinematicsLayer

edges = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (12, 13), (11, 14), (14, 15), (15, 16), (16, 17), (11, 18), (18, 19), (19, 20), (20, 21)
]

fk_layer = ForwardKinematicsLayer(device=torch.device("cpu"))


coord = fk_layer(torch.eye(3).repeat((22,1,1)).unsqueeze(0)).numpy() # bs X 22 X 3
coord = coord.squeeze()
print(coord[::3].shape)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(coord[:,::3], coord[:,1::3], coord[:,2::3])


for e in edges:
    ax.plot(
        [coord[:, 0][e[0]], coord[:, 0][e[1]]],
        [coord[:, 1][e[0]], coord[:, 1][e[1]]],
        [coord[:, 2][e[0]], coord[:, 2][e[1]]])


plt.savefig("./ch.png")

vis_multiple_frames(coord.reshape(-1,22,3)[100:100+100] , dest_img_path = "./ch.jpg")