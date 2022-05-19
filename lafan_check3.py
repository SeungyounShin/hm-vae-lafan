import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import torch
from fk_layer import ForwardKinematicsLayer
from utils.lafan_extract import *

edges = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (12, 13), (11, 14), (14, 15), (15, 16), (16, 17), (11, 18), (18, 19), (19, 20), (20, 21)
]

ani = read_bvh("/home/dataset/ubisoft-laforge-animation-dataset/output/BVH/walk3_subject2.bvh")
coord = ani.pos # [T, 22, 3]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
print(coord[10])
ax.scatter(coord[10,:, 0], coord[10,:, 1], coord[10,:, 2])

"""
for e in edges:
    ax.plot(
        [coord[0,:, 0][e[0]], coord[0,:, 0][e[1]]],
        [coord[0,:, 1][e[0]], coord[0,:, 1][e[1]]],
        [coord[0,:, 2][e[0]], coord[0,:, 2][e[1]]])
"""


plt.savefig("./ch.png")
