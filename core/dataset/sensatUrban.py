import os
import os.path
from core.utils.ply_tools import read_ply_data

import numpy as np
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize

from plyfile import PlyData, PlyElement
import pandas as pd

from helper_ply import read_ply


class sensatUrban:

    def __init__(self, root, split, voxel_size, num_points):
        super().__init__()
        self.root = root
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.files = []
        self.split = split
        if self.split == "train":
            fs = sorted(os.listdir(os.path.join(root, "train")))
            fs = [os.path.join(self.root, 'train', x) for x in fs]
        elif self.split == "val":
            fs = sorted(os.listdir(os.path.join(root, "val")))
            fs = [os.path.join(self.root, 'val', x) for x in fs]
        else:
            fs = sorted(os.listdir(os.path.join(root, "test")))
            fs = [os.path.join(self.root, 'test', x) for x in fs]

        self.files.extend(fs)

    def __getitem__(self, index):
        if self.split == "train":
            xyzs, rgbs, labels = read_ply_data(self.files[index], with_rgb=True, with_label=True)

            # 数据增强，随机旋转
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])
            xyzs = np.dot(xyzs, rot_mat) * scale_factor
        else:
            xyzs, rgbs = read_ply_data(self.files[index], with_rgb=True, with_label=False)
            labels = np.zeros(xyzs.shape[0]).astype(np.int16)
            labels_selected = labels

        voxels = np.round(xyzs / self.voxel_size).astype(np.int16)
        voxels -= voxels.min(0, keepdims=1)

        _, inds, inverse_map = sparse_quantize(voxels,
                                               return_index=True,
                                               return_inverse=True)

        if 'train' in self.split:
            if len(inds) > self.num_points:
                inds = np.random.choice(inds, self.num_points, replace=False)
            labels_selected = labels[inds].astype(np.int16)

        if 'val' in self.split:
            if len(inds) > self.num_points:
                inds = np.random.choice(inds, self.num_points, replace=False)
            labels_selected = labels[inds].astype(np.int16)

        rgbs = rgbs.astype(np.int16)
        xyz_rgb_selected = np.hstack([xyzs, rgbs])[inds].astype(np.float16)
        voxel_rgb_selected = np.hstack([voxels, rgbs])[inds]

        lidar_ST = SparseTensor(xyz_rgb_selected, voxel_rgb_selected)
        labels_ST = SparseTensor(labels_selected, voxel_rgb_selected)
        all_labels_ST = SparseTensor(labels, voxels)
        inverse_map = SparseTensor(inverse_map, voxels)

        return {
            'lidar': lidar_ST,
            'targets': labels_ST,
            'targets_mapped': all_labels_ST,
            'inverse_map': inverse_map,
            'file_name': self.files[index]
        }

    def __len__(self):
        return len(self.files)

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)
