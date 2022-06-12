from helper_ply import read_ply, write_ply
import numpy as np

def read_ply_data(path, with_rgb=True, with_label=True):
    data = read_ply(path)
    xyz = np.vstack((data['x'], data['y'], data['z'])).T
    if with_rgb and with_label:
        rgb = np.vstack((data['red'], data['green'], data['blue'])).T
        labels = data['class']
        return xyz.astype(np.float32), rgb.astype(np.uint8), labels.astype(np.uint8)
    elif with_rgb and not with_label:
        rgb = np.vstack((data['red'], data['green'], data['blue'])).T
        return xyz.astype(np.float32), rgb.astype(np.uint8)
    elif not with_rgb and with_label:
        labels = data['class']
        return xyz.astype(np.float32), labels.astype(np.uint8)
    elif not with_rgb and not with_label:
        return xyz.astype(np.float32)
