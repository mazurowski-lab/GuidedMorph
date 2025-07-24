import os, glob
import torch
from torch.utils.data import Dataset
from .data_utils import pkload
import numpy as np

class JHUBrainDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        x, y = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)

class JHUBrainInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_seg, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)

class JHUBrainInferBreastDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        loaded = pkload(path)
        x, y = loaded[0], loaded[1]
        segs = loaded[2:]
        # segs: [x_seg1, y_seg1, x_seg2, y_seg2, ...]
        label_pairs = []
        for i in range(0, len(segs), 2):
            x_seg = segs[i][None, ...]
            y_seg = segs[i+1][None, ...]
            x_seg, y_seg = self.transforms([x_seg, y_seg])
            x_seg = np.ascontiguousarray(x_seg)
            y_seg = np.ascontiguousarray(y_seg)
            x_seg = torch.from_numpy(x_seg)
            y_seg = torch.from_numpy(y_seg)
            label_pairs.append((x_seg, y_seg))
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x, y, label_pairs

    def __len__(self):
        return len(self.paths)