# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:27:30 2021

@author: RKorkin
"""

from torch.utils.data.dataset import Dataset
import numpy as np
import torch
from ModelParams import ModelParams

np.random.seed(ModelParams().random_seed)

class LocalizationDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.seq_len = self.data['tracks'].shape[1]
        self.seq_num = self.data['tracks'].shape[0]
        self.env_map = self.data['map']
        self.map_mean = self.env_map.mean()
        self.map_std = self.env_map.std()
        self.samp_seq_len = None

    def __len__(self):
        return self.seq_num

    def set_samp_seq_len(self, seq_len):
        self.samp_seq_len = seq_len

    def __getitem__(self, index):
        seq_idx = index % self.seq_num
        env_map = self.env_map
        env_map = torch.FloatTensor(env_map).unsqueeze(0)
        env_map = (env_map - self.map_mean) / self.map_std
        track = self.data['tracks'][seq_idx]

        if self.samp_seq_len is not None and self.samp_seq_len != self.seq_len:
            start = np.random.randint(0, self.seq_len - self.samp_seq_len + 1)
            track = track[start:start + self.samp_seq_len]

        measurement = track[:, 6:]
        motion = track[:, 3:6]
        gt_location = track[:, :3]

        return (env_map, measurement, gt_location, motion)