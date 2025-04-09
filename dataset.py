# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:27:30 2021

@author: RKorkin
"""
import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from scipy.ndimage import gaussian_filter


class LocalizationDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.seq_len = len(self.data['tracks'][0])
        self.seq_num = len(self.data['tracks'])

        self.samp_seq_len = None


    def __len__(self):
        return self.seq_num

    def set_samp_seq_len(self, seq_len):
        self.samp_seq_len = seq_len

    def __getitem__(self, index):
        seq_idx = index % self.seq_num

        env_map = self.data['map']
        traj = self.data['tracks'][seq_idx]

        self.map_mean = np.mean(env_map)
        self.map_std = np.std(env_map)
        env_map = (env_map - self.map_mean) / self.map_std
        traj = torch.FloatTensor(traj)
        env_map = torch.FloatTensor(env_map).unsqueeze(0)

        if self.samp_seq_len is not None and self.samp_seq_len != self.seq_len:
            start = np.random.randint(0, self.seq_len - self.samp_seq_len + 1)
            traj = traj[start:start + self.samp_seq_len]

        obs = traj[:, 6:]
        action = traj[:, 3:6]
        gt_pos = traj[:, :3]

        return (env_map, obs, gt_pos, action)