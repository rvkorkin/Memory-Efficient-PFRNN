# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 18:08:45 2021

@author: RKorkin
"""

import torch.nn as nn
import torch
from pf_rnn import PFLSTM, PFGRU
from model_rnn import MainModel
from dataset import LocalizationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from ModelParams import ModelParams
from matplotlib import pyplot as plt
import pandas as pd

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_tracks = 1000

track_data = np.array(pd.read_csv('trajectories.csv', header=None))
track_len = track_data.shape[0] // num_tracks
track = np.zeros((num_tracks, track_len, track_data.shape[1]))
for i in range(num_tracks):
    track[i] = track_data[i*track_len:(i+1)*track_len]

rng = np.random.default_rng()
eval_numbers = rng.choice(num_tracks, size=num_tracks//10, replace=False)
train_numbers = np.array([i for i in range(num_tracks) if i not in eval_numbers])

train_data = track[train_numbers]
eval_data = track[eval_numbers]


world = np.loadtxt('environment.csv', delimiter=',')
world0 = world.copy()
world = torch.tensor((world - world.mean()) / world.std()).to(device).double()
world = torch.unsqueeze(world, 0)
world = torch.unsqueeze(world, 0)


train_dataset = LocalizationDataset(train_data)

eval_dataset = LocalizationDataset(eval_data)

params = ModelParams()


train_loader = DataLoader(train_dataset, batch_size=params.batch_size, pin_memory=True, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=params.batch_size, pin_memory=True, shuffle=False)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = MainModel().to(device).double()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 10
grad_clip = 3

for epoch in tqdm(range(epochs)):
    model.train()
    train_loss = []
    curr_loss1 = 0
    for iteration, data in enumerate(train_loader):
        measurement, location, motion = data

        measurement = measurement.to(device).double()
        
        location = location.to(device).double()
        motion = motion.to(device).double()

        optimizer.zero_grad()
        loss, _, particle_pred = model.step(torch.cat(measurement.size(0) * [world]), measurement, motion, location, params.bpdecay)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        curr_loss1 += loss.to('cpu').detach().numpy()
    train_loss.append(curr_loss1)
    
    model.eval()
    eval_loss = []
    curr_loss2 = 0
    for iteration, data in enumerate(eval_loader):
        measurement, location, motion = data

        measurement = measurement.to(device).double()
        
        location = location.to(device).double()
        motion = motion.to(device).double()

        with torch.no_grad():
            loss, _, particle_pred = model.step(torch.cat(measurement.size(0) * [world]), measurement, motion, location, params.bpdecay)
            curr_loss2 += loss.to('cpu').detach().numpy()
    eval_loss.append(curr_loss2)
    print('epoch, train/eval loss', epoch, curr_loss1, curr_loss2)

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle as rect
import numpy as np
world = np.loadtxt('environment.csv', delimiter=',')
world0 = world.copy()

plt.close('all')
fig, ax = plt.subplots()
s_v, s_h = world0.shape[0], world0.shape[1]

for i in range(world0.shape[0]):
    for j in range(world0.shape[1]):
        if world0[i, j] == 0.0:
            r = rect((i, j), 1, 1, facecolor='white')
            ax.add_patch(r)
        if world0[i, j] == 1.0:
            r = rect((i, j), 1, 1, facecolor='black')
            ax.add_patch(r)
        if world0[i, j] == 2.0:
            r= rect((i, j), 1, 1, facecolor='red')
            ax.add_patch(r)
plt.xlim([0, world.shape[1]])
plt.ylim([0, world.shape[0]])