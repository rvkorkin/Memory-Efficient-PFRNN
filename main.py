# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 18:08:45 2021

@author: RKorkin
"""

import torch
from model_rnn import MainModel
from dataset import LocalizationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from ModelParams import ModelParams
import pandas as pd
np.random.seed(ModelParams().random_seed)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_tracks = 10000

track_data = np.array(pd.read_csv('trajectories.csv', header=None))
world = np.loadtxt('environment.csv', delimiter=',')
all_data = dict()

track_len = track_data.shape[0] // num_tracks
track = np.zeros((num_tracks, track_len, track_data.shape[1]))
for i in range(num_tracks):
    track[i] = track_data[i*track_len:(i+1)*track_len]


eval_test_numbers = np.random.choice(num_tracks, size=num_tracks//5, replace=False)
eval_numbers = eval_test_numbers[:len(eval_test_numbers)//2]
test_numbers = eval_test_numbers[len(eval_test_numbers)//2:]
train_numbers = np.array([i for i in range(num_tracks) if i not in eval_test_numbers])

train_data = dict()
eval_data = dict()
test_data = dict()

train_data['tracks'] = track[train_numbers]
train_data['map'] = world

eval_data['tracks'] = track[eval_numbers]
eval_data['map'] = world

test_data['tracks'] = track[test_numbers]
test_data['map'] = world


train_dataset = LocalizationDataset(train_data)
eval_dataset = LocalizationDataset(eval_data)
test_dataset = LocalizationDataset(test_data)

params = ModelParams()

train_loader = DataLoader(train_dataset, batch_size=params.batch_size, pin_memory=True, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=params.batch_size, pin_memory=True, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, pin_memory=True, shuffle=False)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = MainModel().to(device).double()
optimizer = torch.optim.SGD(model.parameters(), lr=5e-3)

epochs = 1000
grad_clip = 3

best_loss = np.Inf

for epoch in tqdm(range(epochs)):
    model.train()
    train_loss = []
    curr_loss1 = 0
    for iteration, data in enumerate(train_loader):
        env_map, measurement, location, motion = data
        
        env_map = env_map.to(device).double()
        measurement = measurement.to(device).double()
        location = location.to(device).double()
        motion = motion.to(device).double()

        optimizer.zero_grad()
        loss, last_loss, particle_pred = model.step(env_map, measurement, motion, location, params.bpdecay)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        curr_loss1 += loss.to('cpu').detach().numpy() / len(train_numbers)
    train_loss.append(curr_loss1)

    model.eval()
    eval_loss = []
    curr_loss2 = 0
    for iteration, data in enumerate(eval_loader):
        env_map, measurement, location, motion = data

        env_map = env_map.to(device).double()
        measurement = measurement.to(device).double()
        location = location.to(device).double()
        motion = motion.to(device).double()

        with torch.no_grad():
            loss, last_loss, particle_pred = model.step(env_map, measurement, motion, location, params.bpdecay)
            curr_loss2 += loss.to('cpu').detach().numpy() / len(eval_numbers)
    eval_loss.append(curr_loss2)
    print('epoch, train/eval loss', epoch+1, curr_loss1, curr_loss2)
    if curr_loss2 < best_loss:
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        filepath='pfrnn_model'+str(ModelParams().random_seed)+'.ptm'
        torch.save(state, filepath)
        best_loss = curr_loss2

def load_checkpoint(filepath='pfrnn_model'+str(ModelParams().random_seed)+'.ptm'):
    checkpoint = torch.load(filepath)
    print('epoch: ', checkpoint['epoch'])
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model

model = load_checkpoint()

curr_loss3 = 0
pose_err = 0
coord = np.zeros((len(test_loader), params.pNumber+1, track_len, 2))
for iteration, data in enumerate(test_loader):
    env_map, measurement, location, motion = data

    env_map = env_map.to(device).double()
    measurement = measurement.to(device).double()
    location = location.to(device).double()
    motion = motion.to(device).double()

    with torch.no_grad():
        loss, last_loss, particle_pred = model.step(env_map, measurement, motion, location, params.bpdecay)

        currtest_num = 0
        curr_loss3 += loss / len(test_numbers)
        pose_err += last_loss / len(test_numbers)

    coord[iteration, 0, :, :] = location[0, :, :2].cpu().numpy()
    coord[iteration, 1:, :, :] = torch.squeeze(particle_pred, 1)[:, :, :2].cpu().numpy() * params.map_size

print('total MSE over trajectory', curr_loss3)
print('total last point error over trajectory', pose_err)

np.savetxt('results_seed_' + str(ModelParams().random_seed) + '.csv', coord.flatten(), delimiter=",")
np.savetxt('errors_seed_' + str(ModelParams().random_seed) + '.csv', np.vstack((curr_loss3.to('cpu').numpy(), pose_err.to('cpu').numpy())), delimiter=",")


# to open it: results = np.loadtxt('results_seed_' + str(random_seed) + '.csv', delimiter=',').reshape((len(test_loader), params.pNumber+1, track_len, 2))

'''

test_num = 10
space = world
import matplotlib as mpl
from matplotlib import pyplot as plt
plt.close('all')
for k in range(track_len):
    if k % 10 == 0:
        fig, ax = plt.subplots()
        for i in range(space.shape[0]):
            yy = space.shape[0] - i - 1
            for j in range(space.shape[1]):
                xx = j
                if space[i, j] == 1.0:
                    r = mpl.patches.Rectangle((xx, yy), 1, 1, facecolor='black')
                    ax.add_patch(r)
                if space[i, j] == 2.0:
                    r= mpl.patches.Rectangle((xx, yy), 1, 1, facecolor='red')
                    ax.add_patch(r)
        ax.scatter(coord[test_num][0:1, k, 0], coord[test_num][0:1, k, 1], s=50, color='green')
        ax.scatter(coord[test_num][1:, k, 0], coord[test_num][1:, k, 1], s=50, color='brown')
        ax.set_ylim(0, params.map_size)
        ax.set_xlim(0, params.map_size)
        plt.show()

'''