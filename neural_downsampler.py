# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 22:30:07 2022

@author: RKorkin
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy import interpolate
from scipy import stats


class SubSampler(nn.Module):
    def __init__(self, number, sub_number, debug=False, resamp_alpha=0.5, tau=1e-3):
        super(SubSampler, self).__init__()
        self.resamp_alpha = resamp_alpha
        self.tau = tau
        self.debug = debug
        self.number = number
        self.sub_number = sub_number
        torch.manual_seed(0)
        self.att = torch.randn(number, sub_number)
        self.y = torch.rand(sub_number, 2, requires_grad=True).double()
        self.v = torch.rand(sub_number, requires_grad=True).double()

    def init_centroids(self, x, w, N_output):
        np.random.seed(0)
        torch.manual_seed(0)
        indices = torch.topk(w, N_output)
        _, indices = torch.topk(w, self.sub_number)
        y = x[indices]
        return y

    def reset_params(self):
        torch.manual_seed(0)
        torch.nn.init.uniform_(self.y, a=0, b=1)

    def distances(self, x, y):
        D = torch.cdist(x, y, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
        return D

    def attention(self, d):
        return torch.softmax(-d**2 / self.tau, axis=1)

    def forward(self, x, w, y=None):
        x = x.double()
        if y is None:
            self.y = self.init_centroids(x, w, self.sub_number)
        dist = self.distances(x, self.y) # distances between all initial points x and "clusters" y
        self.att = self.attention(dist) # this is like attentian, which is actually softmax with dist^2 and tau (like temperature or size of clusters)
        self.att = self.att.permute(1, 0)
        if self.debug:
            print(self.att.shape, torch.unsqueeze(w, 1).shape, torch.matmul(self.att, torch.unsqueeze(w, 1)).shape)
        self.y = torch.matmul(self.att, (x * torch.unsqueeze(w, 1))) / torch.matmul(self.att, torch.unsqueeze(w, 1))
        #self.v = torch.matmul(self.att, torch.unsqueeze(w, 1)) / torch.sum(self.att, axis=1)
        self.v = self.att @ w
        return self.y, self.v


def data_loader(number):
    if number == -1:
        x = torch.tensor(np.array([[0.01, 0.09], [0.09, 0.01], [0.01, 0.01], [0.09, 0.09], 
                           [0.91, 0.99], [0.99, 0.91], [0.91, 0.91], [0.99, 0.99],
                           [0.41, 0.49], [0.49, 0.41], [0.41, 0.41], [0.49, 0.49],
                           [0.01, 0.99], [0.09, 0.91], [0.01, 0.91], [0.09, 0.99]]), requires_grad=True)
        w = torch.ones(16, requires_grad=True) / 16
        return x, w
        
    np.random.seed(0)
    z = np.random.uniform(1, number, size=100)
    z = (number * np.exp(z - z.mean()) / (np.exp(z - z.mean())).sum()).astype(int)
    z[-1] += (number - z.sum())
    z = z[z>0]
    X = np.zeros((0, 2))
    w = np.zeros((0))
    for i in range(len(z)):
        p = np.random.uniform(0, 1, size=2)
        centers = p + np.array([[0.04 * np.random.randn(), 0.04 * np.random.randn()] for j in range(z[i])])
        centers = np.abs(centers)
        centers[centers>1] = 0.99
        weight = np.random.uniform(size=z[i])
        X = np.vstack((X, centers))
        w = np.hstack((w, weight))
    w /= w.sum()
    indxs = torch.randperm(len(X))
    torch.manual_seed(0)
    X = X[indxs]
    return torch.tensor(X, requires_grad=True).double(), torch.tensor(w, requires_grad=True).double()

if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    N_input, N_output = 64, 8
    x, w = data_loader(N_input)
    
    
    model = SubSampler(N_input, N_output, debug=False).double()
    
    model.train()
    criterion = nn.L1Loss()
    epochs = 50
    result = np.zeros((epochs))
    y, v = model(x, w)
    for i in range(epochs):
        y_old = y
        y, v = model(x, w, y)
        loss = criterion(y_old, y)
        result[i] = loss.item()
        if loss.data.numpy() < 1e-3:
            break
    plt.close('all')
    plt.plot(result)
    plt.xticks(color = 'black', fontsize = 10)
    plt.yticks(color = 'black', fontsize = 10)
    plt.figure(figsize=(12, 4))
    z = plt.imshow(model.att.detach().numpy())
    plt.colorbar(z)
    plt.xlabel('original particle number')
    plt.ylabel('cluster number')
    plt.title('contribution of given particle to the given cluster')
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x[:, 0].detach().numpy(), x[:, 1].detach().numpy(), s=1000*w.detach().numpy(), color='blue')
    plt.xlabel('x coord')
    plt.ylabel('y coord')
    plt.title('particles on x-y plane with size proportional to weight')
    v = (model.att @ w)
    plt.scatter(y[:, 0].detach().numpy(), y[:, 1].detach().numpy(), s=1000*v.detach().numpy(), alpha=0.5, color='red')
    print(v == model.v)