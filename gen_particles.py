# -*- coding: utf-8 -*-
"""
Created on Mon May 23 22:44:18 2022

@author: RKorkin
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

                
N = 5000
pNumber = 50
emb_dim = 2

x = np.random.uniform(0, 1, size=N*pNumber*emb_dim).reshape(N, pNumber, emb_dim)
p = 0.1*(torch.randn(N*pNumber)**2).view(N, pNumber).double()
p = F.log_softmax(p, dim=-1)

x = torch.from_numpy(x).double()
data = torch.cat((x, p.view(N, pNumber, 1)), dim=2)
data = data.view(-1, emb_dim+1).numpy()

np.savetxt('particles_data.csv', data, delimiter=",")