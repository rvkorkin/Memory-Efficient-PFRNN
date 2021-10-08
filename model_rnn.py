# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:15:05 2021

@author: RKorkin
"""

import torch.nn as nn
import torch
from pf_rnn import PFLSTM, PFGRU
import numpy as np
from ModelParams import ModelParams

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
np.random.seed(ModelParams().random_seed)
torch.random.manual_seed(ModelParams().random_seed)

def conv(in_channels, out_channels, kernel_size=3, stride=1, dropout=0.0):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout)
            ).double()

class MainModel(nn.Module):
    def __init__(self, params=ModelParams()):
        super(MainModel, self).__init__()
        self.pNumber = params.pNumber
        self.hidden_dim = params.hidden_dim
        self.map_size = params.map_size
        self.map_emb = params.map_emb
        self.measurement_emb = params.measurement_emb
        self.motion_emb = params.motion_emb
        self.total_emb = self.measurement_emb + self.motion_emb
        self.model = PFGRU
        self.num_measurement = params.num_measurement
        self.h_weight = params.h_weight
        self.bpdecay = params.bpdecay
        self.l1_weight = params.l1_weight
        self.l2_weight = params.l2_weight
        self.elbo_weight = params.elbo_weight
        self.dropout_rate = 0.5
        '''
        self.rnn = PFLSTM(self.pNumber, self.total_emb,1000
                self.hidden_dim, 32, 32, 0.5)
        '''
        self.rnn = PFGRU(self.pNumber, self.total_emb, self.hidden_dim, 32, 32, 0.5)

        self.hidden2label = nn.Linear(self.hidden_dim, 3)

        self.conv1 = conv(1, 16, kernel_size=5, stride=2, dropout=0.2)
        if self.map_size > 18:
            self.conv1_2 = conv(16, 16, kernel_size=5, stride=2,
                    dropout=0.2)
            self.conv2_2 = conv(32, 32, kernel_size=3, stride=2, dropout=0.2)
        self.conv2 = conv(16, 32, kernel_size=3, stride=1, dropout=0.2)
        self.conv3 = conv(32, 32, kernel_size=3, stride=1, dropout=0)
        fake_map = torch.zeros(1, 1, self.map_size, self.map_size)
        fake_out = self.encode(fake_map)
        out_dim = np.prod(fake_out.shape).astype(int)

        self.map_embedding = nn.Linear(out_dim, self.map_emb)
        self.map2measurement = nn.Linear(self.map_emb, self.measurement_emb)
        self.map2motion = nn.Linear(self.map_emb, self.motion_emb)
        self.measurement_embedding = nn.Linear(self.num_measurement, self.measurement_emb)
        self.motion_embedding = nn.Linear(3, self.motion_emb)
        self.hnn_dropout = nn.Dropout(self.dropout_rate)

        self.initialize = 'rand'
        self.bp_length = params.bp_length

    def encode(self, map_in):
        """
        Encode the map
        :param map_in: the input map
        :return: map embeddings after convs
        """
        out1 = self.conv1(map_in.double())
        out2 = self.conv2(out1)
        out = self.conv3(out2)
        return out

    def init_hidden(self, batch_size):
        initializer = torch.rand if self.initialize == 'rand' else torch.zeros

        '''
        # for  PFLSTM
        h0 = initializer(batch_size * self.pNumber, self.hidden_dim)
        c0 = initializer(batch_size * self.pNumber, self.hidden_dim)
        p0 = torch.ones(batch_size * self.pNumber, 1) * np.log(1 / self.pNumber)
        hidden = (h0, c0, p0)
        '''

        #for PFGRU
        h0 = initializer(batch_size * self.pNumber, self.hidden_dim)
        p0 = torch.ones(batch_size * self.pNumber, 1) * np.log(1 / self.pNumber)
        hidden = (h0, p0)

        def cudify_hidden(h):
            if isinstance(h, tuple):
                return tuple([cudify_hidden(h_) for h_ in h])
            else:
                return h.cuda()

        if torch.cuda.is_available():
            hidden = cudify_hidden(hidden)

        return hidden

    def detach_hidden(self, hidden):
        if isinstance(hidden, tuple):
            return tuple([h.detach() for h in hidden])
        else:
            return hidden.detach()

    def forward(self, map_in, measurement_in, motion_in):
        emb_map = self.encode(map_in)
        batch_size = emb_map.size(0)
        
        emb_map = emb_map.view(batch_size, -1)
        emb_map = torch.relu(self.map_embedding(emb_map))
        measurement_map = torch.relu(self.map2measurement(emb_map))
        motion_map = torch.relu(self.map2motion(emb_map))
        emb_measurement = torch.relu(self.measurement_embedding(measurement_in))
        emb_motion = torch.relu(self.motion_embedding(motion_in))
        measurement_input = emb_measurement * measurement_map.unsqueeze(1)
        motion_input = emb_motion * motion_map.unsqueeze(1)

        embedding = torch.cat((measurement_input, motion_input), dim=2)

        # repeat the input if using the PF-RNN
        embedding = embedding.repeat(self.pNumber, 1, 1)
        seq_len = embedding.size(1)
        hidden = self.init_hidden(batch_size)
        hidden_states = []
        probs = []

        for step in range(seq_len):
            hidden = self.rnn(embedding[:, step, :], hidden)
            hidden_states.append(hidden[0])
            probs.append(hidden[-1])

        hidden_states = torch.stack(hidden_states, dim=0)
        hidden_states = self.hnn_dropout(hidden_states)

        probs = torch.stack(probs, dim=0)
        prob_reshape = probs.view([seq_len, self.pNumber, -1, 1])
        out_reshape = hidden_states.view([seq_len, self.pNumber, -1, self.hidden_dim])
        y = out_reshape * torch.exp(prob_reshape)
        y = torch.sum(y, dim=1)
        y = self.hidden2label(y)
        pf_labels = self.hidden2label(hidden_states)

        y_out_xy = torch.sigmoid(y[:, :, :2])
        y_out_h = torch.sigmoid(y[:, :, 2:])
        y_out = torch.cat([y_out_xy, y_out_h], dim=2)

        pf_out_xy = torch.sigmoid(pf_labels[:, :, :2])
        pf_out_h = torch.sigmoid(pf_labels[:, :, 2:])
        pf_out = torch.cat([pf_out_xy, pf_out_h], dim=2)
        
        return y_out, pf_out

    def step(self, map_in, measurement_in, motion_in, gt_location, bpdecay):
        pred, particle_pred = self.forward(map_in, measurement_in, motion_in)

        gt_xy_normalized = gt_location[:, :, :2] / self.map_size
        gt_theta_normalized = gt_location[:, :, 2:] / (np.pi * 2)
        gt_normalized = torch.cat([gt_xy_normalized, gt_theta_normalized], dim=2)
 
        batch_size = pred.size(1)
        sl = pred.size(0)
        bpdecay_params = np.exp(bpdecay * np.arange(sl))
        bpdecay_params = bpdecay_params / np.sum(bpdecay_params)
        if torch.cuda.is_available():
            bpdecay_params = torch.FloatTensor(bpdecay_params).cuda()
        else:
            bpdecay_params = torch.FloatTensor(bpdecay_params)

        bpdecay_params = bpdecay_params.unsqueeze(0)
        bpdecay_params = bpdecay_params.unsqueeze(2)
        pred = pred.transpose(0, 1).contiguous()

        l2_pred_loss = torch.nn.functional.mse_loss(pred, gt_normalized, reduction='none') * bpdecay_params
        l1_pred_loss = torch.nn.functional.l1_loss(pred, gt_normalized, reduction='none') * bpdecay_params

        l2_xy_loss = torch.sum(l2_pred_loss[:, :, :2])
        l2_h_loss = torch.sum(l2_pred_loss[:, :, 2])
        l2_loss = l2_xy_loss + self.h_weight * l2_h_loss

        l1_xy_loss = torch.mean(l1_pred_loss[:, :, :2])
        l1_h_loss = torch.mean(l1_pred_loss[:, :, 2])
        l1_loss = 10*l1_xy_loss + self.h_weight * l1_h_loss

        pred_loss = self.l2_weight * l2_loss + self.l1_weight * l1_loss

        total_loss = pred_loss

        particle_pred = particle_pred.transpose(0, 1).contiguous()
        particle_gt = gt_normalized.repeat(self.pNumber, 1, 1)
        l2_particle_loss = torch.nn.functional.mse_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params
        l1_particle_loss = torch.nn.functional.l1_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params

        # p(y_t| \tau_{1:t}, x_{1:t}, \theta) is assumed to be a Gaussian with variance = 1.
        # other more complicated distributions could be used to improve the performance
        y_prob_l2 = torch.exp(-l2_particle_loss).view(self.pNumber, -1, sl, 3)
        l2_particle_loss = - y_prob_l2.mean(dim=0).log()

        y_prob_l1 = torch.exp(-l1_particle_loss).view(self.pNumber, -1, sl, 3)
        l1_particle_loss = - y_prob_l1.mean(dim=0).log()

        xy_l2_particle_loss = torch.mean(l2_particle_loss[:, :, :2])
        h_l2_particle_loss = torch.mean(l2_particle_loss[:, :, 2])
        l2_particle_loss = xy_l2_particle_loss + self.h_weight * h_l2_particle_loss

        xy_l1_particle_loss = torch.mean(l1_particle_loss[:, :, :2])
        h_l1_particle_loss = torch.mean(l1_particle_loss[:, :, 2])
        l1_particle_loss = 10 * xy_l1_particle_loss + self.h_weight * h_l1_particle_loss

        belief_loss = self.l2_weight * l2_particle_loss + self.l1_weight * l1_particle_loss
        total_loss = total_loss + self.elbo_weight * belief_loss

        loss_last = torch.nn.functional.mse_loss(pred[:, -1, :2] * self.map_size, gt_location[:, -1, :2])

        particle_pred = particle_pred.view(self.pNumber, batch_size, sl, 3)

        return total_loss, loss_last, particle_pred