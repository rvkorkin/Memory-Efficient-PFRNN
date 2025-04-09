# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:15:05 2021

@author: RKorkin
"""
import torch.nn as nn
import torch
from pf_rnn import PFGRUCell
import numpy as np
from ModelParams import ModelParams
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class MainModel(nn.Module):
    def __init__(self, points, params=ModelParams()):
        super(MainModel, self).__init__()
        self.num_particles = params.pNumber
        self.hidden_dim = params.hidden_dim
        self.height = params.height
        self.width = params.width
        self.map_emb = params.emb_map
        self.obs_emb = params.emb_obs
        self.act_emb = params.emb_act
        self.dropout_rate = params.dropout
        total_emb = self.obs_emb + self.act_emb
        self.num_obs = params.obs_num
        self.resamp_alpha = 0.5
        self.act_extractor = nn.Sequential(
            nn.Linear(self.obs_emb+self.act_emb, self.act_emb),
            nn.LeakyReLU()
        )
        self.h_weight = params.h_weight
        self.elbo_weight = params.elbo_weight
        self.points = points
        self.obs_embedding = nn.Linear(self.num_obs, self.obs_emb)
        self.act_embedding = nn.Linear(3, self.act_emb)
        self.rnn = PFGRUCell(self.points)

        self.hnn_dropout = nn.Dropout(self.dropout_rate)

        self.bp_length = params.bp_length
        self.l2_weight = params.l2_weight
        self.l1_weight = params.l1_weight

    def init_hidden(self, batch_size):
        h0 = torch.rand(batch_size * self.num_particles, self.hidden_dim)
        p0 = - np.log(self.num_particles) * torch.ones(batch_size * self.num_particles, 1)
        return h0.to(device), p0.to(device)


    def forward(self, obs_in, act_in):
        batch_size = obs_in.size(0)

        # repeat the input if using the PF-RNN
        seq_len = obs_in.size(1)
        h0, p0 = self.init_hidden(batch_size)
        act_emb = torch.relu(self.act_embedding(act_in))
        obs_emb = torch.relu(self.obs_embedding(obs_in))
        embedding = torch.cat((obs_emb, act_emb), dim=2)
        embedding = embedding.repeat(self.num_particles, 1, 1)
        emb_act = self.act_extractor(embedding)
        obs_raw = obs_in.repeat(self.num_particles, 1, 1)
        hidden_states = []
        probs = []
        do_resamp = True
        for step in range(seq_len):
            '''
            if (step+1) % 3:
                do_resamp = True
            else:
                do_resamp = False
            '''
            h0, p0 = self.rnn(emb_act[:, step], obs_raw[:, step], h0, p0)
            hidden_states.append(h0)
            probs.append(p0)

            # if step % self.bp_length == 0:
            #     hidden = self.detach_hidden(hidden)

        hidden_states = torch.stack(hidden_states, dim=0)

        probs = torch.stack(probs, dim=0)
        prob_reshape = probs.view([seq_len, self.num_particles, -1, 1])
        out_reshape = hidden_states.view([seq_len, self.num_particles, -1, self.hidden_dim])
        y = out_reshape * torch.exp(prob_reshape)

        y = y.sum(dim=1)

        y_out = self.rnn.hidden2label(y)

        pf_out = self.rnn.hidden2label(hidden_states)

        return y_out, pf_out

    def step(self, obs_in, act_in, gt_pos, bpdecay):

        pred, particle_pred = self.forward(obs_in, act_in)

        gt_x_normalized = gt_pos[:, :, :1] / self.width
        gt_y_normalized = gt_pos[:, :, 1:2] / self.height
        gt_theta_normalized = gt_pos[:, :, 2:] # within 2 * np.pi range
        gt_normalized = torch.cat([gt_x_normalized, gt_y_normalized, gt_theta_normalized], dim=2)

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

        l2_xy_loss = torch.sum(l2_pred_loss[:, :, :2])
        l2_h_loss1 = torch.nn.functional.mse_loss(torch.cos(2*np.pi*pred[:, :, 2]), torch.cos(gt_normalized[:, :, 2])) * bpdecay_params
        l2_h_loss2 = torch.nn.functional.mse_loss(torch.sin(2*np.pi*pred[:, :, 2]), torch.sin(gt_normalized[:, :, 2])) * bpdecay_params
        l2_h_loss = torch.sum(l2_h_loss1 + l2_h_loss2)
        l2_loss = l2_xy_loss + self.h_weight * l2_h_loss

        particle_pred = particle_pred.transpose(0, 1).contiguous()
        particle_gt = gt_normalized.repeat(self.num_particles, 1, 1)
        l2_particle_loss = torch.nn.functional.mse_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params
        y_prob_l2 = torch.exp(-l2_particle_loss).view(self.num_particles, -1, sl, 3)
        l2_particle_loss = - y_prob_l2.mean(dim=0).log()
        xy_l2_particle_loss = torch.mean(l2_particle_loss[:, :, :2])
        h_l2_particle_loss = torch.mean(l2_particle_loss[:, :, 2])
        l2_particle_loss = xy_l2_particle_loss + self.h_weight * h_l2_particle_loss
        pred_loss = self.l2_weight * l2_loss
        belief_loss = self.l2_weight * l2_particle_loss
        total_loss = pred_loss + self.elbo_weight * belief_loss

        loss_last = (torch.nn.functional.mse_loss(pred[:, -1, 0] * self.width, gt_pos[:, -1, 0]) + torch.nn.functional.mse_loss(pred[:, -1, 1] * self.height, gt_pos[:, -1, 1]))
        particle_pred = particle_pred.view(self.num_particles, batch_size, sl, 3)

        return total_loss, loss_last, pred, particle_pred