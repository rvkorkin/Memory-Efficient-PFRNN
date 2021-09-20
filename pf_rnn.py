# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:39:31 2021

@author: RKorkin
"""

import torch
from torch import nn
import numpy as np
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class BaseModel(nn.Module):

    def __init__(self, pNumber, input_size, hidden_size, ext_measurement, ext_motion, resamp_alpha):
        """
        :param pNumber: number of particles for a PF-RNN
        :param input_size: the size of input x_t
        :param hidden_size: the size of the hidden particle h_t^i
        :param ext_measurement: the size for o_t(x_t)
        :param ext_motion: the size for u_t(x_t)
        :param resamp_alpha: the control parameter \alpha for soft-resampling.
        We use the importance sampling with a proposal distribution q(i) = \alpha w_t^i + (1 - \alpha) (1 / K)
        """
        super(BaseModel, self).__init__()
        self.pNumber = pNumber
        self.input_size = input_size
        self.h_dim = hidden_size
        self.ext_measurement = ext_measurement
        self.ext_motion = ext_motion
        self.resamp_alpha = resamp_alpha

        self.measurement_extractor = nn.Sequential(
            nn.Linear(self.input_size, self.ext_measurement),
            nn.LeakyReLU()
        )
        self.motion_extractor = nn.Sequential(
            nn.Linear(self.input_size, self.ext_motion),
            nn.LeakyReLU()
        )

        self.fc_measurement = nn.Linear(self.ext_measurement + self.h_dim, 1)
        self.batch_norm = nn.BatchNorm1d(self.pNumber)

    def resampling(self, particles, prob):
        """
        The implementation of soft-resampling. We implement soft-resampling in a batch-manner.
        :param particles: \{(h_t^i, c_t^i)\}_{i=1}^K for PF-LSTM and \{h_t^i\}_{i=1}^K for PF-GRU.
                        each tensor has a shape: [pNumber * batch_size, h_dim]
        :param prob: weights for particles in the log space. Each tensor has a shape: [pNumber * batch_size, 1]
        :return: resampled particles and weights according to soft-resampling scheme.
        """
        resamp_prob = self.resamp_alpha * torch.exp(prob) + (1 - self.resamp_alpha) * 1 / self.pNumber
        resamp_prob = resamp_prob.view(self.pNumber, -1)
        indices = torch.multinomial(resamp_prob.transpose(0, 1), num_samples=self.pNumber, replacement=True)
        batch_size = indices.size(0)
        indices = indices.transpose(1, 0).contiguous()
        offset = torch.arange(batch_size).type(torch.LongTensor).unsqueeze(0)
        if torch.cuda.is_available():
            offset = offset.cuda()
        indices = offset + indices * batch_size
        flatten_indices = indices.view(-1, 1).squeeze()

        # PFLSTM
        #particles_new = (particles[0][flatten_indices], particles[1][flatten_indices])
        # PFGRU
        particles_new = particles[flatten_indices]

        prob_new = torch.exp(prob.view(-1, 1)[flatten_indices])
        prob_new = prob_new / (self.resamp_alpha * prob_new + (1 - self.resamp_alpha) / self.pNumber)
        prob_new = torch.log(prob_new).view(self.pNumber, -1, 1)
        prob_new = prob_new - torch.logsumexp(prob_new, dim=0, keepdim=True)
        prob_new = prob_new.view(-1, 1)

        return particles_new, prob_new

    def reparameterize(self, mu, var):
        """
        Reparameterization trick
        :param mu: mean
        :param var: variance
        :return: new samples from the Gaussian distribution
        """
        std = torch.nn.functional.softplus(var)
        eps = torch.FloatTensor(std.shape).normal_().to(device)

        return mu + eps * std


class PFLSTM(BaseModel):
    def __init__(self, pNumber, input_size, hidden_size, ext_measurement, ext_motion, resamp_alpha):
        super().__init__(pNumber, input_size, hidden_size, ext_measurement, ext_motion, resamp_alpha)

        self.fc_ih = nn.Linear(self.ext_motion, 5 * self.h_dim)
        self.fc_hh = nn.Linear(self.h_dim, 5 * self.h_dim)

    def forward(self, input_, hx):
        h0, c0, p0 = hx
        wh_b = self.fc_hh(h0)

        measurement = self.measurement_extractor(input_)
        motion = self.motion_extractor(input_)

        wi = self.fc_ih(motion)
        s = wh_b + wi
        f, i, o, mu, var = torch.split(s, split_size_or_sections=self.h_dim, dim=1)
        g_ = self.reparameterize(mu, var).view(self.pNumber, -1, self.h_dim).transpose(0, 1).contiguous()
        g = self.batch_norm(g_).transpose(
            0, 1).contiguous().view(-1, self.h_dim)
        c1 = torch.sigmoid(f) * c0 + torch.sigmoid(i) * nn.functional.leaky_relu(g)
        h1 = torch.sigmoid(o) * torch.tanh(c1)

        att = torch.cat((measurement, h1), dim=1)
        logpdf_measurement = self.fc_measurement(att)

        p1 = logpdf_measurement.view(self.pNumber, -1, 1) + \
            p0.view(self.pNumber, -1, 1)

        p1 = p1 - torch.logsumexp(p1, dim=0, keepdim=True)
        (h1, c1), p1 = self.resampling((h1, c1), p1)

        return h1, c1, p1


class PFGRU(BaseModel):
    def __init__(self, pNumber, input_size, hidden_size, ext_measurement, ext_motion, resamp_alpha):
        super().__init__(pNumber, input_size, hidden_size, ext_measurement, ext_motion, resamp_alpha)
        self.fc_z = nn.Linear(self.h_dim + self.ext_motion, self.h_dim)
        self.fc_r = nn.Linear(self.h_dim + self.ext_motion, self.h_dim)
        self.fc_n = nn.Linear(self.h_dim + self.ext_motion, self.h_dim * 2)

    def forward(self, input_, hx):
        h0, p0 = hx

        measurement = self.measurement_extractor(input_)
        motion = self.motion_extractor(input_)

        z = torch.sigmoid(self.fc_z(torch.cat((h0, motion), dim=1)))
        r = torch.sigmoid(self.fc_r(torch.cat((h0, motion), dim=1)))
        n = self.fc_n(torch.cat((r * h0, motion), dim=1))

        mu_n, var_n = torch.split(n, split_size_or_sections=self.h_dim, dim=1)
        n = self.reparameterize(mu_n, var_n)

        n = n.view(self.pNumber, -1, self.h_dim).transpose(0, 1).contiguous()
        n = self.batch_norm(n)
        n = n.transpose(0, 1).contiguous().view(-1, self.h_dim)
        n = nn.functional.leaky_relu(n)

        h1 = (1 - z) * n + z * h0

        att = torch.cat((h1, measurement), dim=1)
        logpdf_measurement = self.fc_measurement(att)

        p1 = logpdf_measurement + p0

        p1 = p1.view(self.pNumber, -1, 1)
        p1 = p1 - torch.logsumexp(p1, dim=0, keepdim=True)

        h1, p1 = self.resampling(h1, p1)

        return h1, p1