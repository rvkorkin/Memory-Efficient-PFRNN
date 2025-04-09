

import torch
import torch.nn as nn
from ModelParams import ModelParams
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class PFGRUCell(nn.Module):
    def __init__(self, points, params=ModelParams()):
        super().__init__()
        self.measurement_num = params.obs_num
        self.width = params.width
        self.height = params.height
        self.num_particles = params.pNumber
        self.hidden_dim = params.hidden_dim
        self.ext_obs = params.emb_obs
        self.ext_act = params.emb_act
        self.resamp_alpha = 0.5
        self.points = points
        self.fc_z = nn.Linear(self.hidden_dim + self.ext_act + 1*self.measurement_num, self.hidden_dim)
        self.fc_r = nn.Linear(self.hidden_dim + self.ext_act + 1*self.measurement_num, self.hidden_dim)
        self.fc_n = nn.Linear(self.hidden_dim + self.ext_act + 1*self.measurement_num, self.hidden_dim * 2)

        self.hidden2label = nn.Sequential(
            nn.Linear(self.hidden_dim, 3),
            nn.Sigmoid()
            )

        self.fc_obs = nn.Sequential(
            nn.Linear(self.measurement_num, self.measurement_num),
            nn.LeakyReLU(0.1),
            nn.Linear(self.measurement_num, 1, bias=False)
            )

        self.batch_norm = nn.BatchNorm1d(self.num_particles)

    def resampling(self, particles, prob):
        resamp_prob = self.resamp_alpha * torch.exp(prob) + (1 - self.resamp_alpha) * 1 / self.num_particles
        resamp_prob = resamp_prob.view(self.num_particles, -1)
        indices = torch.multinomial(resamp_prob.transpose(0, 1), num_samples=self.num_particles, replacement=True)
        batch_size = indices.size(0)
        indices = indices.transpose(1, 0).contiguous()
        offset = torch.arange(batch_size).type(torch.LongTensor).unsqueeze(0)
        if torch.cuda.is_available():
            offset = offset.cuda()
        indices = offset + indices * batch_size
        flatten_indices = indices.view(-1, 1).squeeze()

        particles_new = particles[flatten_indices]

        prob_new = torch.exp(prob.view(-1, 1)[flatten_indices])
        prob_new = prob_new / (self.resamp_alpha * prob_new + (1 - self.resamp_alpha) / self.num_particles)
        prob_new = torch.log(prob_new).view(self.num_particles, -1, 1)
        prob_new = prob_new - torch.logsumexp(prob_new, dim=0, keepdim=True)
        prob_new = prob_new.view(-1, 1)

        return particles_new, prob_new

    def distance_to_sensors(self, state):
        pts = self.points.unsqueeze(0)
        distances = torch.cdist(state, self.points, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
        distances = distances.float().to(device)
        distances_sorted, _ = torch.topk(distances, self.measurement_num, axis=-1, largest=False)
        return distances_sorted

    def reparameterize(self, mu, var):
        """
        Reparameterization trick
        :param mu: mean
        :param var: variance
        :return: new samples from the Gaussian distribution
        """
        std = torch.nn.functional.softplus(var)
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.shape).normal_()
        else:
            eps = torch.FloatTensor(std.shape).normal_()

        return mu + eps * std

    def forward(self, emb_act, obs_raw, h0, p0, do_resamp=True):
        # find the distance to sensors
        # concat with measured distances
        act = torch.cat((emb_act, obs_raw.view(-1, self.measurement_num)), dim=1)
        #this is kalman filter analogue -- we move according to motions and make correction based on measurementions 
        z = torch.sigmoid(self.fc_z(torch.cat((h0, act), dim=1)))
        r = torch.sigmoid(self.fc_r(torch.cat((h0, act), dim=1)))
        n = self.fc_n(torch.cat((r * h0, act), dim=1))
        #this is standard GRUcell-like approach
        mu_n, var_n = torch.split(n, split_size_or_sections=self.hidden_dim, dim=1)
        n = self.reparameterize(mu_n, var_n)
        #reparametrization trick
        n = n.view(self.num_particles, -1, self.hidden_dim).transpose(0, 1).contiguous()
        n = self.batch_norm(n)
        n = n.transpose(0, 1).contiguous().view(-1, self.hidden_dim)
        n = nn.functional.leaky_relu(n)
        #as claimed in PFRNN paper with ref to original papers batchnorm + leakyrelu works better than tanh (at least works longer in terms os bptt)
        h1 = (1 - z) * n + z * h0
        #an updated hidden state
        # now we do correction on weights based on new measurements -- pure particle filter feature
        logpdf_obs = self.fc_obs(obs_raw.view(-1, self.measurement_num)**2)
        p1 = logpdf_obs + p0
        #update log weights
        p1 = p1.view(self.num_particles, -1, 1)
        p1 = p1 - torch.logsumexp(p1, dim=0, keepdim=True)
        if do_resamp:
            h1, p1 = self.resampling(h1, p1)
            #resampling
        return h1, p1