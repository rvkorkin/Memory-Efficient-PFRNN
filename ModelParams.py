# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 18:27:46 2021

@author: RKorkin
"""

class ModelParams(object):
    def __init__(self):
        self.pNumber = 30
        self.speed = 0.5
        self.speed_noise = 0.02
        self.theta_noise = 0.01
        self.sensor_noise = 0.1
        self.hidden_dim = 64
        self.map_size = 34
        self.width = 34
        self.height = 14
        self.emb_map = 64
        self.emb_obs = 32
        self.emb_act = 32
        self.obs_num = 5
        self.bp_length = 10
        self.bpdecay = 0.0
        self.h_weight = 0.1
        self.l1_weight = 0
        self.l2_weight = 1
        self.elbo_weight = 0
        self.batch_size = 300
        self.random_seed = 0
        self.dropout = 0.5
        self.measurement_noise_roughening = 0.1