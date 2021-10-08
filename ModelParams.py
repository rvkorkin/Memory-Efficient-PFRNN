# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 18:27:46 2021

@author: RKorkin
"""
class ModelParams(object):
    def __init__(self):
        self.speed = 0.2
        self.theta_noise = 0.01
        self.speed_noise = 0.02
        self.sensor_noise = 0.1
        self.pNumber = 30
        self.hidden_dim = 64
        self.map_size = 10
        self.map_emb = 64
        self.measurement_emb = 32
        self.motion_emb = 32
        self.num_measurement = 5
        self.bp_length = 10
        self.bpdecay = 0.1
        self.h_weight = 0.1
        self.l1_weight = 0
        self.l2_weight = 1
        self.elbo_weight = 1
        self.batch_size = 32
        self.random_seed = 5