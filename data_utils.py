# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:44:44 2021

@author: RKorkin
"""

import random
import numpy as np
import pandas as pd
from ModelParams import ModelParams

class Robot():
    def __init__(self, world_data):
        super(Robot, self).__init__()
        self.speed_noise = ModelParams().speed_noise
        self.theta_noise = ModelParams().theta_noise
        self.sensor_noise = ModelParams().sensor_noise
        self.speed = ModelParams().speed
        self.world_data = world_data
        self.height = self.world_data.shape[0]
        self.width = self.world_data.shape[1]
        self.blocks = []
        self.sensors = []
        self.direction = random.uniform(-1, 1)
        self.x, self.y = self.random_place()
        self.old_x, self.old_y = self.x, self.y
        for y, line in enumerate(self.world_data):
            for x, block in enumerate(line):
                if block:
                    nb_y = self.height - y - 1
                    self.blocks.append((x, nb_y))
                    if block == 2:
                        self.sensors.extend(((x, nb_y), (x + 1, nb_y), (x, nb_y + 1), (x + 1, nb_y + 1)))

    def inside(self, x, y):
        if x < 0 or y < 0 or x > self.width or y > self.height:
            return False
        return True

    def free(self, x, y):
        if not self.inside(x, y):
            return False
        return self.world_data[self.height - int(y) - 1][int(x)] == 0

    def random_place(self):
        while True:
            x, y = random.uniform(0, self.width), random.uniform(0, self.height)
            if self.free(x, y):
                return x, y

    def distance(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def distance_to_sensors(self, x, y, measurement_num=5):
        distances = []
        for c_x, c_y in self.sensors:
            d = self.distance(c_x, c_y, x, y)
            distances.append(d)
        return sorted(distances)[:measurement_num]

    def read_sensor(self, measurement_num):
        measurement = self.distance_to_sensors(self.x, self.y, measurement_num)
        return [x + np.random.normal(scale=self.sensor_noise) for x in measurement]

    def move(self):
        while True:
            direction = self.direction
            speed = self.speed
            speed += self.speed_noise * np.random.normal(1)
            direction += self.theta_noise * np.random.normal(1)
            dx = np.sin(direction) * speed
            dy = np.cos(direction) * speed
            if self.free(self.x + dx, self.y + dy):
                self.x += dx
                self.y += dy
                break
            self.direction = np.random.normal(1)

def gen_track(track_len=40, measurement_num=5):
    world_data = np.loadtxt('environment.csv', delimiter=',')
    robot = Robot(world_data)
    track_ret = []

    for _ in range(track_len):
        sensor_data = robot.read_sensor(measurement_num)
        step_data = [robot.x, robot.y, robot.direction]
        old_x = robot.x
        old_y = robot.y
        old_dir = robot.direction
        robot.move()
        d_x = robot.x - old_x
        d_y = robot.y - old_y
        d_d = robot.direction - old_dir
        motion_data = [d_x, d_y, d_d]
        step_data = step_data + motion_data + sensor_data
        track_ret.append(step_data)

    return np.array(track_ret), world_data

def gen_data(num_tracks=100, track_len=40, measurement_num=5):
    data = {'tracks': []}

    from tqdm import tqdm
    for _ in tqdm(range(num_tracks)):
        track_data, world_data = gen_track(track_len, measurement_num)
        data['tracks'].append(track_data)

    data['map'] = np.array(world_data)
    tracks = np.array(data['tracks'])
    Matr = np.zeros((0, tracks[0].shape[1]))
    for i in range(num_tracks):
        Matr = np.vstack((Matr, tracks[i]))
    np.savetxt('trajectories.csv', Matr, delimiter=",")
    np.savetxt('environment.csv', np.array(data['map']).astype(int), delimiter=",")
    data['tracks'] = np.array(data['tracks'])
    return data
