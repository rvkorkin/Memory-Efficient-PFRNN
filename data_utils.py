# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:44:44 2021

@author: RKorkin
"""

import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ModelParams import ModelParams
from dataset import LocalizationDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import matplotlib as mpl
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


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
        self.direction = np.random.normal(2 * np.pi)
        self.x, self.y = self.random_place()
        for i, line in enumerate(self.world_data):
            nb_y = self.height - i - 1
            for j, block in enumerate(line):
                if block:
                    nb_y = self.height - i - 1
                    self.blocks.append((j, nb_y))
                    if block == 2:
                        self.sensors.extend([[j+0.5, nb_y+0.5]])

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
            x, y = np.random.uniform(0, self.width), np.random.uniform(0, self.height)
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
        return [x + self.sensor_noise * np.random.uniform(-1, 1) for x in measurement]

    def move(self):
        direction = self.direction
        speed = self.speed
        speed += self.speed_noise * np.random.uniform(-1, 1)
        dtheta = self.theta_noise * np.random.normal(2 * np.pi)
        while True:
            dx = np.cos(direction + dtheta) * speed
            dy = np.sin(direction + dtheta) * speed
            if self.free(self.x + dx, self.y + dy):
                break
            direction = np.random.normal(2 * np.pi)
        self.x += dx
        self.y += dy
        self.dphi = direction - self.direction
        self.direction = direction
        return self.speed, self.dphi

def gen_track(track_len=100, measurement_num=5):
    world_data = np.loadtxt('environment.csv', delimiter=',')
    robot = Robot(world_data)
    track_ret = []

    for _ in range(track_len):
        old_x = robot.x
        old_y = robot.y
        old_dir = robot.direction
        robot.speed, robot.dphi = robot.move()
        sensor_data = robot.read_sensor(measurement_num)
        d_x = robot.x - old_x
        d_y = robot.y - old_y
        d_d = robot.direction - old_dir
        motion_data = [d_x, d_y, d_d]
        step_data = [robot.x, robot.y, robot.direction]
        step_data = step_data + motion_data + sensor_data
        track_ret.append(step_data)
    return np.array(track_ret), world_data

def gen_data(num_tracks=50000, track_len=50, measurement_num=5):
    data_tracks = {'tracks': []}

    for _ in tqdm(range(num_tracks)):
        track_data, world_data = gen_track(track_len, measurement_num)
        data_tracks['tracks'].append(track_data)

    data_tracks['map'] = np.array(world_data)
    tracks = np.array(data_tracks['tracks'])
    Matr = np.zeros((0, tracks[0].shape[1]))
    for i in tqdm(range(num_tracks)):
        Matr = np.vstack((Matr, tracks[i]))
    print('saving started')
    np.savetxt('trajectories.csv', Matr, delimiter=",")
    print('saving done')
    np.savetxt('environment.csv', np.array(data_tracks['map']).astype(int), delimiter=",")
    data_tracks['tracks'] = np.array(data_tracks['tracks'])
    return data_tracks

if __name__ == "__main__":
    #np.random.seed(42)

    def draw(world, location):
        fig, ax = plt.subplots(figsize=(17, 7))
        for i in range(world.shape[0]):
            yy = world.shape[0] - i - 1
            for j in range(world.shape[1]):
                xx = j
                if world[i, j] == 1.0:
                    r = mpl.patches.Rectangle((xx, yy), 1, 1, facecolor='gray', alpha=0.5)
                    ax.add_patch(r)
                if world[i, j] == 2.0:
                    r = mpl.patches.Rectangle((xx, yy), 1, 1, facecolor='black', alpha=0.5)
                    ax.add_patch(r)
                    #el = mpl.patches.Ellipse((xx+0.5, yy+0.5), 0.2, 0.2, facecolor='black')
                    #ax.add_patch(el)
                    ax.scatter(xx+0.5, yy+0.5, s=1000, color='orange', marker='X')
        ax.scatter(location[:, 0], location[:, 1], color='red', alpha=0.5, label='true location')
        plt.xlim(0, world.shape[1])
        plt.ylim(0, world.shape[0])
        plt.legend(loc='lower left', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=18)
    plt.close('all')
    data_tracks = gen_data(num_tracks=50000)
    world = np.loadtxt('environment.csv', delimiter=',')

    dataset = LocalizationDataset(data_tracks)
    loader= DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False)
    world = data_tracks['map']
    for data in loader:
        _, measurement, location, motion = data
    measurement, location, motion = measurement.squeeze(0), location.squeeze(0), motion.squeeze(0)
    draw(world, location)