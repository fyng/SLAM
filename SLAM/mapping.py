import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from SLAM.utils.load_data import get_lidar, get_encoder, get_imu
from SLAM.utils.test_load_data import replay_lidar

class SLAM():
    def __init__(self, width=470, wheel_diameter=254, enc_to_rev=360):
        # Physical car parameters (units in mm)
        self.width = width
        self.wheel_diameter = wheel_diameter
        self.enc_to_rev = enc_to_rev

        self.data = {}
        self.pos = None 
        self.map = None

    def load_encoder(self, path):
        FL, FR, RL, RR, ts = get_encoder(path)
        self.data.update({'FL': FL, 'FR': FR, 'RL': RL, 'RR': RR, 'ts_encoder': ts})
    
    def load_lidar(self, path):
        lidar = get_lidar(path)
        self.data.update({'lidar': lidar})

    def load_imu(self, path):
        acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, ts = get_imu(path)
        self.data.update({'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z, 'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z, 'ts_imu': ts})

    def _encoder_ticks_to_dist(self, ticks):
        radians = ticks * 2 * np.pi / self.enc_to_rev
        return ticks * self.wheel_diameter / 2
    
    def dead_reckoning(self):
        # splippage between (FL-RL), (FR-RR) < 0.1% across time course. Just take the front for now
        # [t, theta, x, y]
        n_timesteps = len(self.data['ts_encoder'])
        pos = np.zeros((n_timesteps, 4))
        x, y, theta = 0, 0, 0
        for i, t in enumerate(self.data['ts_encoder']):
            FR = self.data['FR'][i]
            FL = self.data['FL'][i]

            theta += self._encoder_ticks_to_dist(FR - FL) / self.width
            
            dist = self._encoder_ticks_to_dist(FR + FL)
            x += dist * np.cos(theta)
            y += dist * np.sin(theta)

            pos[i] += np.array([t, theta, x, y])
        self.pos = np.array(pos)

        return self.pos


    
