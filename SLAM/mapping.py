import matplotlib.pyplot as plt
import numpy as np
from SLAM.utils.load_data import get_lidar, get_encoder, get_imu
from SLAM.MapUtils.MapUtils import getMapCellsFromRay

class SLAM():
    def __init__(self, width=470, wheel_radius=127, enc_to_rev=360):
        # Physical car parameters (units in mm)
        self.width = width
        self.wheel_radius = wheel_radius
        self.enc_to_rev = enc_to_rev

        self.data = {}
        self.pos = None 
        self.map = None

    def _encoder_ticks_to_dist(self, ticks):
        radians = ticks / self.enc_to_rev * 2 * np.pi 
        return radians * self.wheel_radius

    def load_encoder(self, path):
        # load encoder data and convert to distance
        FL, FR, RL, RR, ts = get_encoder(path)
        FL = self._encoder_ticks_to_dist(FL)
        FR = self._encoder_ticks_to_dist(FR)
        RL = self._encoder_ticks_to_dist(RL)
        RR = self._encoder_ticks_to_dist(RR)
        self.data.update({'FL': FL, 'FR': FR, 'RL': RL, 'RR': RR, 'ts_encoder': ts})
    
    def load_lidar(self, path):
        # dictionary of lidar data
        # {'t': timestamp, 'scan': range, 'angle': angle}
        lidar = get_lidar(path)
        self.data.update({'lidar': lidar})

    def load_imu(self, path):
        acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, ts = get_imu(path)
        self.data.update({'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z, 'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z, 'ts_imu': ts})
    
    def dead_reckoning(self):
        # FIXME! numbers look wrong
        # splippage between (FL-RL), (FR-RR) < 0.1% across time course. Just take the front for now
        # [t, theta, x, y]
        n_timesteps = len(self.data['ts_encoder'])

        left = (self.data['FL'] + self.data['RL']) / 2
        right = (self.data['FR'] + self.data['RR']) / 2

        pos = np.zeros((n_timesteps, 4))
        x, y, theta = 0, 0, 0
        for i, t in enumerate(self.data['ts_encoder']):
            theta += (right[i] - left[i]) / self.width
            
            x += (right[i] + left[i]) / 2 * np.cos(theta)
            y += (right[i] + left[i]) / 2 * np.sin(theta)

            pos[i] += np.array([t, theta, x, y])
        self.pos = np.array(pos)

        return self.pos


    
