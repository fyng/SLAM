import matplotlib.pyplot as plt
import numpy as np
from SLAM.utils.load_data import get_lidar, get_encoder, get_imu
from SLAM.utils.MapUtils_fclad import getMapCellsFromRay_fclad

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
    
    def map_lidar(self):
        # get lidar data
        lidar = self.data['lidar']
        n_timesteps = len(lidar['t'])
        map = np.zeros((n_timesteps, 2))
        for i in range(n_timesteps):
            ranges = lidar['scan'][i]
            angles = lidar['angle']
            indValid = np.logical_and((ranges < 30),(ranges> 0.1))
            ranges = ranges[indValid]
            angles = angles[indValid]

            # init MAP
            MAP = {}
            MAP['res']   = 0.05 #meters
            MAP['xmin']  = -20  #meters
            MAP['ymin']  = -20
            MAP['xmax']  =  20
            MAP['ymax']  =  20
            MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
            MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

            MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8)#DATA TYPE: char or int8

            # xy position in the sensor frame
            xs0 = np.array([ranges*np.cos(angles)])
            ys0 = np.array([ranges*np.sin(angles)])

            # convert position in the map frame here
            Y = np.concatenate([np.concatenate([xs0,ys0],axis=0),
                np.zeros(xs0.shape)],axis=0)
            ### HERE
            # convert from meters to cells
            xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
            yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

            r2 = getMapCellsFromRay_fclad(
                np.ceil(-MAP['xmin']/ MAP['res']) - 1,
                np.ceil(-MAP['ymin']/ MAP['res']) - 1,
                xis[0],yis[0], 400)



    
