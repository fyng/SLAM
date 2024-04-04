import matplotlib.pyplot as plt
import numpy as np
from SLAM.utils.load_data import get_lidar, get_encoder, get_imu
from SLAM.MapUtilsCython import MapUtils_fclad as maputils
from SLAM.MapUtils import MapUtils as maputils_py

class SLAM():
    def __init__(self, width=470, wheel_radius=127, enc_to_rev=360):
        # Physical car parameters (units in mm)
        self.width = width
        self.wheel_radius = wheel_radius
        self.enc_to_rev = enc_to_rev

        # map params (units in meters)
        self.mapsize = 30 # seems good enough from the current plots
        self.mapres = 0.05 

        self.data = {}
        self.pos = None 
        # map params
        self.map, self.map_meta = self._init_map()
        self.logodds_occ = 0.9
        self.logodds_free = -0.5
        self.logodds_range = 15


    def _init_map(self):
        # init MAP
        MAP = {}
        MAP['res'] = self.mapres
        MAP['xmin'] = -self.mapsize 
        MAP['ymin'] = -self.mapsize 
        MAP['xmax'] = self.mapsize
        MAP['ymax'] = self.mapsize  
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
        map = np.zeros((MAP['sizex'],MAP['sizey']))
        return map, MAP

    def _encoder_ticks_to_dist(self, ticks):
        radians = ticks / self.enc_to_rev * 2 * np.pi 
        return radians * self.wheel_radius
    
    def _x_meters_to_cells(self, x):
        return np.array([np.ceil((x - self.map_meta['xmin']) / self.map_meta['res']).astype(np.int16)-1])
    
    def _y_meters_to_cells(self, y):
        return np.array([np.ceil((y - self.map_meta['ymin']) / self.map_meta['res']).astype(np.int16)-1])

    def load_encoder(self, path):
        # load encoder data and convert to distance
        FL, FR, RL, RR, ts = get_encoder(path)
        FL = self._encoder_ticks_to_dist(FL)
        FR = self._encoder_ticks_to_dist(FR)
        RL = self._encoder_ticks_to_dist(RL)
        RR = self._encoder_ticks_to_dist(RR)
        self.data.update({'FL': FL, 'FR': FR, 'RL': RL, 'RR': RR, 't_encoder': ts})
    
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
        n_timesteps = len(self.data['t_encoder'])
        left = (self.data['FL'] + self.data['RL']) / 2
        right = (self.data['FR'] + self.data['RR']) / 2

        pos = np.zeros((n_timesteps, 4))
        x, y, theta = 0, 0, 0
        for i, t in enumerate(self.data['t_encoder']):
            theta += (right[i] - left[i]) / self.width
            x += (right[i] + left[i]) / 2 * np.cos(theta)
            y += (right[i] + left[i]) / 2 * np.sin(theta)
            pos[i] += np.array([t, theta, x, y])

        pos[:,-2:] /= 1000 # convert to meters. Probably build a unit treatment
        self.pos = np.array(pos)
        x = self._x_meters_to_cells(self.pos[:,-2])
        y = self._y_meters_to_cells(self.pos[:,-1])

        return x, y
    
    def update_map_occupancy(self, passthrough, obstables):
        '''
        Update occupancy grid map of log probabilities. 

        Params:
            passthrough: [2, P] array of (x, y) indices of free cells
            obstables: [2, P_obst] array of (x, y) indices of occupied cells
        '''
        self.map[passthrough[0,:], passthrough[1,:]] += self.logodds_free
        self.map[obstables[0,:], obstables[1,:]] += self.logodds_occ
        self.map = np.clip(self.map, -self.logodds_range, self.logodds_range)

    def map_lidar(self):
        '''
        Map lidar data: rays of (range, angle) 
        to grid occupancy (x, y) on the map
        '''
        # get lidar data
        lidar = self.data['lidar']
        n_timesteps = len(lidar)
        t_encoder = self.pos[:,0]
        for i in range(n_timesteps):
            # align lidar and encoder time
            i_aligned = np.abs(t_encoder - lidar[i]['t']).argmin()  
            t = t_encoder[i_aligned]
            theta_platform = self.pos[i_aligned, 1]
            x_platform = self.pos[i_aligned, -2]
            y_platform = self.pos[i_aligned, -1]

            ranges = lidar[i]['scan'].reshape(-1)
            angles = lidar[i]['angle'].reshape(-1) # in radians
            indValid = np.logical_and((ranges < 30),(ranges> 0.1))
            ranges = ranges[indValid]
            angles = angles[indValid]

            # xy position in platform frame
            xs = ranges*np.cos(angles+theta_platform) + x_platform
            ys = ranges*np.sin(angles+theta_platform) + y_platform

            # convert from meters to cells
            xis = self._x_meters_to_cells(xs)
            yis = self._y_meters_to_cells(ys)
            x = self._x_meters_to_cells(x_platform)[0]
            y = self._y_meters_to_cells(y_platform)[0]

            r2 = maputils.getMapCellsFromRay_fclad(
                x, y,
                xis.reshape(-1), yis.reshape(-1),
                self.map_meta['sizex']
            )

            # update map occupancy
            self.update_map_occupancy(passthrough=r2, obstables=np.concatenate((xis, yis), axis=0))

        return self.map

    



    
