import matplotlib.pyplot as plt
import numpy as np
from SLAM.utils.load_data import get_lidar, get_encoder, get_imu
from SLAM.MapUtilsCython import MapUtils_fclad as maputils
from SLAM.MapUtils import MapUtils as maputils_py
from tqdm import tqdm

class SLAM():
    def __init__(self, width=470, wheel_radius=127, enc_to_rev=360):
        self.data = {}
        self.pos = None 

        # Physical car parameters (units in mm)
        self.width = width
        self.wheel_radius = wheel_radius
        self.enc_to_rev = enc_to_rev

        # map params (units in meters)
        self.mapsize = 30 # seems good enough from the current plots
        self.mapres = 0.02 

        # map params
        self.map, self.map_params = self._init_map()
        self.logodds_occ = 0.9 
        self.logodds_free = -0.3 # 3 passthrough to wipe a hit
        self.logodds_range = 15

        # particle filter params
        # FIXME: update params!
        self.n_particles = 1  #recommended 30-100
        self.xy_noise = self.mapres # std = 1 grid 
        self.theta_noise = 0.5 * 2 * np.pi / 360 # std = 0.5 degree
        self.seeding_interval = 10 # prune and reseed particles every 10 timesteps

    def _init_map(self):
        map_params = {}
        map_params['res'] = self.mapres
        map_params['xmin'] = -self.mapsize 
        map_params['ymin'] = -self.mapsize 
        map_params['xmax'] = self.mapsize
        map_params['ymax'] = self.mapsize  
        map_params['sizex']  = int(np.ceil((map_params['xmax'] - map_params['xmin']) / map_params['res'] + 1)) #cells
        map_params['sizey']  = int(np.ceil((map_params['ymax'] - map_params['ymin']) / map_params['res'] + 1))
        map = np.zeros((map_params['sizex'],map_params['sizey']))
        return map, map_params
    
    def _init_particle_map(self):
        return np.zeros((self.n_particles, self.map_params['sizex'], self.map_params['sizey']))

    def _encoder_ticks_to_dist(self, ticks):
        radians = ticks / self.enc_to_rev * 2 * np.pi 
        return radians * self.wheel_radius
    
    def _x_meters_to_cells(self, x):
        return np.array([np.ceil((x - self.map_params['xmin']) / self.map_params['res']).astype(np.int16)-1])
    
    def _y_meters_to_cells(self, y):
        return np.array([np.ceil((y - self.map_params['ymin']) / self.map_params['res']).astype(np.int16)-1])

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

    def sync_timestamps(self):
        '''
        Align encoder timestamps with lidar timestamps
        '''
        # TODO: implement this for IMU data too

        # align encoder to lidar timepoint
        t_lidar = [data['t'] for data in self.data['lidar']]
        enc_lidar_idx = np.zeros(len(self.data['t_encoder']), dtype=int)
        for i,t in enumerate(self.data['t_encoder']):
            lidar_idx = np.abs(t_lidar - t).argmin()
            enc_lidar_idx[i] = lidar_idx
        self.data['idx_enc_to_lidar'] = enc_lidar_idx

    def _update_map_occupancy(self, map, passthrough, obstables):
        '''
        Update occupancy grid map of log probabilities. 

        Params:
            map: [sizex, sizey] array of occupancy grid map
            passthrough: [2, P] array of (x, y) indices of free cells
            obstables: [2, P_obst] array of (x, y) indices of occupied cells
        '''
        map[passthrough[0,:], passthrough[1,:]] += self.logodds_free
        map[obstables[0,:], obstables[1,:]] += self.logodds_occ
        return np.clip(map, -self.logodds_range, self.logodds_range)

    def map_lidar(self, pos, map, idx_lidar):
        '''
        Map lidar data: rays of (range, angle) to grid occupancy (x, y) on the map

        Params:
            pos: [n_particles, 3] array of (theta, x, y)
            map: [n_particles, sizex, sizey] array of occupancy grid map
            idx_lidar: index of corresponding lidar data in self.data['lidar']for the given pos 
        '''
        assert pos.shape[0] == map.shape[0]

        theta_platform = pos[:, 0]
        x_platform = pos[:, 1]
        y_platform = pos[:, 2]

        ranges = self.data['lidar'][idx_lidar]['scan'].reshape(-1)
        angles = self.data['lidar'][idx_lidar]['angle'].reshape(-1) # in radians
        # remove self reflection and too far away points
        indValid = np.logical_and((ranges < 30),(ranges> 0.1))
        ranges = ranges[indValid]
        angles = angles[indValid]

        for i in range(pos.shape[0]):
            # xy position in platform frame
            xs = ranges*np.cos(angles+theta_platform[i]) + x_platform[i]
            ys = ranges*np.sin(angles+theta_platform[i]) + y_platform[i]

            # convert from meters to cells
            xis = self._x_meters_to_cells(xs)
            yis = self._y_meters_to_cells(ys)
            x = self._x_meters_to_cells(x_platform[i])[0]
            y = self._y_meters_to_cells(y_platform[i])[0]

            r2 = maputils.getMapCellsFromRay_fclad(
                x, y,
                xis.reshape(-1), yis.reshape(-1),
                self.map_params['sizex']
            )

            # update map occupancy
            map[i] = self._update_map_occupancy(map[i], passthrough=r2, obstables=np.concatenate((xis, yis), axis=0))

        return map
    
    def resample_particle_from_map(self, map):
        '''
        Resample particles based on the consensus map. The basic idea that if the environment is static, known regions of the map should stay as obstacle/empty space. 

        1. If a region is not observed in the consensus map, mask it out from error computation. We cannot a priori say if the region is occupied or not.
        '''
        weights = np.zeros(self.n_particles)
        n_particles = map.shape[0]
        mask = (self.map != 0)

        for i in range(n_particles):
            # compute L2 norm between map and particle
            error = (map[i] - self.map)[mask]
            weights[i] = 1 / (np.linalg.norm(error) + 1e-10)

        weights /= np.sum(weights)
        resampled_indices = np.random.choice(np.arange(n_particles), size=n_particles, replace=True, p=weights)

        return resampled_indices

    def _sample_motion_noise(self, n_particles):
        '''
        Sample motion noise from a gaussian distribution
        '''
        # sample noise from a gaussian distribution
        x_noise = np.random.normal(0, self.xy_noise, n_particles)
        y_noise = np.random.normal(0, self.xy_noise, n_particles)
        rotation_noise = np.random.normal(0, self.theta_noise, n_particles)

        return np.stack((rotation_noise, x_noise, y_noise), axis=1)

    def dead_reckoning(self, pos, left_dist, right_dist):
        '''
        update position for each particle

        params:
            pos: [n_particles x n_dims], where n_dims[-3:] is [theta, x, y]
            left_distance: int, distance travelled by left wheel since last timestep
            right: int, distance travelled by right wheel since last timestep
        '''        
        d_pos = np.zeros_like(pos)
        d_pos[:,-3] += (right_dist - left_dist) / self.width # theta update
        d_pos[:,-2] += (right_dist + left_dist) / 2 * np.cos(pos[:,-3]) # x update
        d_pos[:,-1] += (right_dist + left_dist) / 2 * np.sin(pos[:,-3]) # y update
        d_pos /= 1000 # convert to meters. Probably build a unit treatment

        return pos + d_pos
    
    def map_localize(self):
        ''''
        Main driver function to localize the car.

        Loop over all timesteps, use particle filter to model motion and update the map.

        Operations are in meters. Grid discretization occurs at the final step.
        '''
        self.sync_timestamps() # sync encoder and lidar timestamps

        n_timesteps = len(self.data['idx_enc_to_lidar'])
        left = (self.data['FL'] + self.data['RL']) / 2 # in mm
        right = (self.data['FR'] + self.data['RR']) / 2 # in mm
        
        pos = np.zeros((n_timesteps+1, self.n_particles, 3)) # pos[t][p] = [x, y, theta]
        particle_maps = self._init_particle_map() # [n_particles, sizex, sizey] 
        for i, idx in enumerate(tqdm(self.data['idx_enc_to_lidar'])):
            i_pos = i + 1 # first timestep is a dummy
            if i == self.seeding_interval:
                # save out the first consensus map
                # no noise added so just take the first particle
                self.map = particle_maps[0]

            if i % self.seeding_interval == 0:
                resampled_indices = self.resample_particle_from_map(particle_maps)
                # update map with current best
                best_particple = np.bincount(resampled_indices).argmax()
                self.map = particle_maps[best_particple]
                # resample particles and their maps
                pos[i_pos] = pos[i_pos-1][resampled_indices]
                particle_maps = particle_maps[resampled_indices]
                # add noise
                pos[i_pos] += self._sample_motion_noise(self.n_particles)
            else:
                pos[i_pos] = pos[i_pos-1]
                
            d_pos = self.dead_reckoning(pos[i_pos], left[i], right[i])
            pos[i_pos] += d_pos

            # map lidar data to each particle
            particle_maps = self.map_lidar(pos[i_pos], particle_maps, idx)
            
        # final update: write to self.map
        best_particple = np.bincount(self.resample_particle_from_map(particle_maps)).argmax()
        self.map = particle_maps[best_particple]

        return self.map




    



    
