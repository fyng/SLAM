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
        self.best_particle = None

        # Physical car parameters (units in mm)
        self.width = width
        self.wheel_radius = wheel_radius
        self.enc_to_rev = enc_to_rev

        # map params (units in meters)
        self.mapsize = 55 # seems good enough from the current plots
        self.mapres = 0.05 

        # map params
        self.map, self.map_params = self._init_map()
        self.logodds_occ = 1  
        self.logodds_free = -0.05 # 3 passthrough to wipe a hit
        self.logodds_range = 10

        # particle filter params
        self.n_particles = 100  #recommended 30-100
        self.xy_noise = 0  # std = 10mm
        self.theta_noise = 0.5 * (2 * np.pi / 360) # std = 1 degree
        self.seeding_interval = 40 # prune and reseed particles every 30 timesteps

        #lidar params
        self.lidar_minrange = 0.1
        self.lidar_maxrange = 30

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

    def _encoder_ticks_to_dist(self, ticks):
        radians = ticks / self.enc_to_rev * 2 * np.pi 
        return radians * self.wheel_radius # in mm
    
    def _x_meters_to_cells(self, x):
        return np.array([np.ceil((x - self.map_params['xmin']) / self.map_params['res']).astype(np.int16)-1])
    
    def _y_meters_to_cells(self, y):
        return np.array([np.ceil((y - self.map_params['ymin']) / self.map_params['res']).astype(np.int16)-1])

    def load_encoder(self, path):
        # load encoder data and convert to distance
        FR, FL, RR, RL, ts = get_encoder(path)
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
        # align encoder to lidar timepoint
        t_lidar = [data['t'] for data in self.data['lidar']]
        self.data['idx_enc_to_lidar'] = [np.abs(t_lidar - t).argmin() for t in self.data['t_encoder']]
        # self.data['idx_lidar_to_enc'] = [np.abs(self.data['t_encoder'] - t).argmin() for t in t_lidar]

    def _update_map_occupancy(self, map, passthrough, obstables):
        '''
        Update occupancy grid map of log probabilities. 

        Params:
            map: [sizex, sizey] array of occupancy grid map
            passthrough: [2, P] array of (x, y) indices of free cells
            obstables: [2, P_obst] array of (x, y) indices of occupied cells
        '''   
        map[passthrough[0,...], passthrough[1,...]] += self.logodds_free
        map[obstables[0,...], obstables[1,...]] += self.logodds_occ
        return np.clip(map, -self.logodds_range, self.logodds_range)

    # def map_lidar(self, pos, map, idx_lidar):
    #     '''
    #     Map lidar data: rays of (range, angle) to grid occupancy (x, y) on the map

    #     Params:
    #         pos: [n_particles, 3] array of (theta, x, y). IN MILLIMETERS
    #         map: [n_particles, sizex, sizey] array of occupancy grid map
    #         idx_lidar: index of corresponding lidar data in self.data['lidar']for the given pos 
    #     '''
    #     assert pos.shape[0] == map.shape[0]
    #     n_particles = pos.shape[0]
    #     theta_platform = pos[:, 0].reshape(-1, 1)
    #     x_platform = pos[:, 1].reshape(-1, 1) / 1000 # convert to meters
    #     y_platform = pos[:, 2].reshape(-1, 1) / 1000 # convert to meters
    #     ranges = self.data['lidar'][idx_lidar]['scan'].reshape(-1)
    #     angles = self.data['lidar'][idx_lidar]['angle'].reshape(-1) # in radians

    #     # remove self reflection and too far away points
    #     indValid = np.logical_and((ranges < self.lidar_maxrange),(ranges > self.lidar_minrange))
    #     ranges = np.tile(ranges[indValid], (n_particles, 1))
    #     angles = np.tile(angles[indValid], (n_particles, 1))

    #     xs = ranges*np.cos(angles+theta_platform) + x_platform
    #     ys = ranges*np.sin(angles+theta_platform) + y_platform
    #     x = self._x_meters_to_cells(x_platform)[0]
    #     y = self._y_meters_to_cells(y_platform)[0]
    #     xis = self._x_meters_to_cells(xs)[0]
    #     yis = self._y_meters_to_cells(ys)[0]

    #     for i in range(pos.shape[0]):
    #         r2 = maputils.getMapCellsFromRay_fclad(
    #             x[i], y[i],
    #             xis[i].reshape(-1), yis[i].reshape(-1),
    #             self.map_params['sizex']
    #         )
    #         # update map occupancy
    #         map[i] = self._update_map_occupancy(map[i], passthrough=r2, obstables=np.concatenate((xis[i].reshape(1,-1), yis[i].reshape(1,-1)), axis=0))

    #     return map
    
    # def resample_particle_from_map(self, map):
    #     '''
    #     Resample particles based on the consensus map. The basic idea that if the environment is static, known regions of the map should stay as obstacle/empty space. 

    #     1. If a region is not observed in the consensus map, mask it out from error computation. We cannot a priori say if the region is occupied or not.

    #     params:
    #         map: [n_particles, sizex, sizey] array of occupancy grid map
    #     '''
    #     n_particles = map.shape[0]
    #     mask = (np.abs(self.map) > 0.2) # mask out unknown and low confidence regions
    #     prev_map = self.map[mask].reshape(1,-1)
    #     particle_maps = map[:,mask]
    #     weights = np.corrcoef(np.concatenate((prev_map, particle_maps), axis=0))[0][1:]

    #     # for i in range(n_particles):
    #     #     # weigh by inverse of L2 distance
    #     #     weights[i] = np.corrcoef(map[i][mask].reshape(-1), self.map[mask].reshape(-1))

    #     weights /= np.sum(weights)
    #     resampled_indices = np.random.choice(np.arange(n_particles), size=n_particles, replace=True, p=weights)

    #     return resampled_indices
    
    def update_weights(self, pos, map, prev_weghts, idx_lidar):
        '''
        Given a set of particles and their position, calculate the lidar scan for each particle. 

        At each wall, extract the score from the previous map. Weight the particles by the average of the wall scores -> sigmoid (avoid negative)

        Params:
            pos: [n_particles, 3] array of (theta, x, y). IN MILLIMETERS
            map: [sizex, sizey] array of occupancy grid map
            idx_lidar: index of corresponding lidar data in self.data['lidar']for the given pos 

        Returns:
            score: [n_particles] array of scores for each particle
            best_particle: index of the best particle

        '''
        n_particles = pos.shape[0]
        theta_platform = pos[:, 0].reshape(-1, 1)
        x_platform = pos[:, 1].reshape(-1, 1) / 1000 # convert to meters
        y_platform = pos[:, 2].reshape(-1, 1) / 1000 # convert to meters
        ranges = self.data['lidar'][idx_lidar]['scan'].reshape(-1)
        angles = self.data['lidar'][idx_lidar]['angle'].reshape(-1) # in radians

        # remove self reflection and too far away points
        indValid = np.logical_and((ranges < self.lidar_maxrange),(ranges > self.lidar_minrange))
        ranges = np.tile(ranges[indValid], (n_particles, 1))
        angles = np.tile(angles[indValid], (n_particles, 1))

        xs = ranges*np.cos(angles+theta_platform) + x_platform
        ys = ranges*np.sin(angles+theta_platform) + y_platform
        x = self._x_meters_to_cells(x_platform)[0]
        y = self._y_meters_to_cells(y_platform)[0]
        xis = self._x_meters_to_cells(xs)[0]
        yis = self._y_meters_to_cells(ys)[0]

        weights = map[xis, yis] 
        weights = 1 / (1 + np.exp(np.mean(weights, axis=1))) * prev_weghts
        weights /= np.sum(weights)

        return weights


    def new_map_lidar(self, best_pos, map, idx_lidar):
        '''
        Map lidar data: rays of (range, angle) to grid occupancy (x, y) on the map

        Params:
            best_pos: len=3. array of (theta, x, y). IN MILLIMETERS
            map: [sizex, sizey] array of occupancy grid map
            idx_lidar: index of corresponding lidar data in self.data['lidar']for the given pos 
        '''
        theta_platform, x_platform, y_platform = best_pos
        x_platform /= 1000 # convert to meters
        y_platform /= 1000 # convert to meters
        ranges = self.data['lidar'][idx_lidar]['scan'].reshape(-1)
        angles = self.data['lidar'][idx_lidar]['angle'].reshape(-1) # in radians

        # remove self reflection and too far away points
        indValid = np.logical_and((ranges < self.lidar_maxrange),(ranges > self.lidar_minrange))
        ranges = ranges[indValid]
        angles = angles[indValid]
        xs = ranges*np.cos(angles+theta_platform) + x_platform
        ys = ranges*np.sin(angles+theta_platform) + y_platform

        x = self._x_meters_to_cells(x_platform)[0]
        y = self._y_meters_to_cells(y_platform)[0]
        xis = self._x_meters_to_cells(xs)[0]
        yis = self._y_meters_to_cells(ys)[0]

        r2 = maputils.getMapCellsFromRay_fclad(
            x, y,
            xis.reshape(-1), yis.reshape(-1),
            self.map_params['sizex']
        )

        # update map occupancy
        obstables = np.concatenate((xis.reshape(1, -1), yis.reshape(1,-1)), axis=0)
        map = self._update_map_occupancy(map, passthrough=r2, obstables=obstables)

        return map

    def _sample_motion_noise(self, n_particles):
        '''
        Sample motion noise from a gaussian distribution
        in MM
        '''
        noise = np.zeros((n_particles, 3))
        # sample noise from a gaussian distribution
        noise[:,0] = np.random.normal(0, self.theta_noise, n_particles)
        # try adding only rotation noise.
        # noise[:,1] = np.random.normal(0, self.xy_noise, n_particles)
        # noise[:,2] = np.random.normal(0, self.xy_noise, n_particles)
        return noise

    def dead_reckoning(self, pos, left_dist, right_dist):
        '''
        update position for each particle.

        in millimeters

        params:
            pos: [n_particles x n_dims], where n_dims[-3:] is [theta, x, y]
            left_distance: int, distance travelled by left wheel since last timestep
            right: int, distance travelled by right wheel since last timestep
        '''
        pos[:,0] += (right_dist - left_dist) / self.width # theta update
        pos[:,1] += (right_dist + left_dist) / 2 * np.cos(pos[:,0]) # x update
        pos[:,2] += (right_dist + left_dist) / 2 * np.sin(pos[:,0]) # y update

        return pos

    def map_localize(self):
        ''''
        Main driver function to localize the car.

        Loop over all timesteps, use particle filter to model motion and update the map.

        Operations are in meters. Grid discretization occurs at the final step.
        '''
        self.sync_timestamps() # sync encoder and lidar timestamps

        n_timesteps = len(self.data['idx_enc_to_lidar'])
        left = (self.data['FL'] + self.data['RL']) / 2 # in mm
        right = (self.data['FR'] + self.data['RR']) / 2
        
        pos = np.zeros((n_timesteps+1, self.n_particles, 3)) # pos[t][p] = [theta, x, y]
        weights = np.ones((self.n_particles))
        best_particple_idx = 0
        
        for t, idx in enumerate(tqdm(self.data['idx_enc_to_lidar'])):
            t_dummy = t + 1 # first timestep is a dummy
            pos[t_dummy] += pos[t_dummy-1]
            pos[t_dummy] = self.dead_reckoning(pos[t_dummy], left[t], right[t])
            
            if t_dummy > 1:           
                weights = self.update_weights(pos[t_dummy], self.map, weights, idx)
                # resample if insufficient effective particles
                n_effective = (np.sum(weights))**2 / np.sum(weights**2)
                print(n_effective)
                if n_effective < self.n_particles * 0.75:
                    resampled_idx = np.random.choice(np.arange(self.n_particles), size=self.n_particles, replace=True, p=weights)
                    best_particple_idx = np.argmax(weights)
                    # update particles
                    pos[t_dummy] = pos[t_dummy][resampled_idx,...]
                    pos[t_dummy] += self._sample_motion_noise(self.n_particles) 
                    weights = np.ones((self.n_particles))

            # pick the best particle
            self.map = self.new_map_lidar(pos[t_dummy][best_particple_idx], self.map, idx)
        self.pos = pos
        return self.map
    
    
    def get_pos(self, best=True):
        if self.best_particle and best:
            return self.pos[self.best_particle]
        
        return self.pos




    



    
