import matplotlib.pyplot as plt
import numpy as np
from SLAM.utils.load_data import get_lidar, get_encoder, get_imu
from SLAM.MapUtilsCython import MapUtils_fclad as maputils
from SLAM.MapUtils import MapUtils as maputils_py
from tqdm import tqdm

class SLAM():
    def __init__(
        self, 
        width: int = 470, 
        wheel_radius: int =127, 
        enc_to_rev: int =360,
        mapsize: int = 55,
        mapres: float = 0.05,
        n_particles: int = 300,
    ):
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
        self.logodds_free = -0.1 # 3 passthrough to wipe a hit
        self.logodds_range = 15

        # particle filter params
        self.n_particles = 100  #recommended 30-100
        self.xy_noise = 0  # std = 10mm
        self.theta_noise = 0.5 * (2 * np.pi / 360) # std = 1 degree
        self.theta_scale = 2 
        self.reseed_interval = 20

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
        # align encoder to lidar timepoint
        t_lidar = [data['t'] for data in self.data['lidar']]
        self.data['idx_enc_to_lidar'] = [np.abs(t_lidar - t).argmin() for t in self.data['t_encoder']]


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
        ranges = self.data['lidar'][idx_lidar]['scan'].reshape(-1,1)
        angles = self.data['lidar'][idx_lidar]['angle'].reshape(-1,1) # in radians
        # remove self reflection and too far away points
        indValid = np.logical_and((ranges < self.lidar_maxrange),(ranges > self.lidar_minrange))
        ranges = np.tile(ranges[indValid], (n_particles, 1))
        angles = np.tile(angles[indValid], (n_particles, 1))
        angles += theta_platform

        xs = ranges*np.cos(angles) + x_platform
        ys = ranges*np.sin(angles) + y_platform
        xis = self._x_meters_to_cells(xs)[0]
        yis = self._y_meters_to_cells(ys)[0]

        weights = 1 / (1 + np.exp(np.mean(map[xis, yis] , axis=1))) # sigmoid
        weights *= prev_weghts # update
        weights /= np.sum(weights) # re-normalize

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
        noise[:,1] = np.random.normal(0, self.xy_noise, n_particles)
        noise[:,2] = np.random.normal(0, self.xy_noise, n_particles)
        return noise

    def dead_reckoning(self, pos, left_dist, right_dist, noise=False):
        '''
        update position for each particle.

        in millimeters

        params:
            pos: [n_particles x n_dims], where n_dims[-3:] is [theta, x, y]
            left_distance: int, distance travelled by left wheel since last timestep
            right: int, distance travelled by right wheel since last timestep
        '''
        d_theta = (right_dist - left_dist) / self.width
        pos[...,0] += d_theta
        pos[...,1] += (right_dist + left_dist) / 2 * np.cos(pos[...,0]) # x update
        pos[...,2] += (right_dist + left_dist) / 2 * np.sin(pos[...,0]) # y update

        n_particles = pos.shape[-2]
        if noise:
            pos[...,:,0] += np.random.normal(0, self.theta_scale * d_theta, n_particles)

        return pos

    def map_localize(self):
        ''''
        Main driver function to localize the car.

        Loop over all timesteps, use particle filter to model motion and update the map.

        Operations are in meters. Grid discretization occurs at the final step.
        '''
        # self.sync_timestamps() # sync encoder and lidar timestamps
        # n_timesteps = len(self.data['idx_enc_to_lidar'])

        n_timesteps = min(len(self.data['lidar']), len(self.data['t_encoder']))
        left = (self.data['FL'] + self.data['RL']) / 2 # in mm
        right = (self.data['FR'] + self.data['RR']) / 2
        
        pos = np.zeros((n_timesteps+1, self.n_particles, 3)) # pos[t][p] = [theta, x, y]
        weights = np.ones((self.n_particles))
        self.best_particle = 0
        n_effective = self.n_particles
        
        for t in tqdm(np.arange(n_timesteps)):
            if n_effective < self.n_particles * 0.7:
                # resample if insufficient effective particle
            # if t > 0 and t % self.reseed_interval == 0:
                # # resample at fixed internals

                self.best_particle = np.argmax(weights)
                # resampled_idx = np.random.choice(np.arange(self.n_particles), size=self.n_particles, replace=True, p=weights) # resample with replacement
                resampled_idx = np.repeat(self.best_particle, self.n_particles) # pick the best particle
                pos[t] = pos[t][resampled_idx,...]
                weights = np.ones((self.n_particles))

                # update particle position with noise
                pos[t] = self.dead_reckoning(
                    pos[t], left_dist=left[t], right_dist=right[t],
                    noise=True
                )
            elif t == 20: # initialize noise after cold start
                pos[t] = self.dead_reckoning(
                    pos[t], left_dist=left[t], right_dist=right[t],
                    noise=True
                )
            else:
                pos[t] = self.dead_reckoning(
                    pos[t], left_dist=left[t], right_dist=right[t],
                    noise=False
                )
            
            weights = self.update_weights(pos[t], self.map, weights, t)
            n_effective = (np.sum(weights))**2 / np.sum(weights**2)

            # update map with the best particle
            self.map = self.new_map_lidar(
                best_pos=pos[t][self.best_particle], 
                map=self.map, 
                idx_lidar=t)
            
            pos[t+1] = pos[t]

            #     plt.imshow(self.map, cmap='RdBu')
            #     plt.savefig(f'new_plots/map{t_dummy}.png')

        self.pos = pos
        return self.map
    
    def get_pos(self, best=False):
        if self.best_particle and best:
            return self.pos[self.best_particle]
        
        return self.pos

    
    



    
