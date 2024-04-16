import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from SLAM.utils.load_data import get_lidar, get_encoder, get_imu
from SLAM.utils.test_load_data import replay_lidar
from SLAM.mapping import SLAM

train_path = Path('data/test')

# map current file structure to a file dictionary
data_dict = {}
for file in os.listdir(train_path):
    pattern = re.compile(r"([a-zA-Z]+)(\d+)\.mat")
    res = pattern.match(file)
    if res:
        type = res.group(1)
        num = res.group(2)
        if num not in data_dict:
            data_dict[num] = {}
        match type:
            case 'Encoders':
                data_dict[num]['Encoder'] = train_path / file
            case 'Hokuyo':
                data_dict[num]['Lidar'] = train_path / file
            case 'imu':
                data_dict[num]['IMU'] = train_path / file
            case _:
                raise ValueError('Unrecognized file type: ', type)
    else:
        print(f'Skipped unrecognized filename configuration: {file}')
    
use_slam = True
# set car variables
width = 730
wheel_radius = 254 / 2
enc_to_rev = 360   

os.makedirs('new_plots', exist_ok=True)

# try to visualize one of the data point
for run in list(data_dict.keys()):
    print(run, data_dict[run]['Encoder'])

    mapping = SLAM(
        width=width, 
        wheel_radius=wheel_radius, 
        enc_to_rev=enc_to_rev,
        slam=use_slam,
    )

    mapping.load_encoder(data_dict[run]['Encoder'])
    mapping.load_lidar(data_dict[run]['Lidar'])

    map = mapping.map_localize()
    plt.imshow(map, cmap='RdBu')
    plt.colorbar()
    plt.savefig(f'figures/map{run}_{width}_slam.png')
    plt.close()

    pos = mapping.get_pos(best=False)
    plt.plot(pos[...,1], pos[...,2], '-')
    plt.savefig(f'figures/map{run}_path_{width}_slam.png')
    plt.close()


        
