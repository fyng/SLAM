import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from SLAM.utils.load_data import get_lidar, get_encoder, get_imu
from SLAM.utils.test_load_data import replay_lidar
from SLAM.mapping import SLAM

train_path = Path('data/train')

width=470
wheel_diameter=254
enc_to_rev=360

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
    
# try to visualize one of the data point
for run in list(data_dict.keys()):
    print(run)

    mapping = SLAM(width, wheel_diameter, enc_to_rev)
    mapping.load_encoder(data_dict[run]['Encoder'])
    mapping.load_lidar(data_dict[run]['Lidar'])
    mapping.load_imu(data_dict[run]['IMU'])
    pos = mapping.dead_reckoning()

    plt.plot(pos[:,1], pos[:,2], '-')
    plt.savefig(f'plots/map{run}.png')
    plt.close()


        
