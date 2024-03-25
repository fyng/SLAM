import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
from utils.load_data import get_lidar, get_encoder, get_imu
from utils.test_load_data import replay_lidar


train_path = Path('../data/train')

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
                FL, FR, RL, RR, ts = get_encoder(train_path / file)
                data_dict[num].update({'FL': FL, 'FR': FR, 'RL': RL, 'RR': RR, 'ts_encoder': ts})
            case 'Hokuyo':
                lidar = get_lidar(train_path / file)
                data_dict[num].update({'lidar': lidar})
            case 'imu':
                acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, ts = get_imu(train_path / file)
                data_dict[num].update({'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z, 'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z, 'ts_imu': ts})
            case _:
                raise ValueError('Unrecognized file type: ', type)
                # maybe switch to print or logging instead of raising an error
    else:
        print(f'Skipped unrecognized filename configuration: {file}')
    
# try to visualize one of the data point
run = list(data_dict.keys())[0]
lidar = data_dict[run]['lidar']
replay_lidar(lidar)

# great!
        


        
