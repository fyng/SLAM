## exploratory data analysis
from pathlib import Path
from SLAM.utils.load_data import get_encoder, get_imu, get_lidar
from matplotlib import pyplot as plt    
import numpy as np

enc_path = Path('data/train/Encoders21.mat')
lidar_path = Path('data/train/Hokuyo21.mat')

FL, FR, RL, RR, ts = get_encoder(enc_path)
lidar = get_lidar(lidar_path)
t_lidar = [data['t'] for data in lidar]

n = 100
# plt.plot(ts[:n], FL[:n], label='FL')
# plt.plot(ts[:n], FR[:n], label='FR')
# plt.plot(ts[:n], RL[:n], label='RL')
# plt.plot(ts[:n], RR[:n], label='RR')
# plt.plot(t_lidar[:n], [1]*len(t_lidar[:n]), label='lidar')
# plt.legend()
# plt.savefig(f'plots/data20_first_{n}.png')


# for i in range(len(lidar))[:n]:
#     idx = np.abs(ts - lidar[i]['t']).argmin()
#     print(idx)

plt.plot(lidar[0]['scan'].reshape(-1))
plt.savefig(f'plots/data20_lidar_range.png')