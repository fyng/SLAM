# SLAM
Robotic environment sensing using wheel odometry and 2D LIDAR data

# Usage
Install micromamba or mamba as the package manager. To install micromamba, refer to the [installation guide](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)

To install the classifier:
1. Clone the repo
```
git clone https://github.com/fyng/SLAM.git
```
```
cd SLAM
```

2. Create virtual environment
```
micromamba env create -f environment.yml
```
```
micromamba activate slam
```

3. Create directory for data and plots


4. Run model
In `main.py`, update the directory of the test folder. Chnage the car variables as needed
```
python main.py
```