# Introduction
In this project, I implemented a Simultaneous Localization and Mapping (SLAM) algorithm for environment mapping using robot wheel odometry and LIDAR data. Reproducible code is available as a [git repository](https://github.com/fyng/SLAM)


# Problem Statement
We are given time-series measurements of wheel odometry data and LIDAR scan data, as well as geometric measurements of the robot. The goal is to map the robot's trajectory while mapping the environment (empty space v.s. obstructions) on a 2D plane.


# Data
The dataset consists of physical measurements of the robot, LIDAR scan data, and wheel odometry data of a 4-wheeled robot. 
- Physical measurements: refer to the [spec sheet](./docs/platform_config.pdf) for details. 
    - wheel diameter: 254 mm
    - wheelbase: 330.2 mm
    - distance between wheels (inner walls): 311.15 mm
    - distance between wheels (outer walls): 476.25 mm
- Odometry measurements: each wheel contains an encoder which measures the wheel rotation. Each data entry is the sum of encoder ticks since the previous timepoint (positive value -> net forward motion). Each tick is 1/360 revolution. The encoder data has the following channels:
    - time: `int` type
    - FR: front right (ticks) 
    - FL: front left (ticks) 
    - RR: rear right (ticks) 
    - RL: rear left (ticks)
- LIDAR measurements: A LIDAR is mounted on the robot and continously scans a cone of -135 to 135 degrees in front of the robot. When the light beam hits an obstable, it bounces back to the detector and a distance is calculated. The LIDAR data has the following channels:
    - time: `int` type
    - scan: `array` of distances of detected obstacles (meters)
    - angle: `array` of angles correspond to the scans (radians)

# Approach
Here I implement two approaches: the naive (odometry only) approach and SLAM. 

In the naive approach, I use dead reckoning to construct a trajectory of the robot. The limitation of this approach is its sensitivity of small errors, e.g. when the robot slips or small deviations in robot dimension parameters. Since the position update is relative, small initial errors can accumulate and produce large deviations in final trajectory. 

SLAM samples from Gaussian noise to create a large number of particles with independently evolving trajectories. A consensus map is constructed taking a weighted sample of the particles based on the agreement of their mapping and the previous consensus map. The sampling procedure can be thought of as simulating a range of robot parameters, making SLAM robust to error. 


## 1. Dead Reckoning
For each time point, we get the incremental distance traveled by a wheel by converting encoder ticks to distance $d$ (mm) via $d = 2 \pi (\frac{ticks}{360})$. To reduce the effect of slippage, we take the average of the front and back wheels to get the distance traveled by the left and right wheels. We also take the average of left and right wheels to get the average distance traveled by the center of mass of the car. Since the interval between timestamps are small, we can get the angle update using the small angle approximation (valid in general for angles < 1 degree):
$$ d\theta = (\frac{\text{right} - \text{left}}{\text{width}}), \qquad \theta = \theta + d\theta$$

Using grade school trigonometry:

$$ dx = (\frac{\text{left} + \text{right}}{2}) \cos \theta \qquad dy = (\frac{\text{left} + \text{right}}{2}) \sin \theta$$

This is implemented as a vectorized numpy operation for better runtime scaling over 100s of particles. To limit the error induced by floating precision, all distance calculations are made in $mm$ and converted to other units downstream as needed.


## 2. Environment mapping
The model maintains a $55m \times 55m$ occupancy grid map at $5cm$ resolution. The value of a grid is its log-probability of being occupied by an obstacle. At each time point, the mounted LIDAR scans the environment and the light ray passes through unoccupied empty space until it hits the first obstacle along its path and bounces back. Each LIDAR beam is mapped to (1) a single occupied grid coordinates, and (2) a series of empty grid coordinates via `getMapCellsFromRay_fclad()`. 

For each time point, a contact updates the log-probability of a grid coordinate by +1, while a passthrough updates the log-probability of a grid coordinate by -0.1. Therefore, a LIDAR beam will need to pass through a coordinate for 10 subsequent timepoints to clear a obstacle registration. The absolute log-probability of each grid is capped at 15, which prevents the map from becoming too confident.

>**NOTE ON LIDAR DATA**:
 Each LIDAR beam consists of $n$ (angle, distance) tuples undergoes preprocessing before being used in mapping. First, LIDAR beams with distance < 0.1 meters are discarded as this might be reflecting off the robot itself. The maximum spec-ed range of the LIDAR is 30 meters, so any beams with distance > 30 meters is also removed. Lastly, the (angle, distance) tuples are translated from the robot frame to the world frame. 


## 3. SLAM


# Results
**Test dataset:**
TODO: add images for odometry only
TODO: add images for SLAM

Train split:
TODO: add images

Validation split:
TODO: ADD Images


# Appendix
