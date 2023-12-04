from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3.common.vec_env import VecNormalize

from wrappers import RewardWrapper, FrenetObsWrapper
from wrappers import ActionRandomizer, LidarRandomizer
from wrappers import DelayedAction

from gym.wrappers import FilterObservation, TimeLimit
from gym.wrappers import RescaleAction
from gym.wrappers import FlattenObservation, FrameStack

import numpy as np
import gym
from PIL import Image

NUM_BEAMS=1440
DTYPE = np.float64

def create_env(maps, seed=5, domain_randomize=True, flatten=True):
    env = gym.make(
        "f110_gym:f110-v0",
        num_agents=1,
        maps=maps,
        seed=seed,
        num_beams=NUM_BEAMS,
    )

    env = FrenetObsWrapper(env)
    env = RewardWrapper(env)
    
    env = FilterObservation(env, filter_keys=["scans", "linear_vel"])
    env = TimeLimit(env, max_episode_steps=10000)
    env = RescaleAction(env, np.array([-1.0, 0.0]), np.array([1.0, 1.0]))
    
    if domain_randomize:
        env = LidarRandomizer(env)
        env = ActionRandomizer(env)
        env = DelayedAction(env)
    
    if flatten:
        env = FlattenObservation(env)
        # env = FrameStack(env, 3)
            
    env = Monitor(env, info_keywords=("is_success",), filename='./metrics/data')
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_reward=True, norm_obs=False)

    

    return env

from stable_baselines3.common.vec_env import SubprocVecEnv
# Other imports remain the same

def create_vec_env(maps, seed=5, domain_randomize=True, flatten=True, n_envs=4):
    def make_env(rank):
        def _init():
            env = gym.make(
                "f110_gym:f110-v0",
                num_agents=1,
                maps=maps,
                seed=seed + rank,
                num_beams=NUM_BEAMS,
            )
            env = FrenetObsWrapper(env)
            env = RewardWrapper(env)
            
            env = FilterObservation(env, filter_keys=["scans", "linear_vel"])
            env = TimeLimit(env, max_episode_steps=10000)
            env = RescaleAction(env, np.array([-1.0, 0.0]), np.array([1.0, 1.0]))
            
            if domain_randomize:
                env = LidarRandomizer(env)
                env = ActionRandomizer(env)
                env = DelayedAction(env)
            
            if flatten:
                env = FlattenObservation(env)
                # env = FrameStack(env, 3)
            return env
        return _init

    envs = [make_env(i) for i in range(n_envs)]
    env = SubprocVecEnv(envs) if n_envs > 1 else DummyVecEnv([make_env(0)])
    env = VecNormalize(env, norm_reward=True, norm_obs=False)

    return env



def linear_schedule(initial_learning_rate: float):
    def schedule(progress_remaining: float):
        return initial_learning_rate * progress_remaining

    return schedule

def lidar_to_image(lidar_data, image_size=(512, 512)):
    # Create an empty numpy array for the image
    image = np.zeros(image_size, dtype=np.uint8)
    
    # Convert the LIDAR data to grayscale (0-255) with free area as black
    lidar_data = 255 * (1 - lidar_data)
    
    # Get the middle point of the image
    mid_x, mid_y = image_size[0] // 2, image_size[1] // 2
    
    # Get the angular step for each data point in the LIDAR array
    angle_step = 270.0 / len(lidar_data)
    
    # Iterate over the LIDAR data array
    for i, range_val in enumerate(lidar_data):
        # Calculate the angle of this data point
        angle = np.deg2rad(i * angle_step - 135)  # Start from -135 degree for a 270 degree FoV
        
        # Convert the range value to a pixel offset
        # The maximum offset (for range value of 255) should be half of the image size
        offset = int((range_val / 255.0) * (min(image_size) // 2))
        
        # Calculate the pixel coordinates
        x = mid_x + int(offset * np.cos(angle))
        y = mid_y + int(offset * np.sin(angle))
        
        # Make sure the coordinates are within the image boundaries
        x = np.clip(x, 0, image_size[0] - 1)
        y = np.clip(y, 0, image_size[1] - 1)
        
        # Set this pixel in the image array
        image[y, x] = range_val
    
    # Convert the numpy array to a PIL image
    image = Image.fromarray(image)
    
    return image