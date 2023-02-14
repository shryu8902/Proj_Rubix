#%%
import gymnasium as gym
from envs.rubixube333 import RubiXubeEnv
import matplotlib.pyplot as plt
#%%
env = RubiXubeEnv(render_mode='human')
observation = env.reset(10)

for _ in range(10):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation= env.reset(10)

env.close()
