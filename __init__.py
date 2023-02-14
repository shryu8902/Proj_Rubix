from gymnasium.envs.registration import register

register(
    id='rubixube-v0',
    entry_point='Proj_Rubix.envs:RubiXubeEnv',
    max_episode_steps=300,
)