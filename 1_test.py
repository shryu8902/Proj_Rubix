
#%%
env = RubiksCubeEnv()
# %%
env.step('F')
env.print_state()
env.step("L")
env.print_state()
env.step("B")
env.reset(5)
env.scramble_history
env.print_state()
#%%
env = RubiksCubeEnv()
env.reset(10)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()