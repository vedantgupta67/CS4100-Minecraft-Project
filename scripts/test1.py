import gym
import minerl
env = gym.make('MineRLBasaltFindCave-v0')
obs = env.reset()
print('Environment created successfully!')
print('Observation space:', env.observation_space)
print('Action space:', env.action_space)
env.close()

## Need to change venv to 3.10 so it will install correctly
## Should write README setup for how to setup minerl