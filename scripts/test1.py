import gym
import minerl
env = gym.make('MineRLBasaltFindCave-v0')
env.seed(88292929494)
obs = env.reset()
print('Environment created successfully!')
print('Observation space:', env.observation_space)
print('Action space:', env.action_space)


for i in range(1000):
    action = env.action_space.sample()
    # Just look around and walk forward
    #action["camera"] = [0, 3]      # turn camera right
    action["ESC"] = 0


    obs, reward, done, info = env.step(action)
    env.render()

    #print(f"Step {i}, reward: {reward}, done: {done}")
    if done:
        print("Episode ended")
        break

env.close()

## Changes made to run: 
# in MCP-Reborn/launchClient.sh:
# - change final java line to: java -XstartOnFirstThread -Xmx$maxMem -jar $fatjar --envPort=$port
#   - ** only on mac
    