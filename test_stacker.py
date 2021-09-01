import gym
import dmc2gym
import time
from dm_control import  suite
import local_dm_control_suite

for domain_name , task_name in local_dm_control_suite.BENCHMARKING:
    print(domain_name, task_name)
# make the dm_control environment

# env = dmc2gym.make(domain_name="manipulator", task_name="bring_ball")
env = dmc2gym.make(domain_name="manipulation", task_name="stack_2_bricks_vision")


print(env.observation_space)

# use same syntax as in gym
env.reset()
for t in range(1000):
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    env.render(mode="rgb_array")
    time.sleep(1)
    print(t, observation)

