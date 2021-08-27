from env.mykuka2 import KukaPAP

env = KukaPAP(renders=True, isDiscrete=False)
env.reset()


for t in range(60):
    action = env.action_space.sample()
    next_obs, reward, done, _ = env.step(action)
    if done == True:
        env.reset()
    obs = next_obs

