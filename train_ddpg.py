from ddpg.ddpg_agent_pytorch import DDPGAgent
from env.mykuka import KukaPAP as KukaDiverseObjectEnv


if __name__ == "__main__":
    agent = DDPGAgent(15, 2)
    env = KukaDiverseObjectEnv()
    agent.run(env=env)