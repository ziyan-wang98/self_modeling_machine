from ddpg.ddpg_agent_pytorch import DDPGAgent
from env.mykuka import KukaPAP as KukaDiverseObjectEnv


if __name__ == "__main__":
    agent = DDPGAgent(BATCH_SIZE=32, TOTAL_STEPS=10000)
    env = KukaDiverseObjectEnv()
    agent.run(env=env)