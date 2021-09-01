'''
MIT License

Copyright (c) 2018-2020 Ekaterina Nikonova,
Research School of Computer Science, Australian National University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This version of the agent is based on the work presented in: https://arxiv.org/abs/1910.01806

'''

from __future__ import division

import numpy as np
import random
# import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


import os
import torch

from .ReplayBuffer import ReplayBuffer
from .ActorNetwork import ActorNetwork
from .CriticNetwork import CriticNetwork
from env.mykuka import KukaPAP as KukaDiverseObjectEnv

import copy


LRA = 0.0001
LRC = 0.001
GAMMA = 0.95
TAU = 0.001

class DDPGAgent:
    def __init__(self, TOTAL_STEPS, BATCH_SIZE):
        self.TOTAL_STEPS = TOTAL_STEPS
        self.BATCH_SIZE = BATCH_SIZE
        self.memory_size = 10000
        self.n_features = (48, 48, 3)
        self.n_action = 3
        self.memory = ReplayBuffer(self.memory_size)
        self.MODEL_PATH = "./result"

        self.actor = ActorNetwork()
        self.critic = CriticNetwork()

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.criterion_critic = torch.nn.MSELoss(reduction='sum')

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=LRA)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=LRC)

        self.epsilon = 1
        self.epsilon_decay = 0.001



    def train(self):
        if self.memory.count() <= self.BATCH_SIZE:
            return

        batch = self.memory.getBatch(self.BATCH_SIZE)

        states = torch.cat([e[0] for e in batch], dim=0)
        actions = torch.cat([e[1] for e in batch], dim=0)
        rewards = np.asarray([[e[2],] for e in batch])
        new_states = torch.cat([e[3] for e in batch], dim=0)
        dones = np.asarray([e[4] for e in batch])
        y_t = np.zeros_like(rewards)

        # use target network to calculate target_q_value
        target_q_values = self.target_critic(new_states, self.target_actor(new_states)).detach()

        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + GAMMA * target_q_values[k].numpy()
        y_t = torch.as_tensor(y_t).float()

        # training
        self.optimizer_critic.zero_grad()
        self.optimizer_actor.zero_grad()
        q_values = self.critic(states, actions.detach())
        loss = self.criterion_critic(q_values, y_t.detach())
        loss.backward()  ##for param in critic.parameters(): param.grad.data.clamp(-1, 1)
        self.optimizer_critic.step()

        self.optimizer_critic.zero_grad()
        self.optimizer_actor.zero_grad()
        q_values_for_grad = self.critic(states, self.actor(states))
        a_loss = - q_values_for_grad.sum()
        a_loss.backward()
        self.optimizer_actor.step()

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        self.epsilon = max(0, self.epsilon - self.epsilon_decay)


    def run(self, env:KukaDiverseObjectEnv):

        MODEL_PATH = self.MODEL_PATH

        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        # Run loop
        obs = env.reset()
        for env_step in range(1, self.TOTAL_STEPS+1):
            if env_step % 1000 == 0:
                print("Training: {} / {}".format(env_step+1, self.TOTAL_STEPS+1))
            obs = torch.FloatTensor(obs).transpose(0,2).unsqueeze(0) / 255
            action = self.actor(obs)
            next_obs, reward, done, _ = env.step(action[0])


            self.memory.add(
                state=obs,
                action=action,
                reward=reward,
                new_state=torch.FloatTensor(next_obs).transpose(0,2).unsqueeze(0) / 255,
                done=done
            )
            obs = next_obs
            self.train()
            if done:
                obs = env.reset()

            if env_step % 10 == 0:
                        torch.save(self.actor.state_dict(), self.MODEL_PATH + '/actormodel.pth')
                        torch.save(self.critic.state_dict(), self.MODEL_PATH + '/criticmodel.pth')

    def collect(self, env:KukaDiverseObjectEnv, num_samples, replay_buffer, env_id, MODEL_PATH=None):
        print("Start colleccting the Expert Trajectory")
        if not MODEL_PATH is None:
            self.actor = torch.load(MODEL_PATH + '/actormodel.pth')
        else:
            MODEL_PATH = self.MODEL_PATH
            if not os.path.exists(MODEL_PATH):
                os.makedirs(MODEL_PATH)
        # Run loop
        obs = env.reset()
        for env_step in range(1, num_samples+1):
            obs_x = torch.FloatTensor(obs).transpose(0,2).unsqueeze(0) / 255
            action = self.actor(obs_x)
            next_obs, reward, done, _ = env.step(action[0])
            replay_buffer.add(env_id, obs, action.detach().numpy()[0], reward, next_obs, done)

            obs = next_obs
            self.train()
            if done:
                obs = env.reset()
        return replay_buffer

def collect_ddpg_data(env, env_id, num_samples, replay_buffer, mode='load'):
    agent = DDPGAgent(50000, 128)
    if mode == 'train':
        agent.run(env)
    return agent.collect(env, num_samples, replay_buffer, env_id)




