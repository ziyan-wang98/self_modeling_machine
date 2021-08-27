# pylint: disable=not-callable
# pylint: disable=no-member

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


class Agent(object):
    def __init__(self, env, planner, logger=None):
        self.env = env
        self.planner = planner
        self.logger = logger

    def get_seed_episodes(self, buffer, n_episodes=1):
        for _ in range(n_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                buffer.add(state, action, reward, next_state)
                state = deepcopy(next_state)
                if done:
                    break
        return buffer

    def run_episode(self, buffer=None, action_noise=None, recorder=None):
        total_reward = 0
        total_steps = 0
        done = False

        with torch.no_grad():
            state = self.env.reset()
            while not done:

                action = self.planner(state)
                if action_noise is not None:
                    action = self._add_action_noise(action, action_noise)
                action = action.cpu().detach().numpy()

                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                total_steps += 1

                if self.logger is not None and total_steps % 25 == 0:
                    self.logger.log(
                        "> Step {} [reward {:.2f}]".format(total_steps, total_reward)
                    )

                if buffer is not None:
                    buffer.add(state, action, reward, next_state)
                if recorder is not None:
                    recorder.capture_frame()

                state = deepcopy(next_state)
                if done:
                    break

        if recorder is not None:
            recorder.close()
            del recorder

        self.env.close()
        stats = self.planner.return_stats()
        return total_reward, total_steps, stats

    def _add_action_noise(self, action, noise):
        if noise is not None:
            action = action + noise * torch.randn_like(action)
        return action
