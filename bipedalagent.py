

from typing import Dict
import numpy as np
import gym
import torch as t
from hebbian_layer import HebbianLayer
from gym.wrappers import *
from torch import nn
from wrappers import EnvReward

from evostrat import Individual

class BipedaAgentPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Linear(24,128,bias=False),nn.Tanh(),
            nn.Linear(128,64,bias=False),nn.Tanh(),
            nn.Linear(64,5,bias=False),nn.Tanh()
        )

    def forward(self,observation):
        state =t.tensor(observation,dtype=t.float32)
        return self.cnn(state)


class HebbianBipedalAgentPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn=nn.Sequential(
            HebbianLayer(25,128,nn.Tanh()),
            HebbianLayer(128,5,nn.Tanh())
        )
    def forward(self,observation):
        state =t.tensor(observation,dtype=t.float32)
        return self.cnn(state)


class BipedalAgent(Individual):
    def __init__(self):
        self.policy_net = BipedaAgentPolicy()

    @staticmethod
    def from_params(params: Dict[str, t.Tensor]) -> 'BipedalAgent':
        agent = BipedalAgent()
        agent.policy_net.load_state_dict(params)
        return agent

    def fitness(self, render=False)-> float :
        gym.logger.set_level(40)
        env=(gym.make('BipedalWalker-v3'))
        obs = env.reset()
        done = False
        total_reward=0
        negative_reward=0
        while not done and negative_reward < 20 :
            action  = self.action(obs)
            obs, rew,done, info = env.step(action)
            norm_rew = (1 + rew / (1 + abs(rew))) * 0.5

            total_reward+=rew
            negative_reward = negative_reward+1 if rew<-20 else 0
            if render:
                env.render()
        env.close()
        return total_reward

    def get_params(self) -> Dict[str, t.Tensor]:
        return self.policy_net.state_dict()

    def action(self,observation):
        out = self.policy_net(observation)
        return out

class HebbianBipedalAgent(Individual):
    def __init__(self):
        self.policy_net=HebbianBipedalAgentPolicy()

    def from_params(params: Dict[str, t.Tensor]) -> 'HebbianBipedalAgent':
        agent = HebbianBipedalAgent()
        agent.policy_net.load_state_dict(params)
        return agent

    def fitness(self, render=False)-> float :
        gym.logger.set_level(40)
        env= EnvReward(gym.make('BipedalWalker-v3'))
        obs = env.reset()
        obs = np.append(obs,0);
        done = False
        total_reward=0
        negative_reward=0
        while not done and negative_reward < 20 :
            action  = self.action(obs)
            obs, rew,done, info = env.step(action)
            total_reward+=rew
            negative_reward = negative_reward+1 if rew<-20 else 0
            if render:
                env.render()
        env.close()
        return total_reward

    def get_params(self) -> Dict[str, t.Tensor]:
        return self.policy_net.state_dict()

    def action(self,observation):
        out = self.policy_net(observation)
        return out