
from typing import Dict
import numpy as np
import gym
import torch as t
from hebbian_layer import HebbianLayer
from gym.wrappers import *
from torch import nn
from wrappers import EnvReward,SpeedLimitReward

from evostrat import Individual

class BipedaAgentPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Linear(24 ,128 ,bias=False) ,nn.Tanh(),
            nn.Linear(128 ,64 ,bias=False) ,nn.Tanh(),
            nn.Linear(64 ,4 ,bias=False) ,nn.Tanh()
        )

    def forward(self ,observation):
        state =t.tensor(observation, dtype=t.float32)
        return self.cnn(state)


class HebbianBipedalAgentPolicy(nn.Module):
    def __init__(self, i_in, i_out, learn_init, hebbian_update):
        super().__init__()

        self.cnn = nn.Sequential(
            HebbianLayer(i_in, 128, nn.Tanh(), learn_init=learn_init, hebbian_update=hebbian_update),
            HebbianLayer(128, 64, nn.Tanh(), learn_init=learn_init, hebbian_update=hebbian_update),
            HebbianLayer(64, i_out, nn.Tanh(), learn_init=learn_init, hebbian_update=hebbian_update)
        )

    def forward(self, observation):
        state = t.tensor(observation, dtype=t.float32)
        return self.cnn(state)


class BipedalAgent(Individual):
    def __init__(self):
        self.policy_net = BipedaAgentPolicy()

    @staticmethod
    def from_params(params: Dict[str, t.Tensor]) -> 'BipedalAgent':
        agent = BipedalAgent()
        agent.policy_net.load_state_dict(params)
        return agent

    def fitness(self, render=False) -> float:
        gym.logger.set_level(40)

        env = gym.make(self.environment)
        obs = env.reset()
        done = False
        total_reward = 0
        negative_reward = 0
        while not done and negative_reward < 20:
            action = self.action(obs)
            obs, rew, done, info = env.step(action)
            norm_rew = (1 + rew / (1 + abs(rew))) * 0.5

            total_reward += rew
            negative_reward = negative_reward + 1 if rew < -20 else 0
            if render:
                env.render()
        env.close()
        return total_reward

    def get_params(self) -> Dict[str, t.Tensor]:
        return self.policy_net.state_dict()

    def action(self, observation):
        out = self.policy_net(observation)
        return out




class HebbianBipedalAgent(Individual):
    def __init__(self, environment="BipedalWalker-v3",hebbian_update=False, learn_init=True):

        self.environment = environment

        self.policy_net = HebbianBipedalAgentPolicy(24, 4, learn_init, hebbian_update)

    @staticmethod
    def from_params(params: Dict[str, t.Tensor]) -> 'HebbianBipedalAgent':
        agent = HebbianBipedalAgent()
        agent.policy_net.load_state_dict(params)
        return agent

    def fitness(self, render=False) -> float:
        gym.logger.set_level(40)
        env = gym.make(self.environment)
        # env= gym.make('BipedalWalkerHardcore-v3')

        obs = env.reset()  # 24

        done = False
        total_reward = 0
        negative_reward = 0
        speeds = []
        while not done and negative_reward < 20:
            action = self.action(obs)
            obs, rew, done, info = env.step(action)
            #('Vertical Speed: %s' % obs[2])

            #speeds= np.append(speeds,obs[2])
            total_reward += rew
            negative_reward = negative_reward + 1 if rew < -20 else 0
            if render:
                env.render()
        env.close()
        #print("Average speed: %s" % (np.mean(speeds)))
        return total_reward

    def get_params(self) -> Dict[str, t.Tensor]:
        return self.policy_net.state_dict()

    def action(self, observation):
        with t.no_grad():
            out = self.policy_net(observation)
            return out



class HebbianRewardBipedalAgent(Individual):

    def __init__(self, environment="BipedalWalker-v3", hebbian_update=False, learn_init=True):

        self.environment = environment
        self.policy_net = HebbianBipedalAgentPolicy(25, 4, learn_init, hebbian_update)

    @staticmethod
    def from_params(params: Dict[str, t.Tensor]) -> 'HebbianRewardBipedalAgent':
        agent =  HebbianRewardBipedalAgent()
        agent.policy_net.load_state_dict(params)
        return agent

    def fitness(self, render=False) -> float:
        gym.logger.set_level(40)
        env = EnvReward(gym.make(self.environment))
        # env= gym.make('BipedalWalkerHardcore-v3')
        obs = env.reset()  # 24
        obs = np.append(obs, 0);
        done = False
        total_reward = 0
        negative_reward = 0
        speeds = []

        while not done and negative_reward < 40:
            action = self.action(obs)
            obs, rew, done, info = env.step(action)
            total_reward += rew
            #print('Vertical Speed: %s' % obs[2])
            speeds = np.append(speeds, obs[2])
            negative_reward = negative_reward + 1 if rew < -20 else 0
            if render:
                env.render()
        #print("Average speed: %s" % (np.mean(speeds)))
        env.close()
        return total_reward



    def get_params(self) -> Dict[str, t.Tensor]:
        return self.policy_net.state_dict()

    def action(self, observation):
        with t.no_grad():
            out = self.policy_net(observation)

            return out

class HebbianSPBipedalAgent(Individual):

    def __init__(self, environment="BipedalWalker-v3",hebbian_update=False, learn_init=False,):

        self.environment = environment
        self.policy_net = HebbianBipedalAgentPolicy(25, 4, learn_init, hebbian_update)


    def from_params(params: Dict[str, t.Tensor]) -> 'HebbianSPBipedalAgent':
        agent =  HebbianSPBipedalAgent()
        agent.policy_net.load_state_dict(params)
        return agent

    def run(self,render):

        obs = self.env.reset()  # 24
        obs = np.append(obs, 0);
        done = False
        total_reward = 0
        negative_reward = 0
        speeds = []

        while not done and negative_reward < 40:
            action = self.action(obs)
            obs, rew, done, info = self.env.step(action)
            total_reward += rew
            # print('Vertical Speed: %s' % obs[2])
            speeds = np.append(speeds, obs[2])
            negative_reward = negative_reward + 1 if rew < -20 else 0
            if render:
                self.env.render()
        # print("Average speed: %s" % (np.mean(speeds)))
        self.env.close()
        return total_reward

    def fitness(self, render=False) -> float:
        gym.logger.set_level(40)
        fitness_values=[]
        self.env = SpeedLimitReward(gym.make(self.environment),speed=0.25)
        value = self.run(render)
        fitness_values= np.append(fitness_values,value)
        self.env = SpeedLimitReward(gym.make(self.environment), speed=0.3)
        value = self.run(render)
        fitness_values=np.append(fitness_values,value)

        return fitness_values.mean()





    def get_params(self) -> Dict[str, t.Tensor]:
        return self.policy_net.state_dict()

    def action(self, observation):
        with t.no_grad():
            out = self.policy_net(observation)
            return out