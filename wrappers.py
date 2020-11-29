
import gym
import numpy as np
from scipy import stats

class EnvReward(gym.Wrapper):

        def __init__(self,env,limit=False):
                super().__init__(env)
                self.env=env
                self.mean=0.2
                self.std = 0.1
                self.norm= stats.norm(self.mean,self.std)
                self.observation_space=gym.spaces.Box(float('-inf'),float('inf'),shape=(25,),dtype=np.float32)
                self.limit=limit

        def step(self,action):
                obs, reward, done, info = self.env.step(action)
                # for i in range(obs.size):
                #         obs[i] = (1 + obs[i] / (1 + abs(obs[i]))) * 0.5

                reward=self.norm.pdf(obs[2])*reward
                obs= np.append(obs,reward)
                return obs,reward, done, info

