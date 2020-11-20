
import gym
import numpy as np

class EnvReward(gym.Wrapper):

        def __init__(self,env):
                super().__init__(env)
                self.env=env

                self.observation_space=gym.spaces.Box(float('-inf'),float('inf'),shape=(25,),dtype=np.float32)


        def step(self,action):
                obs, reward, done, info = self.env.step(action)
                # for i in range(obs.size):
                #         obs[i] = (1 + obs[i] / (1 + abs(obs[i]))) * 0.5
                obs= np.append(obs,reward)
                return obs,reward, done, info

