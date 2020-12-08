import torch as t

from evostrat import NormalPopulation
from bipedalagent import HebbianBipedalAgent, BipedalAgent,HebbianSPBipedalAgent

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='None', metavar='',
                        help='Input model weights')
    args = parser.parse_args()


    agent = HebbianSPBipedalAgent(hebbian_update=True,learn_init=True)
    shapes = {k: p.shape for k, p in agent.get_params().items()}
    population = NormalPopulation(shapes, HebbianSPBipedalAgent.from_params, std=0.1)
    params = t.load(args.input)
    population.param_means = {k: p for k, p in zip(agent.get_params().keys(), params)}
    inds, logps = zip(*population.sample(2))

    inds[0].fitness(render=True)