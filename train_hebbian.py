import torch as t
import tqdm
import sys
import argparse
from evostrat import NormalPopulation, compute_centered_ranks, MultivariateNormalPopulation
from torch.multiprocessing import Pool, set_start_method
from torch.optim import Adam

import util

from bipedalagent import BipedalAgent, HebbianBipedalAgent


def main(argv):
    set_start_method('spawn')
    parser = argparse.ArgumentParser()

    parser.add_argument('--param_path', type=str, default='None', metavar='',
                        help='Saved parameters')
    parser.add_argument('--generation', type=int, default=1200, help='# of generations')
    parser.add_argument('--pop_size', type=int, default=200, help='population size')
    parser.add_argument('--environment', type=str, default="BipedalWalker-v3",
                        help='Agent environment')
    parser.add_argument('--reward', type=bool, default=False, help='If the agent should use reward as an input')

    parser.add_argument('--static_iteration', type=int, default=20,
                        help="# of generations trained for static network. If 0, using the max_fitness threshold")

    parser.add_argument('--threshold', type=int, default=10,
                        help="Threshold used for checking fitness value")
    parser.add_argument('--fail_generation', type=int, default=0,
                        help="#of failed generation before switch to hebbian")
    parser.add_argument('--out', type=str, default='None', metavar='',
                        help='The folder name of the training')
    args = parser.parse_args()
    train_writer = util.get_writers(args.out)

    agent = HebbianBipedalAgent(args.environment,args.reward, learn_init=True, hebbian_update=False)

    shapes = {k: p.shape for k, p in agent.get_params().items()}
    population = NormalPopulation(shapes, HebbianBipedalAgent.from_params, std=0.1)



    iterations = args.generation
    pop_size = args.pop_size

    max_fitness_mean = float('-inf')
    threshold = args.threshold

    numFailGeneration = args.fail_generation

    numFails = 0

    hebbian = False
    optim = Adam(population.parameters(), lr=0.05)
    pbar = tqdm.tqdm(range(iterations))
    for i in pbar:
        optim.zero_grad()
        with Pool() as pool:
            raw_fitness = population.fitness_grads(pop_size, pool, compute_centered_ranks)

        train_writer.add_scalar('fitness', raw_fitness.mean(), i)
        train_writer.add_scalar('fitness_max', raw_fitness.max(), i)
        train_writer.add_scalar('fitness/std', raw_fitness.std(), i)

        optim.step()
        pbar.set_description("avg fit: %.3f, std: %.3f" % (raw_fitness.mean().item(), raw_fitness.std().item()))

        if hebbian == False:
            if max_fitness_mean + threshold < raw_fitness.mean().item():
                max_fitness_mean = raw_fitness.mean().item()
            else:
                numFails += 1

            if (numFailGeneration > 0 and numFails < numFailGeneration) or (args.static_iteration > 0 and i >  args.static_iteration):
                agent = HebbianBipedalAgent(args.environment,args.reward,True, True)
                static_population = population
                shapes = {k: p.shape for k, p in agent.get_params().items()}
                population = NormalPopulation(shapes, HebbianBipedalAgent.from_params, std=0.1)

                population.param_means = {k: p for k, p in zip(agent.get_params().keys(), static_population.parameters())}
                print("\n Changed to hebbian")
                hebbian = True

        if raw_fitness.mean() > 299:
            t.save(population.parameters(), 'sol.t')
            break


if __name__ == '__main__':
    main(sys.argv)
