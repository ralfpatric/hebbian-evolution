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
    parser.add_argument('--out', type=str, default='None', metavar='',
                        help='The folder name of the training')
    args = parser.parse_args()
    train_writer= util.get_writers(args.out)
    agent = HebbianBipedalAgent(learn_init=True,hebbian_update=False)
    #agent = BipedalAgent()
    shapes = {k: p.shape for k, p in agent.get_params().items()}
    population =NormalPopulation(shapes, HebbianBipedalAgent.from_params,std=0.1)
    #population = MultivariateNormalPopulation(shapes,HebbianBipedalAgent.from_params)
    #population =NormalPopulation(shapes, BipedalAgent.from_params,std=0.1)

    iterations = 1200
    pop_size = 200

    optim = Adam(population.parameters(), lr=0.05)
    pbar = tqdm.tqdm(range(iterations))
    for i in pbar:
        optim.zero_grad()
        with Pool() as pool:
            raw_fitness = population.fitness_grads(pop_size, pool, compute_centered_ranks)

        train_writer.add_scalar('fitness', raw_fitness.mean(), i)
        train_writer.add_scalar('fitness_max',raw_fitness.max(),i)
        train_writer.add_scalar('fitness/std', raw_fitness.std(), i)

        optim.step()
        pbar.set_description("avg fit: %.3f, std: %.3f" % (raw_fitness.mean().item(), raw_fitness.std().item()))

        if i % 20 == 0:
            t.save(population.parameters(),"data_%s_%s_%s.t" % (args.out,i,raw_fitness.mean().item()))

        if raw_fitness.mean() > 299:
            t.save(population.parameters(), 'sol.t')
            break

    t.save(population.parameters(),"/%s/final_params.t" % args.out)

if __name__ == '__main__':
    main(sys.argv)