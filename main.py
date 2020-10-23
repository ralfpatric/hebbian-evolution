from multiprocessing.pool import Pool

import tqdm
from torch.optim import Adam

import util
from es import es_grads
from gmm_hebbian_population import GMMHebbianPopulation
from normal_hebbian_population import NormalHebbianPopulation

if __name__ == '__main__':
    train_writer, test_writer = util.get_writers('hebbian')

    scale = 0.1
    num_learning_rules = 2
    population = NormalHebbianPopulation(scale)


    learning_rate = 0.1
    iterations = 300
    pop_size = 200

    optim = Adam(population.parameters(), lr=learning_rate)
    pbar = tqdm.tqdm(range(iterations))
    for i in pbar:
        optim.zero_grad()
        with Pool(8) as pool:
            avg_fitness = es_grads(population, pop_size, pool, util.compute_centered_ranks)

        train_writer.add_scalar('fitness', avg_fitness, i)
        # for key, ent in population.average_mixing_entroy().items():
        #   train_writer.add_scalar('entropy/%s' % key, ent, i)
        optim.step()
        pbar.set_description("avg fit: %.3f" % avg_fitness)
        population.save('latest.t')
