# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:12:07 2020

@author: SabbirPulok
"""

equation_inputs = [4,-2,3.5,5,-11,-4.7]
num_weights = 6

import numpy

def cal_pop_fitness(equation_inputs, pop):
     fitness = numpy.sum(pop*equation_inputs, axis=1)
     return fitness

def select_mating_pool(pop, fitness, num_parents):

    parents = numpy.empty((num_parents, pop.shape[1]))

    for parent_num in range(num_parents):

        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))

        max_fitness_idx = max_fitness_idx[0][0]

        parents[parent_num, :] = pop[max_fitness_idx, :]

        fitness[max_fitness_idx] = -99999999999

    return parents

def crossover(parents, offspring_size):
     offspring = numpy.empty(offspring_size)
     crossover_point = numpy.uint8(offspring_size[1]/2)
 
     for k in range(offspring_size[0]):
    
         parent1_idx = k%parents.shape[0]
         
         parent2_idx = (k+1)%parents.shape[0]
         
         offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
         
         offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
     return offspring

def mutation(offspring_crossover):

    for idx in range(offspring_crossover.shape[0]):

        random_value = numpy.random.uniform(-1.0, 1.0, 1)

        offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value

    return offspring_crossover


sol_per_pop = 8

pop_size = (sol_per_pop, num_weights)

new_population = numpy.random.uniform(low = -4.0, high =4.0, size = pop_size )



num_generations = 5
num_parents_mating = 4

for generation in range(num_generations):
    fitness = cal_pop_fitness(equation_inputs, new_population)
    parents = select_mating_pool(new_population, fitness, num_parents_mating)
    offspring_crossover = crossover(parents, offspring_size = (pop_size[0]-parents.shape[0], num_weights)) 
    offspring_mutation = mutation(offspring_crossover)
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation
    
    print("Generation: ",generation+1)
    print("Fitness value:")
    print(fitness)
    print("Selected parents:")
    print(parents)
    print("Crossover Results:")
    print(offspring_crossover)
    print("Mutation Results:")
    print(offspring_mutation)
    print("Best result after generation",generation+1 ," : ", numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))

fitness = cal_pop_fitness(equation_inputs, new_population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("\nBest solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])