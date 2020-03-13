import sys
import os
import numpy as np
import tensorflow as tf
import math
import random
from train_network import create_network, generate_logic_func
import matplotlib.pyplot as plt

from numpy.random import random

import copy


def polynomial(x, params):
    out = np.zeros_like(x, dtype = 'float64')
    for i,p in enumerate(params):
        out += p*x**(len(params)-i-1)
    return out

p = [ 4.50256460e-03, -1.85359341e-01,  3.20202666e+00, -3.01585975e+01,
  1.68470214e+02, -5.68788003e+02,  1.13089022e+03, -1.21638369e+03,
  5.52451546e+02]
# params fittted to diffusion curve

def AHL_func(r):
    return polynomial(r, p)

def AHL_func(r):
    return 47/r**2

def AHL_func(r):
    return 1/r

def get_distance(p1, p2):
    return np.sqrt( (p1[0] - p2[0])**2 +  (p1[1] - p2[1])**2 )

def get_weights(grid):
    hidden_weights = []

    for input_pos in grid[0]:
        weights = []
        for hidden_pos in grid[1]:
            r = get_distance(input_pos, hidden_pos)
            weights.append(AHL_func(r))
        hidden_weights.append(weights)

    output_weights = []

    for hidden_pos in grid[1]:
        weights = []
        for output_pos in grid[2]:
            r = get_distance(hidden_pos, output_pos)
            weights.append(AHL_func(r))
        output_weights.append(weights)

    return [np.array(hidden_weights), np.array(output_weights)]

def get_weights_one_AHL(grid):
    hidden_weights = []

    for input_pos in grid[0]:
        weights = []
        for hidden_pos in grid[1]:
            r = get_distance(input_pos, hidden_pos)
            weights.append(AHL_func(r))
        hidden_weights.append(weights)
    # add hidden to hidden weight to the end of here
    output_weights = []

    for hidden_pos in grid[1]:
        weights = []
        for output_pos in grid[2]:
            r = get_distance(hidden_pos, output_pos)
            weights.append(AHL_func(r))
        output_weights.append(weights)

    return [np.array(hidden_weights), np.array([AHL_func(get_distance(grid[1][0], grid[1][1]))]), np.array(output_weights)]

def get_fitness(weights, target_weights):

    fitness = 0
    for i in range(len(weights)):
        fitness -= np.linalg.norm((weights[i] - target_weights[i]))
        '''
        print(np.linalg.norm((weights[i] - target_weights[i])))
        print(weights[i])
        print(target_weights[i])
        '''
    #print()
    return fitness

def mutate(grid):

    for i in range(len(grid)):
        for j in range(len(grid[i])): # randomly mutate the position of each node
            if random() < 0.1:
                grid[i][j][0] += random()-0.5
            if random() < 0.1:
                grid[i][j][1] += random()-0.5

    return grid

def recombine(grid1, grid2):
    grid3 = copy.deepcopy(grid1)
    grid4 = copy.deepcopy(grid2)

    for i in range(len(grid1)):
        for j in range(len(grid1[i])): # randomly swap some nodes
            if random() < 0.5:
                grid3[i][j] = grid2[i][j]
            if random() < 0.5:
                grid4[i][j] = grid1[i][j]
    return grid3, grid4

def generate_random_grids(n, n_hidden):
    grids = []

    for i in range(n):
        grids.append([[[random()*10, random()*10] for i in range(n_in)],
                        [[random()*10,random()*10] for i in range(n_hidden)] ,
                        [[random()*10, random()*10] for i in range(n_out)]])
    return grids

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def move_concurrent_nodes(grid, node_radius):
    #DEBUG THIS
    #checks to see if any nodes on top of each other, if so moves them


    for i in range(len(grid)):
        for j in range(len(grid[i])): # randomly swap some nodes
            pos = grid[i][j]

            for k in range(len(grid)):
                for l in range(len(grid[k])): # randomly swap some nodes
                    pos2 = grid[k][l]
                    dir = np.array(pos) - np.array(pos2)
                    mag = np.linalg.norm(dir)

                    if (mag <= 2*node_radius) and ((not i == k) or (not j == l)):

                        if mag == 0:
                            x, y = pol2cart(2*node_radius, random()*2*math.pi)
                        else:
                            unit_vec = dir/mag # direction to move node

                            translation_mag = (node_radius * 2 - mag)
                            grid[i][j][0] += unit_vec[0] * translation_mag
                            grid[i][j][1] += unit_vec[1] * translation_mag


    return grid

def get_node_positions(minimal_model, n_gens, initial_pop):
    '''
    we need to make AHL_func(r) between each node as close as possible to weights in minimal_model
    '''
    target_weights = minimal_model[1]

    #for one AHL
    target_weights = [target_weights[0], np.array([0]), target_weights[1]]

    n_h1 = np.sum(minimal_model[0][0:5]) # the number of threshold on nodes
    #target_weights = np.load('weights.npy', allow_pickle = True)

    grids = generate_random_grids(initial_pop, n_h1+n_h2)
    grids = [move_concurrent_nodes(grid, node_radius) for grid in grids]
    for gen in range(n_gens):
        print('generation: ', gen)
        # selection
        fitnesses = []

        for grid in grids:
            weights = get_weights_one_AHL(grid)
            fitness = get_fitness(weights, target_weights)
            fitnesses.append(fitness)

        fitnesses = np.array(fitnesses)
        print(len(fitnesses))
        print(np.mean(fitnesses))
        indices = np.argsort(fitnesses)

        if gen == 0 or fitnesses[indices[-1]] > best_fitness:
            best_grid = copy.deepcopy(grids[indices[-1]])
            best_fitness = fitnesses[indices[-1]]

        reproduce = indices[-5000:] # because fitness negative
        print(fitnesses[indices[-1]])

        if gen %10 == 0:

            print('best weights so far: ', get_weights_one_AHL(best_grid))
            print('best positions so far: ', best_grid)
            print('target weights: ', target_weights)
            print()
            print('best fitness: ', best_fitness)
            plot_grid(grids[indices[-1]],gen, n_h1)

        good_grids = []

        for i in reproduce:
            good_grids.append(grids[i])

        #recombination and mutation
        children = []
        for i, grid in enumerate(good_grids):

            if random() < 0.5: # this grid is going to recombine
                grid2 = good_grids[np.random.choice(range(len(good_grids)))]
                child1, child2 = recombine(grid, grid2)
                children.append(child1)
                children.append(child2)

            if random() < 0.1: #this grid is going to mutate
                good_grids[i] = mutate(grid)

        # add children to grids
        good_grids.append(best_grid) # dont loose best grid
        for child in children:
            good_grids.append(child)

        grids = [move_concurrent_nodes(grid, node_radius) for grid in good_grids]

    weights = get_weights_one_AHL(best_grid)
    print('wegihts from positions: ')
    print(weights)

    return best_grid, best_fitness


def plot_grid(grid, iter, n_h1):

    plt.figure()

    # plot nodes
    colours = ['g', 'blue', 'aqua',  'red']
    r = node_radius
    for node in grid[0]:
        circle = plt.Circle((node[0], node[1]), radius = r, fc = 'g', label = '(' + str(round(node[0], 1)) + ',' + str(round(node[1], 1)) + ')')
        plt.gca().add_patch(circle)

    for i, node in enumerate(grid[1]):
        if i < n_h1:
            circle = plt.Circle((node[0], node[1]), radius =r, fc = 'blue', label = '(' + str(round(node[0], 1)) + ',' + str(round(node[1], 1)) + ')')
            plt.gca().add_patch(circle)
        else:
            circle = plt.Circle((node[0], node[1]), radius = r, fc = 'aqua', label = '(' + str(round(node[0], 1)) + ',' + str(round(node[1], 1)) + ')')
            plt.gca().add_patch(circle)

    for node in grid[2]:
        circle = plt.Circle((node[0], node[1]), radius = r, fc = 'red', label = '(' + str(round(node[0], 1)) + ',' + str(round(node[1], 1)) + ')')
        plt.gca().add_patch(circle)



    #plot weights

    plt.legend()
    plt.axis('scaled')
    plt.savefig(str(iter) + '.png')


if __name__ == '__main__':

    x = np.array(range(1,46))*0.2
    plt.plot(x, AHL_func(x))
    plt.show()

    node_radius  = 0.01 #cm
    n_gens = 200
    initial_pop = 100000

    minimal_model = np.load('network_out/minimal_model.npy', allow_pickle = True)
    print(minimal_model)
    #node_positions = np.load('network_out/node_positions.npy', allow_pickle = True)
    n_in = 2
    n_h1 = sum(minimal_model[0][0:5])
    n_h2 = sum(minimal_model[0][5:])
    n_out = 1
    print()

    node_positions, fitness = get_node_positions(minimal_model, n_gens, initial_pop)

    np.save('network_out/node_positions.npy', node_positions)
    plot_grid(node_positions, 'final', 1)

    weights = get_weights_one_AHL(node_positions)
    print('weights from positions: ')
    print(weights)

    print(node_positions)



    minimal_network = create_network(n_in, n_h1, n_h2, n_out, minimal_model[1])
    model_from_pos = create_network(n_in, n_h1, n_h2, n_out, weights)

    x_test, y_test = generate_logic_func(10000, [1,0,0,1])


    ys = minimal_network.predict(x_test).reshape(10000,)
    print(ys)

    plt.close('all')
    plt.figure()
    plt.scatter(x_test[:,0], x_test[:,1], c = ys)

    ys = model_from_pos.predict(x_test).reshape(10000,)
    print(ys)

    plt.figure()
    plt.scatter(x_test[:,0], x_test[:,1], c = ys)
    plt.show()

    print('---------------------------------------------------------------')
    print()

    for layer in minimal_network.layers:
        print(layer.get_weights()) # [0] is weights [1] is bias

    for layer in model_from_pos.layers:
        print(layer.get_weights()) # [0] is weights [1] is bias
