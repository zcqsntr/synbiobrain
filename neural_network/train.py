import sys
import os
import numpy as np
import tensorflow as tf
import math
import random

from tensorflow import keras

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras import regularizers
import tensorflow.keras.constraints as cs

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras import utils as np_utils

from numpy.random import random

import copy

tf.reset_default_graph()

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def threshold_on(x):
    return tf.nn.sigmoid(x - 5)
    #return sigmoid(x-5)

def threshold_off(x):
    return tf.add(- tf.nn.sigmoid(x - 5), 1)
    #return  -sigmoid(x - 5)+ 1

# try bandpass function


def create_network(n_in, n_h1, n_h2, n_out, weights = None):

    regularizer = tf.contrib.layers.l1_regularizer(scale=0.1) # L1 reg encurages sparsity


    #inputs
    inputs = Input(shape= (n_in, ), name = "inputs")

    init = keras.initializers.RandomUniform(minval=0, maxval=20, seed=None) # extremem but it helps as gradients low near 0 for the custom activation runctions

    #h1 = Dense(n_hidden_nodes//2, activation = Activation(threshold_on), kernel_regularizer = regularizers.l1(0.01), kernel_constraint=cs.NonNeg())(inputs)
    h1 = Dense(n_h1, activation = Activation(threshold_on), kernel_constraint=cs.NonNeg(), kernel_initializer = init, kernel_regularizer = regularizers.l1(0.005),  use_bias = False)
    h1_act = h1(inputs)

    #h2 = Dense(n_hidden_nodes//2, activation = Activation(threshold_off), kernel_regularizer = regularizers.l1(0.01), kernel_constraint=cs.NonNeg())(inputs)
    h2 = Dense(n_h2, activation = Activation(threshold_off), kernel_constraint=cs.NonNeg(), kernel_initializer = init, kernel_regularizer = regularizers.l1(0.005), use_bias = False)
    h2_act = h2(inputs)



    h = tf.concat([h1_act, h2_act], axis = 1)

    #outputs = Dense(n_output_nodes,activation = Activation(threshold_on),  name = "outputs", kernel_regularizer = regularizers.l1(0.01), kernel_constraint=cs.NonNeg())(h)
    outputs = Dense(n_out,activation = Activation(threshold_on),kernel_constraint=cs.NonNeg(), kernel_regularizer = regularizers.l1(0.005), use_bias = False)
    outputs_act = outputs(h) # dpnt put the init in here or it breaks

    if weights is not None:
        print(weights[0][:, 0:n_h1].shape)
        h1.set_weights([weights[0][:, 0:n_h1]])
        h2.set_weights([weights[0][:, n_h1:]])
        outputs.set_weights([weights[1]])

    model = tf.keras.Model(inputs = inputs, outputs = outputs_act)
    model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
    return model

def get_minimal_model(model):
    weights = []

    for layer in model.layers:
        for weight in layer.get_weights(): # [0] is weights [1] is bias
            weights.append(weight)

    (h1_w, h2_w,output_w) =  weights

    hidden_active = []
    reduced_h_weights = []
    reduced_output_weights = []

    n_hidden = len(h1_w[0]) + len(h2_w[0])
    for hidden in range(n_hidden):

        if hidden < 5: #  in h1
            if (h1_w[0][hidden] < 1e-5 and h1_w[1][hidden] < 1e-5) or (output_w[hidden][0] < 1e-5 and output_w[hidden][1] < 1e-5):

                hidden_active.append(0)
            else:

                reduced_h_weights.append(h1_w[:,hidden])
                reduced_output_weights.append(output_w[hidden,:])

                hidden_active.append(1)
        else: #if in h2
            if (h2_w[0][hidden-5] < 1e-5 and h2_w[1][hidden-5] < 1e-5) or (output_w[hidden][0] < 1e-5 and output_w[hidden][1] < 1e-5):

                hidden_active.append(0)
            else:
                reduced_h_weights.append(h2_w[:,hidden-5])
                reduced_output_weights.append(output_w[hidden,:])

                hidden_active.append(1)

    reduced_h_weights = np.array(reduced_h_weights)
    reduced_output_weights = np.array(reduced_output_weights)


    return hidden_active, [reduced_h_weights.T, reduced_output_weights]

def generate_bandpass(n):
    x = np.random.rand(n, 1)

    y = np.ones_like(x)
    #print(u)

    y[x > 0.75] = 0
    y[x < 0.25] = 0
    return x, y

def generate_threshold(n):
    x = np.random.rand(n, 1)

    y = np.ones_like(x)
    #print(u)


    y[x < 0.5] = 0
    return x, y

def generate_poly(n):
    x = np.random.rand(n, 1)

    y = 2*x**3 - x**2 +1

    return x, y


def target_function(n):
    x = np.random.rand(n, 1)
    y = np.sin(x*3)*10
    return x,y

def generate_logic_func(n, func_list, mode = 'cont'):
    if mode == 'cont':
        x = np.random.rand(n, 2)
    elif mode == 'disc':
        x = np.random.randint(2, size = (n,2))
    elif mode == 'both':
        x1 = np.random.rand(n//2, 2)
        x2 = np.random.randint(2, size = (n//2,2))
        x = np.append(x1, x2, axis = 0)

    y = np.zeros_like(x[:,0])

    for i in range(len(y)):
        if x[i, 0] < 0.5 and x[i, 1] < 0.5:
            y[i] = func_list[0]
        elif x[i, 0] > 0.5 and x[i, 1] < 0.5:
            y[i] = func_list[1]
        elif x[i, 0] < 0.5 and x[i, 1] > 0.5:
            y[i] = func_list[2]
        elif x[i, 0] > 0.5 and x[i, 1] > 0.5:
            y[i] = func_list[3]

    return x, y


def add(n):
    x = np.random.rand(n, 2)
    y = np.sum(x, axis = 1)
    return x,y

def subtract(n):
    x = np.random.rand(n, 2)
    y = np.subtract(x, axis = 1)
    return x,y

def train_network(layer_sizes, n_epochs):

    model = create_network(*layer_sizes)

    x_train, y_train = add(10000) # training with both eems more robust

    plt.scatter(x_train[:,0], x_train[:,1], c = y_train)
    plt.show()
    #y_train =  np_utils.to_categorical(y_train)

    '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalizing the data to help with the training
    X_train /= 255
    X_test /= 255

    n_classes = 10
    print("Shape before one-hot encoding: ", y_train.shape)
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)
    '''



    model.fit(x_train, y_train, batch_size=32, epochs=n_epochs)

    #loss_and_metrics = model.evaluate(X_test, Y_test)



    x_test, y_test = generate_logic_func(10000, [1,0,0,1])

    ys = model.predict(x_test).reshape(10000,)

    #ys = np.argmax(ys, axis = 1)

    plt.scatter(x_test[:,0], x_test[:,1], c = ys)
    plt.show()

    minimal_model = get_minimal_model(model)


    return minimal_model


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

def get_fitness(weights, target_weights):

    fitness = 0
    for i in range(len(weights)):
        fitness -= np.linalg.norm((weights[i] - target_weights[i]))

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
        grids.append([[[random()*10, random()*10]]* n_in,
                        [[random()*10,random()*10]] * n_hidden,
                        [[random()*10, random()*10]] * n_out])


    return grids

def get_node_positions(minimal_model, n_gens):
    '''
    we need to make AHL_func(r) between each node as close as possible to weights in minimal_model
    '''
    target_weights = minimal_model[1]

    n_h1 = np.sum(minimal_model[0][0:5]) # the number of threshold on nodes
    #target_weights = np.load('weights.npy', allow_pickle = True)
    grids = generate_random_grids(10000, len(target_weights[1]))

    for gen in range(n_gens):
        print('generation: ', gen)
        # selection
        fitnesses = []

        for grid in grids:
            weights = get_weights(grid)
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
            for i in range(len(get_weights(grids[indices[-1]]))):
                print(get_weights(best_grid)[i])
                print(target_weights[i])
                print()
                print(best_fitness)
                plot_grid(grids[indices[-1]], iter, n_h1)

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
        for child in children:
            good_grids.append(child)

        grids = good_grids

    weights = get_weights(best_grid)
    print('wegihts from positions: ')
    print(weights)

    return best_grid, best_fitness


def plot_grid(grid, iter, n_h1):

    plt.figure()

    # plot nodes
    colours = ['g', 'blue', 'aqua',  'red']
    r = 0.01
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


#def feed_forward(weights, input):




if __name__ == '__main__':
    '''
    xs = np.arange(0, 10, 0.1)
    ys1 = [threshold_on(x) for x in xs]
    ys2 = [threshold_off(x) for x in xs]
    print(ys1)
    plt.figure()
    plt.plot(xs, ys1)
    plt.figure()
    plt.plot(xs, ys2)
    plt.show()
    '''



    n_in = 2
    n_out = 1
    n_h1 = 5
    n_h2 = 5
    n_epochs = 100
    n_gens = 200

    layer_sizes = [n_in, n_h1, n_h2, n_out]
    minimal_model = train_network(layer_sizes, n_epochs)
    print(minimal_model)
    np.save('minimal_model.npy', minimal_model)
    minimal_model = np.load('minimal_model.npy', allow_pickle = True)
    node_positions, fitness = get_node_positions(minimal_model, n_gens)


    weights = get_weights(node_positions)
    print('wegihts from positions: ')
    print(weights)


    print(node_positions)


    n_h1 = sum(minimal_model[0][0:5])
    n_h2 = sum(minimal_model[0][5:])

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
