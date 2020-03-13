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
    outputs = Dense(n_out,activation = Activation(threshold_on),kernel_constraint=cs.NonNeg(), kernel_regularizer = regularizers.l1(0.001), use_bias = False) # dpnt put the init in here or it breaks
    outputs_act = outputs(h)

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

def train_network(layer_sizes, x_train, y_train, n_epochs):

    model = create_network(*layer_sizes)


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
    plt.title('trained model output')
    plt.show()

    minimal_model = get_minimal_model(model)


    return minimal_model


if __name__ == '__main__':

    #sys.exit()
    n_in = 2
    n_out = 1
    n_h1 = 5
    n_h2 = 5
    n_epochs = 100

    layer_sizes = [n_in, n_h1, n_h2, n_out]

    x_train, y_train = generate_logic_func(10000, [1,0,0,1])

    '''
    plt.scatter(x_train[:,0], x_train[:,1], c = y_train)
    plt.title('training function')
    plt.show()
    '''

    minimal_model = train_network(layer_sizes, x_train, y_train, n_epochs)
    print(minimal_model)
    np.save('network_out/minimal_model.npy', minimal_model)
