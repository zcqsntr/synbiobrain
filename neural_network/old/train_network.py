import sys
import os
from typing import Any, Union

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import math
import random

from numpy.core._multiarray_umath import ndarray
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
from tensorflow.keras import models
from numpy.random import random
from tensorflow.keras.optimizers import Adam
from tensorflow_model_optimization.sparsity import keras as sparsity
from time import time



# MAYBE SQUASH THE ACTIVATION FUNCTIONS IN THE X DIRECTION
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def threshold_on(x):
    return tf.nn.sigmoid(x - 5)
    #return sigmoid(x-5)

def threshold_off(x):
    return tf.add(- tf.nn.sigmoid(x - 5), 1)

    #return  -sigmoid(x - 5)+ 1

def bandpass(x):
    return tf.add(tf.nn.sigmoid(x-5), -tf.nn.sigmoid(x-15))

def inverse_bandpass(x):
    return tf.add(tf.add(-tf.nn.sigmoid(x-5), tf.nn.sigmoid(x-15)), 1)
# try bandpass function

def create_network(layer_sizes, weights = None):

    n_in = layer_sizes[0]
    hidden_layers = layer_sizes[1:-1]

    n_out = layer_sizes[-1]

    regularizer = tf.contrib.layers.l1_regularizer(scale=0.1) # L1 reg encurages sparsity

    #inputs
    inputs = Input(shape= (n_in, ), name = "inputs")
    # maxval = 20 for everything before tensorflow pruning
    init = keras.initializers.RandomUniform(minval=0, maxval=20, seed=None) # extremem but it helps as gradients low near 0 for the custom activation runctions

    t_total = 0


    prev_layer = inputs
    for i, hidden_layer in enumerate(hidden_layers):
        n_ON, n_OFF, n_BP, n_IBP = hidden_layer

        #h1 = Dense(n_hidden_nodes//2, activation = Activation(threshold_on), kernel_regularizer = regularizers.l1(0.01), kernel_constraint=cs.NonNeg())(inputs)
        #kernel_regularizer = regularizers.l1(0.005)
        h_ON = Dense(n_ON, activation = Activation(threshold_on), kernel_initializer = init, kernel_regularizer = regularizers.l1(0.005), kernel_constraint=cs.NonNeg(), use_bias = False)
        h_ON_act = h_ON(prev_layer)

        #h2 = Dense(n_hidden_nodes//2, activation = Activation(threshold_off), kernel_regularizer = regularizers.l1(0.01), kernel_constraint=cs.NonNeg())(inputs)
        h_OFF = Dense(n_OFF, activation = Activation(threshold_off), kernel_initializer = init, kernel_regularizer = regularizers.l1(0.005), kernel_constraint=cs.NonNeg(), use_bias = False)
        h_OFF_act = h_OFF(prev_layer)

        h_BP = Dense(n_BP, activation=Activation(bandpass),  kernel_initializer = init, kernel_regularizer = regularizers.l1(0.005),kernel_constraint=cs.NonNeg(),use_bias = False)
        h_BP_act = h_BP(prev_layer)

        h_IBP = Dense(n_IBP, activation=Activation(inverse_bandpass), kernel_initializer = init, kernel_regularizer = regularizers.l1(0.005), kernel_constraint=cs.NonNeg(), use_bias = False)
        h_IBP_act = h_IBP(prev_layer)

        t = time()
        if weights is not None:

            h_ON.set_weights([weights[i][:, 0:n_ON]])
            h_OFF.set_weights([weights[i][:, n_ON:n_ON+n_OFF]])
            h_BP.set_weights([weights[i][:, n_ON + n_OFF: n_ON + n_OFF + n_BP]])
            h_IBP.set_weights([weights[i][:, n_ON + n_OFF + n_BP:]])

        t1 = time() - t
        h = tf.concat([h_ON_act, h_OFF_act,h_BP_act,h_IBP_act], axis = 1)
        t_total += t1
        prev_layer = h

    #outputs = Dense(n_output_nodes,activation = Activation(threshold_on),  name = "outputs", kernel_regularizer = regularizers.l1(0.01), kernel_constraint=cs.NonNeg())(h)

    # use l1(0.005) for one hidden layer, l1(0.001) for two hidden layers
    #kernel_regularizer = regularizers.l1(0.001),
    outputs = Dense(n_out,activation = Activation(threshold_on),kernel_constraint=cs.NonNeg(), kernel_regularizer = regularizers.l1(0.001), use_bias = False) # dpnt put the init in here or it breaks
    outputs_act = outputs(prev_layer)

    if weights is not None:
        outputs.set_weights([weights[-1]])

    model = tf.keras.Model(inputs = inputs, outputs = outputs_act)
    # make model prunable
    #model = sparsity.prune_low_magnitude(model, **pruning_params)

    model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])

    layer_inputs = [layer.output for layer in model.layers][5:]


    input_model = models.Model(inputs=model.input, outputs=layer_inputs)
    return model, input_model


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(29, input_dim=2, activation='sigmoid'))
    model.add(Dense(20, input_dim=2, activation='sigmoid'))
    model.add(Dense(20, input_dim=2, activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

def get_minimal_model(model):
    weights = []

    for layer in model.layers:
        for weight in layer.get_weights(): # [0] is weights [1] is bias
            weights.append(weight)


    print('WEIGHTS:')
    for weight in weights:
        print(weight)
    #(h1_w, h2_w,output_w) =  weights

    hidden_weights, output_weights = weights[0:-1], weights[-1]

    # get the number of ON, OFF, BP and IBP
    counts = []
    for weight in hidden_weights:
        counts.append(weight.shape[1])


    print(counts)

    counts = np.array(counts).reshape(-1,4)

    #hidden_weights = [h1ON, h1OFF, h1BP, h2IBP, h2ON, h2OFF, ...]

    # concatenate the hidden weights for each layer

    concat_weights = []
    for i in range(len(hidden_weights)//4):
        concat_weight = np.hstack((hidden_weights[4*i:4*i+4]))
        concat_weights.append(concat_weight)

    concat_weights.append(output_weights)


    all_active = [[1]*n_in] # input nodes
    minimal_h_layer_sizes = np.zeros_like(counts)

    for i in range(len(concat_weights) -1):
        n_ON = counts[i][0]
        n_OFF = counts[i][1]
        n_BP = counts[i][2]
        n_IBP = counts[i][3]

        hidden_active = []
        weights = concat_weights[i]
        next_weights = concat_weights[i+1]

        n_nodes = len(next_weights)

        threshold = 1e-2
        for j in range(n_nodes):

            # if OFF or IBP can only remove nodes that output to no other nodes
            if n_ON <= j < n_ON + n_OFF: #OFF
                if np.all(next_weights[j, :] < threshold):
                    hidden_active.append(0)
                else:
                    hidden_active.append(1)
                    minimal_h_layer_sizes[i][1] += 1

            elif n_ON + n_OFF + n_BP <= j: #IBP
                if np.all(next_weights[j, :] < threshold):
                    hidden_active.append(0)
                else:
                    hidden_active.append(1)
                    minimal_h_layer_sizes[i][3] += 1

            # if On or BP we can also reomove nodes that have no input
            elif j < n_ON: #ON

                if np.all(weights[:, j] < threshold) or np.all(next_weights[j, :] < threshold):
                    hidden_active.append(0)
                else:
                    hidden_active.append(1)
                    minimal_h_layer_sizes[i][0] += 1

            else:  # BP

                if np.all(weights[:, j] < threshold) or np.all(next_weights[j, :] < threshold):
                    hidden_active.append(0)
                else:
                    hidden_active.append(1)
                    minimal_h_layer_sizes[i][2] += 1


        all_active.append(hidden_active)
    all_active.append([1]*n_out) #output_nodes



    all_reduced_weights = []
    for i in range(len(concat_weights)):
        inputs = all_active[i]
        outputs = all_active[i+1]

        reduced_weights = concat_weights[i][np.array(inputs) == 1, :]
        reduced_weights = reduced_weights[:, np.array(outputs) == 1]
        print('rw', reduced_weights.shape)
        all_reduced_weights.append(reduced_weights)

    minimal_layer_sizes = [n_in]
    minimal_layer_sizes.extend(minimal_h_layer_sizes)
    print('hidden layer sizes:', minimal_h_layer_sizes)
    minimal_layer_sizes.append(n_out)
    return minimal_layer_sizes, all_reduced_weights

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
    y = (np.sin(x*10)+1)/2 #> 0.5
    return x,y

def generate_circle(n):
    x = np.random.rand(n,2)
    centre = np.array([0.5, 0.5])



    dist = np.linalg.norm(x-centre)

    y = np.zeros((n,1))

    for i in range(len(y)):
        if np.linalg.norm(x[i] - centre) < 0.25:
            y[i] = 1


    return x,y

def generate_gut_data(n):
    n_each_category = n//3 # 4 distinct inputs to classify

    x = np.zeros((n, 2))
    y = np.zeros((n, 3))

    # set IBS data
    x[0:n_each_category//2, 0] = 1
    x[n_each_category//2:n_each_category, 0] = 0.5
    x[0:n_each_category, 1] = 0

    y[0:n_each_category, 2] = 1

    # set IBD data
    x[n_each_category:2*n_each_category, 0] = 0.5
    x[n_each_category:2*n_each_category, 1] = 0.5

    y[n_each_category:2*n_each_category, 1] = 1

    # set H data
    x[2*n_each_category:, 0] = 0
    x[2*n_each_category:, 1] = 0.5

    y[2*n_each_category:, 0] = 1

    # add some noise
    x += np.random.normal(0, 0.1, size = x.shape)

    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]
    return x,y


def generate_gut_data_new(n):
    x = np.random.random(size=(n,2))


    y = np.zeros((n, 3))

    for i in range(len(x)):

        if 0.33 < x[i, 0] and x[i,1] < 0.33:
            y[i] = 0
        elif 0.33 > x[i, 0] and x[i,1] < 0.33:
            y[i] = 1
        else:
            y[i] = 2
            '''DEFAULTS TO THIS CAT< CHANGE THIS'''

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

    model, input_model = create_network(layer_sizes)

    #model = baseline_model()

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



    #callbacks = [sparsity.UpdatePruningStep()]

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs)


    minimal_model = get_minimal_model(model)
    #minimal_model = 'none'

    return model, minimal_model, history


if __name__ == '__main__':
    working_dir = sys.argv[1]
    #sys.exit()
    n_in = 2
    n_out = 1

    n_ON = 10
    n_OFF = 10
    n_BP = 0
    n_IBP = 0

    n_h1 = [n_ON, n_OFF, n_BP, n_IBP] # [n_threshold, n_inverse_threshold, n_BP, n_IBP]
    n_h2 = [n_ON, n_OFF, n_BP, n_IBP]


    layer_sizes = [n_in, n_h1, n_out]

    training_func = generate_gut_data

    x_train, y_train = generate_logic_func(10000, [1,0,0,1])


    n_epochs = 100
    batch_size = 32
    num_train_samples = x_train.shape[0]
    end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * n_epochs

    print(end_step)

    '''
    pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                     final_sparsity=0.90,
                                                     begin_step=20000,
                                                     end_step=end_step,
                                                     frequency=5000)
    }
    '''
    pruning_params = {
        'pruning_schedule': sparsity.ConstantSparsity(
                                                     target_sparsity=0.7,
                                                     begin_step=2,
                                                     end_step=end_step,
                                                     frequency=1000)
    }
    '''
    plt.figure()
    plt.scatter(x_train,y_train)
    plt.title('training function')
    plt.savefig(working_dir + '/training_function.png', dpi = 300)

    model, minimal_model, history = train_network(layer_sizes, x_train, y_train, n_epochs)
    print(minimal_model)
    np.save(working_dir + '/minimal_model.npy', minimal_model)

    plt.figure()
    plt.plot(history.history['acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(working_dir + '/training_accuracy.png', dpi = 300)

    # loss_and_metrics = model.evaluate(X_test, Y_test)
    x_test, y_test = training_func(10000)

    ys = model.predict(x_test).reshape(10000, )

    # ys = np.argmax(ys, axis = 1)
    plt.figure()
    plt.scatter(x_test,ys)
    plt.title('trained model output')
    plt.savefig(working_dir + '/trained_model_output.png', dpi = 300)



    l_s = minimal_model[1]

    print('minimal layer sizes:', l_s)

    minimal_network, minimal_input_network = create_network(l_s, minimal_model[1])

    weights = []

    for layer in minimal_network.layers:
        for weight in layer.get_weights():  # [0] is weights [1] is bias
            weights.append(weight)

    print('WEIGHTS:')
    for weight in weights:
        print(weight)

    ys = minimal_network.predict(x_test).reshape(10000, )

    plt.figure()
    plt.scatter(x_test,ys)
    plt.title('trained model output')
    plt.savefig(working_dir + '/minimal_model_output.png', dpi = 300)



    '''



    plt.figure()
    plt.scatter(x_train[:,0], x_train[:,1], c = y_train.reshape(10000,))
    plt.title('training function')
    plt.savefig(working_dir + '/training_function.png', dpi = 300)
    for i in range(10):
        print(x_train[i])
    plt.show()

    model, minimal_model, history  = train_network(layer_sizes, x_train, y_train, n_epochs)


    print('WEIGHTS BEFORE MINIMISATION: ')
    for layer in model.layers:
        for weight in layer.get_weights():  # [0] is weights [1] is bias
            print(weight)

    pred_ys = model.predict(x_train)



    print(minimal_model)
    np.save(working_dir + '/minimal_model.npy', minimal_model, dpi = 300)

    print('WEIGHTS AFTER MINIMISATION: ')
    print('weights:', minimal_model[1])
    for weights in minimal_model[1]:
        print()
        print(weights)



    plt.figure()
    plt.plot(history.history['acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(working_dir + '/training_accuracy.png', dpi = 300)


    #loss_and_metrics = model.evaluate(X_test, Y_test)
    x_test, y_test = generate_logic_func(10000, [1,0,0,1])

    x_test = np.random.random(size = x_test.shape)

    ys = model.predict(x_test)

    #ys = np.argmax(ys, axis = 1)
    plt.figure()
    plt.scatter(x_test[:,0], x_test[:,1], c = ys.reshape(10000,))
    plt.title('trained model output')
    plt.savefig(working_dir + '/trained_model_output.png', dpi = 300)


    





    l_s = minimal_model[0]

    print('layer_sizes:', l_s)

    minimal_network, minimal_input_network = create_network(l_s, minimal_model[1])


    ys = minimal_network.predict(x_test)

    plt.figure()
    plt.scatter(x_test[:, 0], x_test[:, 1], c = ys.reshape(10000,))
    plt.title('trained model output')
    plt.savefig(working_dir + '/minimal_model_output.png', dpi = 300)

