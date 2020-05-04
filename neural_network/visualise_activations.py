


from train_network import create_network, generate_logic_func



import sys
import os
from typing import Any, Union

import numpy as np
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

from tensorflow_model_optimization.sparsity import keras as sparsity


if __name__ == '__main__':
    working_dir = sys.argv[1]

    n_in = 2
    n_out = 1

    n_ON = 5
    n_OFF = 5
    n_BP = 0
    n_IBP = 0


    minimal_model = np.load(working_dir + '/minimal_model.npy', allow_pickle=True)


    n_h1 = [sum(minimal_model[0][0][0:n_ON]), sum(minimal_model[0][0][n_ON:n_ON + n_OFF]),
             sum(minimal_model[0][0][n_ON + n_OFF:n_ON + n_OFF + n_BP]), sum(minimal_model[0][0][n_ON + n_OFF + n_BP:])]
    n_h2 = [sum(minimal_model[0][1][0:n_ON]), sum(minimal_model[0][1][n_ON:n_ON + n_OFF]),
             sum(minimal_model[0][1][n_ON + n_OFF:n_ON + n_OFF + n_BP]), sum(minimal_model[0][1][n_ON + n_OFF + n_BP:])]


    l_s = [n_in, n_h1, n_h2, n_out]

    minimal_network, minimal_input_network = create_network(l_s, minimal_model[1])

    x_test, y_test = generate_logic_func(10000, [1, 0, 0, 1])

    x_test = np.array([[0,0], [0,1], [1,0], [1,1]])


    for i in minimal_input_network.predict(x_test):
        print(i)

    print()

    for layer in minimal_network.layers:
        print(layer.get_weights())

