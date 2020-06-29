from mpl_toolkits.mplot3d import axes3d
import numpy as np
import copy
from functools import total_ordering
import math
import matplotlib.pyplot as plt
import sys

from time import time

# grid defined by number of layers, number of nodes in each layer of each activation function and positions of nodes


# generate population of random grids

# assign weights based on positions and simulate a NN using Keras adn get a classification accuracy

# rank nodes best of accuracy and recombine from the population of the good nodes
def hill(conc, n, kd, min, max):
    return min + (max-min)*(conc**n/(kd**n + conc**n))


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def threshold_on(x):
    return sigmoid(x - 5)

def threshold_on(x):
    #x: nmol/mm^2
    n = 1.75
    kd = 17.83
    min = 4230
    max = 54096
    S = 3.8e-3  # nmol/h
    C = 3e-6  # converesion from nM to nmol/mm^2
    min = 0
    return hill(x/C, n, kd, min, max)/max *S


def threshold_off(x):
    return 1 - sigmoid(x - 5)


def threshold_off(x):
    # x: nmol/mm^2
    n = 1.75
    kd = 17.83
    min = 4230
    max = 54096
    S = 3.8e-3  # nmol/h
    C = 3e-6  # converesion from nM to nmol/mm^2
    min = 0
    return (1 - hill(x/C, n, kd, min, max)/max)*S


def bandpass(x):
    return sigmoid(x - 5) - sigmoid(x - 15)


def inverse_bandpass(x):
    return -sigmoid(x - 5) + sigmoid(x - 15) + 1



def sin(n):
    x = np.random.rand(n, 1)
    y = (np.sin(x*10)+1)/2 > 0.5
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

    return x, y.reshape(-1, 1)


@total_ordering
class Node:

    def __init__(self, activation, position):
        self.activation = activation
        self.position = position

    def get_activation(self):
        return self.activation

    def set_activation(self, activation):
        self.activation = activation #string

    def get_position(self):
        return self.position

    # for sorting nodes based on actiation function
    def __lt__(self, other):
        order = ['ON', 'OFF', 'BP', 'IBP']
        return order.index(self.activation) < order.index(other.activation)

    def __eq__(self, other):
        return self.activation == other.activation


class Grid:
    '''
    Handles the spatial configuration and evolution of a grid and conversion into a NN to get its fitness

    '''
    def __init__(self, size, layer_sizes, node_radius):
        self.prod_rate = 3.8e-3  # nmol/h
        #self.prod_rate = 0.5
        self.size = np.array(size) #(max_x, max_y)
        self.node_radius = node_radius
        self.layer_sizes = layer_sizes
        #set up nodes
        self.nodes = []
        input = layer_sizes[0]
        output = layer_sizes[-1]
        hidden_layers = layer_sizes[1:-1]
        self.diffusion_data = np.load('./mathematica_data.npy')

        self.nodes.append([Node('ON', np.random.rand(2) * self.size) for i in range(input)])
        for l in hidden_layers:
            h_l = []
            h_l.extend([Node('ON', np.random.rand(2) * self.size) for i in range(l[0])])
            h_l.extend([Node('OFF', np.random.rand(2) * self.size) for i in range(l[1])])
            h_l.extend([Node('BP', np.random.rand(2) * self.size) for i in range(l[2])])
            h_l.extend([Node('IBP', np.random.rand(2) * self.size) for i in range(l[3])])
            self.nodes.append(h_l)

        self.nodes.append([Node('ON', np.random.rand(2) * self.size) for i in range(output)])

    def get_distance(self,p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def AHL_func(self,r):
        ''' for now hardcode 10 hours between nodes, linear approimation between the two cosest points in the data'''

        prop_time = 10
        min = self.diffusion_data[10, int(math.floor(r))]
        max = self.diffusion_data[10, int(math.ceil(r))]
        if min==max: return min
        grad = (max-min)/(math.ceil(r) - math.floor(r))

        return min + grad*(r - math.floor(r))

    def get_nodes(self):
        return self.nodes

    def reproduce(self, other_parent):
        child1 = copy.deepcopy(self)
        child2 = copy.deepcopy(other_parent)

        for i in range(len(self.get_nodes())):
            for j in range(len(self.get_nodes()[i])):  # randomly swap some nodes
                if np.random.rand() < 0.5:
                    try: # number of nodes might differ between the grids
                        child1.nodes[i][j] = other_parent.nodes[i][j] #WRITE SETTER FOR THIS
                    except:
                        child1.nodes[i].append(self.nodes[i][j])

                if np.random.rand()  < 0.5:
                    try:
                        child2.nodes[i][j] = self.nodes[i][j]
                    except:
                        child2.nodes[i].append(self.nodes[i][j])

        return child1, child2


    def reproduce1(self, other_parent):
        child1 = copy.deepcopy(self)
        child2 = copy.deepcopy(other_parent)

        child1.mutate()
        child2.mutate()
        return child1, child2

    def mutate(self):

        for i in range(len(self.get_nodes())):
            for j in range(len(self.get_nodes()[i])):  # randomly mutate the position of each node


                # randomly change node type
                if np.random.rand() < 0.1 and 1 < i < len(self.get_nodes()) - 1: # onyl for hidden layers
                    # node_type = np.random.choice(['ON', 'OFF', 'BP', 'IBP'])
                    type_index = ['ON', 'OFF', 'BP', 'IBP'].index(self.nodes[i][j].activation)
                    self.layer_sizes[i][type_index] -= 1


                    node_type = np.random.choice(['ON', 'OFF'])
                    self.nodes[i][j].set_activation(node_type)

                    type_index = ['ON', 'OFF', 'BP', 'IBP'].index(node_type)
                    self.layer_sizes[i][type_index] += 1




                if np.random.rand() < 0.2:
                    rand_x = (np.random.rand() - 0.5)*5
                    if 0 < self.nodes[i][j].position[0] + rand_x < 30:
                        self.nodes[i][j].position[0] += rand_x

                if np.random.rand() < 0.2:
                    rand_y = (np.random.rand() - 0.5)*5
                    if 0 < self.nodes[i][j].position[1] + rand_y < 30:
                        self.nodes[i][j].position[1] += rand_y

            # randomly remove or add nodes
            if np.random.rand()< 0.1 and 1 < i < len(self.get_nodes()) - 1: # onyl for hidden layers

                if len(self.get_nodes()[i]) > 0:
                    node = self.nodes[i].pop(np.random.choice(range(len(self.get_nodes()[i]))))

                    # update layer sizes
                    type_index = ['ON', 'OFF', 'BP', 'IBP'].index(node.activation)

                    self.layer_sizes[i][type_index] -= 1

            if np.random.rand()< 0.1 and 0 < i < len(self.get_nodes()) - 1: # onyl for hidden layers

                #node_type = np.random.choice(['ON', 'OFF', 'BP', 'IBP'])
                node_type = np.random.choice(['ON', 'OFF'])
                position = np.random.rand(2) * self.size

                self.nodes[i].append(Node(node_type, position))

                # update layer sizes
                type_index = ['ON', 'OFF', 'BP', 'IBP'].index(node_type)

                self.layer_sizes[i][type_index] += 1
                self.move_concurrent_nodes()


    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    def move_concurrent_nodes(self):
        # DEBUG THIS
        # checks to see if any nodes on top of each other, if so moves them
        nodes = self.get_nodes()
        for i in range(len(nodes)):
            for j in range(len(nodes[i])):
                pos = nodes[i][j].get_position()

                for k in range(len(nodes)):
                    for l in range(len(nodes[k])):
                        pos2 = nodes[k][l].get_position()
                        dir = np.array(pos) - np.array(pos2)
                        mag = np.linalg.norm(dir)

                        if (mag <= 2 * self.node_radius) and ((not i == k) or (not j == l)):

                            if mag == 0:
                                x, y = self.pol2cart(2 * self.node_radius, np.random.rand() * 2 * math.pi)
                            else:
                                unit_vec = dir / mag  # direction to move node

                                translation_mag = (self.node_radius * 2 - mag)
                                self.nodes[i][j].position += unit_vec[0] * translation_mag
                                self.nodes[i][j].position += unit_vec[1] * translation_mag

    def get_weights(self):

        # first rearrange nodes so that they are in the order (ON,OFF, BP, IBP) expected by the NN
        for i in range(1, len(self.nodes)-1): #for each hidden layer
            self.nodes[i].sort()


        hidden_weights = []

        all_weights = []

        for i in range(len(self.nodes) - 1):
            layer_weights = []
            positions = [node.get_position() for node in self.nodes[i]]
            next_positions = [node.get_position() for node in self.nodes[i+1]]

            for pos in positions:
                weights = []
                for next_pos in next_positions:
                    r = self.get_distance(pos, next_pos)

                    weights.append(self.AHL_func(r))

                layer_weights.append(weights)

            all_weights.append(np.array(layer_weights))

        return all_weights

    def create_network(self):
        return create_network(self.layer_sizes, self.get_weights())[0]

    def get_fitness(self, inputs, targets):

        predicted = self.predict(inputs)

        mse = self.get_mse(predicted, targets)
        accuracy = self.get_accuracy(predicted, targets)
        return mse/accuracy # this could also include terms for number of nodes or size of max layer etc

    def get_mse(self, predicted, targets):

        mse = np.sum((predicted.reshape(-1,) - targets)**2/len(targets))

        return mse

    def get_accuracy(self, predicted, targets): # this will change depending on the form of output

        #pred_classes = np.argmax(predicted, axis=1) #  one hot classification
        # targets = np.argmax(targets, axis=1) #one hot classification

        pred_classes = np.round(predicted) #two class classification with one output


        correct = pred_classes.reshape(-1,1) == targets


        return np.sum(correct)/len(correct)

    def plot(self, iter, working_dir):

        plt.figure()

        # plot nodes
        r = self.node_radius
        for node in self.get_nodes()[0]:  # input
            circle = plt.Circle((node.get_position()[0], node.get_position()[1]), radius=r, fc='g',
                                label='(' + str(round(node.get_position()[0], 1)) + ',' + str(round(node.get_position()[1], 1)) + ')')
            plt.gca().add_patch(circle)


        hidden_layers = self.get_nodes()[1:-1]
        colours = {'ON':'blue', 'OFF':'aqua', 'BP':'purple', 'IBP':'magenta'}

        for layer in hidden_layers:
            for node in layer:
                circle = plt.Circle((node.get_position()[0], node.get_position()[1]), radius=r, fc=colours[node.get_activation()],
                                    label='(' + str(round(node.get_position()[0], 1)) + ',' + str(round(node.get_position()[1], 1)) + ')')
                plt.gca().add_patch(circle)


        for node in self.nodes[-1]:  # output
            circle = plt.Circle((node.get_position()[0], node.get_position()[1]), radius=r, fc='red',
                                label='(' + str(round(node.get_position()[0], 1)) + ',' + str(round(node.get_position()[1], 1)) + ')')
            plt.gca().add_patch(circle)

        # plot weights

        plt.legend()
        plt.axis('scaled')
        plt.savefig(working_dir + '/' + str(iter) + '.png')

    def predict(self, inputs):

        inputs = copy.deepcopy(inputs)*self.prod_rate
        hidden_layers = self.layer_sizes[1:-1]

        output_layer = self.layer_sizes[-1]

        weights = self.get_weights()
        #print(weights)
        current_layer = inputs
        for i in range(len(hidden_layers)):

            pre_act = np.matmul(current_layer, weights[i])


            n_ON, n_OFF, n_BP, n_IBP = hidden_layers[i]


            post_act = [threshold_on(pre_act[:, 0:n_ON]), threshold_off(pre_act[:, n_ON:n_ON + n_OFF]),
                        bandpass(pre_act[:, n_ON + n_OFF: n_ON + n_OFF + n_BP]),
                        inverse_bandpass(pre_act[:,  n_ON + n_OFF + n_BP:])]


            post_act = np.hstack(post_act)
            current_layer = post_act


        output = threshold_on(np.matmul(current_layer, weights[-1]))/self.prod_rate
        return output


class GridPopulation:
    '''
    Handles the evolution of a population of grids using the Grid and Model classes

    '''

    def __init__(self, grid_size, n_grids, layer_sizes, node_radius):
        # maybe have layer sizes as a range

        self.population = []

        for i in range(n_grids):
            # put random choices of each activation

            l_s = copy.deepcopy(layer_sizes)

            if type(layer_sizes[1]) is int:

                for j in range(1, len(l_s)-1):
                    nodes_left = layer_sizes[j]

                    activations = [0,0,0,0]

                    while nodes_left > 0:

                        n_nodes = np.random.randint(nodes_left+1)
                        #activations[np.random.randint(4)] = n_nodes # random activations
                        activations[np.random.randint(2)] = n_nodes # threshold or inverse threshold
                        #activations[0] = n_nodes # all threshold
                        nodes_left -= n_nodes


                    l_s[j] = activations

            self.population.append(Grid(grid_size, l_s, node_radius))

        [grid.move_concurrent_nodes() for grid in self.population]
        self.n_grids = n_grids

    def remainder_stochastic_sampling(self, fitnesses):

        norm_f = np.abs(fitnesses - np.max(fitnesses)) # now high fitness is good

        norm_f /= np.mean(norm_f)
        #norm_f += 0.0001
        good_grids = []

        for i,f in enumerate(norm_f):
            good_grids.extend([self.population[i]]*int(f//1))

            if np.random.rand() < f%1:
                good_grids.append(self.population[i])


        return good_grids

        # add the ineger normed fitness

    def naive_selection(self,fitnesses,proportion):
        "ranks the pop and choose the best prop of the popuulation"

        indices = np.argsort(fitnesses)

        reproduce = indices[:len(self.population) // 2]

        good_grids = []

        for i in reproduce:
            good_grids.append(self.population[i])

        return good_grids

    def mutation_recombination(self, good_grids):

        children = []



        for i, grid in enumerate(good_grids):

            if np.random.rand() < 0.3:  # this grid is going to recombine
                grid2 = good_grids[np.random.choice(range(len(good_grids)))]
                child1, child2 = grid.reproduce(grid2)

                good_grids[i] = child1
                children.append(child1)
                children.append(child2)

            if np.random.rand() < 0.5:  # this grid is going to mutate
                good_grids[i].mutate()

        #good_grids.extend(children) # for nbaive selection

        return good_grids


    def evolve_population(self, target_func, n_gens):

        n = 1000
        inputs, targets = target_func(n)

        print(inputs.shape)
        print(targets.shape)
        plt.figure()
        plt.scatter(inputs[:, 0], inputs[:, 1], c=targets.reshape(n, ))
        #plt.scatter(inputs, targets)
        plt.savefig(working_dir + '/training_func.png')

        best_fitness = -1 # minimum fitness is 0
        for gen in range(n_gens):

            fitnesses = [grid.get_fitness(inputs, targets) for grid in self.population]

            indices = np.argsort(fitnesses)
            best_grid = copy.deepcopy(self.population[indices[0]])

            print(len(fitnesses))
            if gen == 0 or fitnesses[indices[0]] < best_fitness:
                print()
                print('new best fitness found: ', fitnesses[indices[0]])
                best_grid = copy.deepcopy(self.population[indices[0]])
                print('best grid accuracy: ', best_grid.get_accuracy(best_grid.predict(inputs), targets))
                print('layer sizes: ', best_grid.layer_sizes)



                best_fitness = fitnesses[indices[0]]
                #print(np.array(fitnesses)[indices[:20]])

            if gen % 10 == 0:

                print()
                print('------------------------------------------------------------')
                print(gen)
                print('best fitness: %.15f ' % best_fitness)
                print('best grid accuracy: ', best_grid.get_accuracy(best_grid.predict(inputs), targets))
                print('population size:', len(self.population))
                print('average fitness: ', np.mean(fitnesses))
                print('average of twenty best fitnesses: ', np.mean(np.array(fitnesses)[indices[:20]]))
                print('min fitness: ', np.min(fitnesses))
                print('layer sizes: ', best_grid.layer_sizes)
                np.save(working_dir + '/best_grid.npy', best_grid)

                #print(np.array(fitnesses)[indices[:20]])

                self.population[indices[0]].plot('position' + str(gen), working_dir)
                inp, t = target_func(n)

                fig = plt.figure()
                predictions = best_grid.predict(inputs).reshape(n, )

                ax = fig.add_subplot(111, projection='3d')

                ax.scatter(inputs[:, 0], inputs[:, 1], predictions)

                #plt.scatter(inp, best_grid.predict(inp))
                plt.savefig(working_dir + '/learned_func' + str(gen) + '.png')


            good_grids = self.remainder_stochastic_sampling(fitnesses)
            good_grids = self.mutation_recombination(good_grids)


            good_grids.append(best_grid)  # keep best grid
            [grid.move_concurrent_nodes() for grid in good_grids]


            self.population = good_grids

        return best_grid, best_fitness


if __name__ == '__main__':
    working_dir = sys.argv[1]

    #node_radius  = 0.01 #cm used for one layer XNOR
    node_radius = 0.5
    n_gens = 200
    initial_pop = 10000
    grid_size = [10,10]


    training_func = lambda n : generate_logic_func(n, [1,0,0,1]) #XNOR
    #training_func = generate_circle
    #training_func = sin
    layer_sizes = [2, 4,  1]  # can either specify in [2,4,4,1] or 2, [1,2,0,1], [2,0,2,0], 1 form
    #layer_sizes = [1, [0,3,0,0], [0,0,2,0], 1]
    t = time()
    grid_population = GridPopulation(grid_size, initial_pop, layer_sizes, node_radius)
    prod_rate = grid_population.population[0].prod_rate
    max_weight = grid_population.population[0].AHL_func(node_radius*2)
    print('max weight: ',max_weight)
    print('proportional activation at max weight: ', threshold_on(prod_rate*max_weight)/prod_rate)
    print('time to init: ', time() - t)

    best_grid, fitness = grid_population.evolve_population(training_func, n_gens)
    print('best fitness: ', fitness)

    inputs, targets = training_func(1000)
    print('best grid accuracy: ', best_grid.get_accuracy(best_grid.predict(inputs), targets))
    np.save(working_dir + '/best_grid.npy', best_grid)

    best_grid.plot('final', working_dir)

    n = 1000

    inputs, targets = training_func(n)
    plt.figure()
    plt.figure()
    plt.scatter(inputs[:, 0], inputs[:, 1], c=best_grid.predict(inputs).reshape(n, ))
    #plt.scatter(inputs, best_grid.predict(targets))
    plt.savefig(working_dir + '/final_grid_output.png')
