import sys
sys.path.append('/Users/neythen/Desktop/Projects/DODL/Macchiato')
import itertools
import numpy as np

from earl_grey import *
from macchiato import macchiato, graph_search, macchiato_v2
from time import time
#from exhaustive_search import *

n_inputs = 3
graph = False
all_outputs = list(map(np.array,list(itertools.product([0, 1], repeat = 2**n_inputs))))

count_dict = {-1:0,0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
single_counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
singles = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
sing = []

ti = time()
for i,outputs in enumerate(all_outputs):
    print(i)
    #best_table= earl_grey(outputs)
    print(outputs)
    best_tables= macchiato_v2(np.array(outputs)) #{-1: 0, 0: 1, 1: 151, 2: 104, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
    #activations = get_activations(best_table, n_inputs, allowed_acts=['TH', 'IT', 'BP', 'IB'])

    #best_table = graph_search(outputs) #{-1: 0, 0: 2, 1: 150, 2: 98, 3: 6, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

    #activations = get_activations(best_table, n_inputs, allowed_acts=['TH', 'IT', 'BP', 'IB'])

    #best_tables = macchiato_v2(outputs, max_queue_size=0)


    #print(activations)
    if graph:
        try:
            n_colonies = 0
            for k in activations.keys():
                n_colonies += len(activations[k])

            count_dict[n_colonies] += 1

            if n_colonies == 3:
                print()
                print('start:', outputs)
                print('result:', best_table[:, -1])

        except:

            count_dict[activations] += 1
    else:
        n_colonies = len(best_tables)
        count_dict[n_colonies] += 1
        '''
        if n_colonies == 3:
            print()
            print('start:', outputs)
            for t in best_tables:
                print(t)
            print()
            
        '''

print('time:', time()-ti)
np.save('single_reciver_logic_gates.npy', sing)
print(count_dict)
