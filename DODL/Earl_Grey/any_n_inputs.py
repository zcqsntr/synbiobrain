
import itertools
import numpy as np
from earl_grey import *
from exhaustive_search import *

n_inputs = 3
all_outputs = list(map(np.array,list(itertools.product([0, 1], repeat = 2**n_inputs))))

count_dict = {-1:0,0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
single_counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
singles = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
sing = []
for outputs in all_outputs:
    best_table= earl_grey(outputs)


    # analyse best table to create a grid of bacterial populations
    # IT only allowed at low end, threshold only allowed at high end
    activations = get_activations(best_table, n_inputs, allowed_acts = [ 'IT', 'BP', 'IB'])
    #print(activations)

    try:
        n_colonies = 0
        for k in activations.keys():
            n_colonies += len(activations[k])

        count_dict[n_colonies] += 1

        if n_colonies == 4:
            print()
            print(best_table)

    except:

        count_dict[activations] += 1


print(count_dict)

string_sing = []

for s in sing:
    new_s = []
    for IS in s:
        string_IS = ''
        for i in range(3):
            string_IS += str(IS[i])
        new_s.append(string_IS)
    string_sing.append(new_s)


satistfies_first_constraint = []

for mp in string_sing:
    if mp.index('001') < mp.index('011') and mp.index('001') < mp.index('101'): #if satisfies first constraint
        satistfies_first_constraint.append(mp)
    else:
        print(mp.index('001') , mp.index('011') , mp.index('001') , mp.index('101'), mp)

print('Number of states satisfying first constraint: ', len(satistfies_first_constraint))


sat_first_and_second = []

for mp in satistfies_first_constraint:
    if mp.index('010') < mp.index('011') and mp.index('010') < mp.index('110'): #if satisfies second constraint
        sat_first_and_second.append(mp)
    else:
        print(mp)

print('Number of states satisfying first and second constraint: ', len(sat_first_and_second))


sat_all_constraints = []

for mp in sat_first_and_second:
    if mp.index('100') < mp.index('101') and mp.index('100') < mp.index('110'): #if satisfies second constraint
        sat_all_constraints.append(mp)
    else:
        print(mp)

print('Number of states satisfying all constraints: ', len(sat_all_constraints))

print(count_dict)
print(single_counts)


new = []
new.extend([tuple(singles[2][i]) for i in range(len(singles[2]))])
new.extend([tuple(singles[6][i]) for i in range(len(singles[6]))])

print()

print()

for i in range(len(singles[2])):
    zs = np.where(singles[2][i] == 0)
    os = np.where(singles[2][i] == 1)

    singles[2][i][zs] = 1
    singles[2][i][os] = 0

new = []
new.extend([tuple(singles[2][i]) for i in range(len(singles[2]))])
new.extend([tuple(singles[6][i]) for i in range(len(singles[6]))])

print(len(set(new)))
#print(set(new))