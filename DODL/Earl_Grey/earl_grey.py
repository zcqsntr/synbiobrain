import numpy as np
import copy
from collections import OrderedDict
import sys
import queue as qu

from time import time
import math
import itertools
from exhaustive_search import *

def covers_from_blocks(blocks):
    # returns the start position and size of each cover
    indices = np.where(blocks[:, 0] == 1)[0]

    sizes = blocks[indices, 1]

    start_pos = [np.sum(blocks[:i, 1]) for i in indices]

    return np.vstack((start_pos, sizes)).T


def can_move(truth_table, frm, to):
    # checks if a state can be moved from a to b based on the AHL concentration constraints

    # if frm < to then to be able to move the state must be able to swap above all states between frm and to
    can_swap = True


    if frm < to:
        for i in range(frm + 1, to + 1):
            can_swap = can_swap and check_constraints(truth_table[i, :-1], truth_table[frm, :-1])
    elif to < frm:
        for i in range(to, frm):
            can_swap = can_swap and check_constraints(truth_table[frm, :-1], truth_table[i, :-1])

    return can_swap


def move(truth_table, frm, to):
    # moves a state from a to b by movin gthe states and shifting all states between a and b down or up one as required

    new_table = copy.deepcopy(truth_table)

    if frm < to:
        lower = new_table[0:frm, :]
        from_state = new_table[frm, :]
        middle = new_table[frm+1:to+1, :]
        upper = new_table[to+1:, :]

        table = np.vstack((lower, middle, from_state, upper))


    elif to < frm:
        lower = new_table[0:to, :]
        from_state = new_table[frm, :]
        middle = new_table[to:frm, :]
        upper = new_table[frm + 1:, :]

        table = np.vstack((lower, from_state, middle,  upper))
    else:
        table = new_table

    return table


def modify_covers(covers, frm, to):
    # updates the list of covers based on what move has happened


    new_covers = []

    if frm < to: # moved forwards
        for cover in covers:
            if cover[0] <= frm <= cover[0] + cover [1]:
                cover[1] -= 1
            elif cover[0] <= to <= cover[0] + cover [1]:
                cover[0] -=1 #start moves down
                cover[1] += 1
            new_covers.append(cover)

    elif to < frm:
        for cover in covers:
            if cover[0] <= frm <= cover[0] + cover[1]:
                cover[0] += 1
                cover[1] -= 1
            elif cover[0] <= to <= cover[0] + cover[1]:
                cover[1] += 1
            new_covers.append(cover)

    return np.array(new_covers)

def sort_truth_table(truth_table):
    '''
    sort according to the number of inputs activated instead of binary orders
    '''
    n_inputs = int(np.log2(truth_table.shape[0]))
    sum_inputs = np.sum(truth_table[:, :n_inputs], axis = 1)
    indices = np.argsort(sum_inputs)

    return truth_table[indices]

def count_output_blocks(output_blocks):

    '''
    counts the number of blocks of ones

    '''

    current = output_blocks[0][0]
    n_blocks = 0
    if current == 1:
        n_blocks += 1

    for input_group in output_blocks:
        for block in input_group:
            if block == 1 and current == 0:
                n_blocks += 1
            current = block

    return n_blocks


def rough_optimisation(truth_table):
    '''
    optimises based on matching he 0s and ones between the different numbers of inputs, gives a really good starting point
    for earl grey to do the final bit of opt
    '''
    n_inputs = int(np.log2(truth_table.shape[0]))


    truth_table = sort_truth_table(truth_table)

    # split into groups based on how many inputs are activated

    input_groups = []
    counter = 0
    for i in range(n_inputs+1):
        n_states = int(math.factorial(n_inputs)/(math.factorial(i)*math.factorial(n_inputs-i)))
        group = truth_table[counter:counter+n_states ]
        sorted_group = group[np.argsort([group[:, -1]])]
        input_groups.append(sorted_group[0])
        counter += n_states


    output_blocks = []
    flips = [] # whether or not each input group has been flipped
    flippable = [] #which input groups have 0s and 1s and therofore can be flipped
    for i,group in enumerate(input_groups):

        active_outputs = np.sum(group[:,-1])

        if active_outputs == 0:
            output_blocks.append([0])
        elif active_outputs == group.shape[0]:
            output_blocks.append([1])
        else:
            flips.append(False)
            output_blocks.append([0,1])
            flippable.append(i)

    # no go through each flip combination and find the one with the smallest number of blocks

    min_blocks = count_output_blocks(output_blocks)
    best_blocks = copy.deepcopy(output_blocks)
    best_flip_comb = flips

    flip_combs = list(itertools.product([False, True], repeat=len(flippable)))

    for flip_comb in flip_combs:
        for i, flip in enumerate(flip_comb):
            if flip:
                output_blocks[flippable[i]] = [1,0]
            else:
                output_blocks[flippable[i]] = [0,1]

        n_blocks = count_output_blocks(output_blocks)
        if n_blocks < min_blocks:

            best_blocks = copy.deepcopy(output_blocks)
            best_flip_comb = list(copy.deepcopy(flip_comb))
            min_blocks = n_blocks


    # assemble the new truth table base don which input groups are flipped

    new_input_groups = copy.deepcopy(input_groups)

    for i, ind in enumerate(flippable):
        if best_flip_comb[i]:
            new_input_groups[ind] = np.flip(new_input_groups[ind], axis = 0)

    return np.vstack(new_input_groups), min_blocks




def earl_grey(outputs):
    n_inputs = int(np.log2(outputs.size))
    truth_table = create_truth_table(outputs)
    #truth_table,_ = rough_optimisation(truth_table)

    current_table = copy.deepcopy(truth_table)
    current_table = truth_table

    will_exit = True


    finished = False

    while not finished:

        finished = True
        blocks = get_blocks(current_table)[0]

        covers = covers_from_blocks(blocks)

        # each block of ones is a cover, try and maximise each cover in turn, starting from largest cover
        cov_sort = np.argsort(covers[:, 1])  # this will bias towards states in the middle, probably not what we want


        test_table = copy.deepcopy(current_table)

        for index in cov_sort:
            # get smallest cover
            smallest_cover = covers[index]


            small_start = smallest_cover[0]
            small_end = smallest_cover[0] + smallest_cover[1]


            # try and eliminate smallest cover by puttin gones into the covers on either side


            if index > 0:
                lower_cover = covers[index -1]
            else:
                lower_cover = None

            if index < len(covers) - 1:
                higher_cover = covers[index+ 1]
            else:
                higher_cover = None

            test_table = copy.deepcopy(current_table)


            if lower_cover is not None:

                # go through oes and put as many into the lower cover as possible
                for i, state in enumerate(range(small_start, small_end)):

                    if can_move(test_table, state, lower_cover[0] + lower_cover[1] + i):
                        test_table = move(test_table, state, lower_cover[0] + lower_cover[1] + i)
                        smallest_cover[0] += 1
                        smallest_cover[1] -=1

            small_start = smallest_cover[0]
            small_end = smallest_cover[0] + smallest_cover[1]

            if higher_cover is not None:
                for i, state in enumerate(range(small_start, small_end)[::-1]):
                    if can_move(test_table, state, higher_cover[0] - i - 1):
                        test_table = move(test_table, state, higher_cover[0] - i - 1)
                        smallest_cover[1] -= 1

            if smallest_cover[1] == 0:
                current_table = test_table
                finished = False
                break

    return current_table



if __name__ == '__main__':
    outputs = np.array([[0,0,0,1,0,0,1,1]]) #dependant on one input
    outputs = np.array([[0,0,1,1,1,1,0,1]])

    outputs = np.array([[1,0,1,0,1,1,1,0]])
    #outputs = np.array([[1,1,0,0,1,1,0,1]]) #0xCD

    outputs = np.array([0,0,0,1,1,0,1,1])


    #outputs = np.array([[0,0,1,1,0,1,1,1,0,0,1,1,0,1,1,1]]) #threshold

    #priority encoder
    #outputs = np.array([[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]])
    #outputs = np.array([[0,0,1,1,0,0,0,0,1,1,1,1,1,1,1,1]])
    #outputs = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])


    n_inputs = int(np.log2(outputs.size))


    t = time()
    best_table = earl_grey(outputs)
    print(best_table)
    print('time:', time()-t)

    #print('result:', current_table)

    valid = True
    for i in range(2**n_inputs):
        for j in range(2**n_inputs):
            if i < j:
                valid = valid and check_constraints(best_table[i, :-1], best_table[j, :-1])
            if j < i:
                valid = valid and check_constraints(best_table[j, :-1], best_table[i, :-1])


    print('valid:', valid)

    activations = get_activations(best_table, n_inputs)

    print('NODES: ')
    for act in activations.keys():
        for state_mapping in activations[act]:


            simplified_state_mapping = simplify(state_mapping, n_inputs)
            print(act)
            for s in simplified_state_mapping:
                print(s)














