import numpy as np
import copy
from collections import OrderedDict
import sys
import queue as qu


from earl_grey import *

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






outputs = np.array([[0,1,0,1,0,1,0,1]]) #dependant on one input
outputs = np.array([[1,1,0,0,1,1,0,1]]) #0xCD

n_inputs = 4
outputs = np.array([[0,0,1,1,0,1,1,1,0,0,1,1,0,1,1,1]]) #threshold

#priority encoder
#outputs = np.array([[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]])
#outputs = np.array([[0,0,1,1,0,0,0,0,1,1,1,1,1,1,1,1]])
#outputs = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

truth_table = create_truth_table(outputs)
discovered_tables = {hash_table(truth_table)} # use a set for this

current_table = copy.deepcopy(truth_table)
print(current_table)
n_steps = 0
while True:
    n_steps += 1
    blocks = get_blocks(current_table)[0]

    covers = covers_from_blocks(blocks)

    # each block of ones is a cover, try and maximise each cover in turn, starting from largest cover
    cov_sort = np.argsort(covers[:, 1])[::-1] # this will bias towards states in the middle, probably not what we want


    new_table = current_table

    will_exit = True


    index = 0
    while index < len(covers): # go through covers largest to smallest
        has_moved = False
        print(covers)
        cover = covers[index]

        if cover[1] < 0:
            print('COVER NEGATIVE SIZE ')
            sys.exit()
        if cover[1] == 0: # if cover has been modified to be empty
            continue

        cov_start, cov_size = cover
        cov_end = cov_start + cov_size - 1




        for i in range(cov_start - 1, 0, -1): # move ones below block to block start, iterate backwards so we can change order of nodes without messing the loop up
            # recalculate in case a cover changed


            if new_table[i, -1] == 1 and can_move(new_table, i, cov_start-1):
                new_table = move(new_table, i, cov_start-1)
                print('COVERS1:')
                print(i, cov_start - 1)
                print('cov:', covers)
                covers = modify_covers(covers, i, cov_start - 1)
                print('nc:', covers)
                print('----')
                has_moved = True



        for i in range(cov_end + 1, outputs.size): # move ones aove block to end

            # recalculate in case a cover changed

            if new_table[i, - 1] == 1 and can_move(new_table, i, cov_end + 1):
                new_table = move(new_table, i, cov_end + 1)
                print('COVERS2:')
                print(i, cov_end + 1)
                print('cov: ', covers)
                covers = modify_covers(covers, i, cov_end + 1)
                print('nc:',covers)
                print('-----')
                has_moved = True



        print(new_table)
        if hash_table(new_table) not in discovered_tables:
            will_exit = False
            print('moved')
            print()
            discovered_tables.add(hash_table(new_table))
            current_table = new_table
            break
        else:
            index += 1
            print('not moved')


    if will_exit:
        print('exited in n steps: ', n_steps)

        print(current_table)
        sys.exit()







