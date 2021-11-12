from earl_grey import *




n_inputs = 5
all_outputs = list(map(np.array,list(itertools.product([0, 1], repeat = 2**n_inputs))))

count_dict = {-1:0,0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
single_counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
singles = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
sing = []

max_blocks = -1
for outputs in all_outputs:

    #outputs = np.array([[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]])
    truth_table = create_truth_table(outputs)
    truth_table, mb = rough_optimisation(truth_table)
    if mb> max_blocks:
        max_blocks = mb


print(max_blocks)