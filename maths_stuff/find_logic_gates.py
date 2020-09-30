import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def threshold(x, t):
    if x > t:
        return 1
    else:
        return 0

def inverse_threshold(x, t):
    if x > t:
        return 0
    else:
        return 1

def bandpass(x, t1, t2):
    if t1 < x < t2:
        return 1
    else:
        return 0

def inverse_bandpass(x, t1, t2):
    if t1 < x < t2:
        return 0
    else:
        return 1

def get_activation_map(AHL, activation_function):
    activation_map = np.zeros_like(AHL)
    for i, a in enumerate(AHL):
        activation_map[i] = activation_function(a)
    return activation_map

t = 8e-8
t1 = 7e-8
t2 = 5e-7
nx = 200
ny = 200

i = 6
j = 18

threshold_acts = []
inverse_threshold_acts = []
bandpass_acts = []
inverse_bandpass_acts = []

for k in range(3):
    AHL = np.load('/home/neythen/Desktop/Projects/synbiobrain/maths_stuff/verify_comb_maths_results/' + str(i) + ',' + str(j) + ': ' + str(k) + '.npy')

    print(AHL.shape)
    '''
    print('max: ', np.max(AHL))
    print('min: ', np.min(AHL))
    print('mean: ', np.mean(AHL))
    print('median: ', np.median(AHL))
    print()
    '''
    threshold_act = get_activation_map(AHL[-1, :], lambda x: threshold(x, t))



    inverse_threshold_act = get_activation_map(AHL[-1, :], lambda x: inverse_threshold(x, t))
    bandpass_act = get_activation_map(AHL[-1, :], lambda x: bandpass(x, t1, t2))
    inverse_bandpass_act = get_activation_map(AHL[-1, :], lambda x: inverse_bandpass(x, t1, t2))

    threshold_acts.append(threshold_act)
    inverse_threshold_acts.append(inverse_threshold_act)
    bandpass_acts.append(bandpass_act)
    inverse_bandpass_acts.append(inverse_bandpass_act)



    fig = plt.figure()
    plot = plt.imshow(np.zeros_like(AHL[-1, :]).reshape(nx, ny), cmap = 'plasma')
    plt.title('Threshold in state ' + str(k))
    plt.savefig('Threshold in state ' + str(k) + '.pdf')





    '''
    cmap = mpl.colors.ListedColormap(['k', 'g'])
    bounds = [0., 0.5, 1.]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    #fig, axs = plt.subplots(1, 3)
    fig = plt.figure()
    plot = plt.imshow(threshold_act.reshape(nx, ny), cmap = cmap)
    plt.title('Threshold in state ' + str(k))
    plt.savefig('Threshold in state ' + str(k) + '.pdf')

    fig = plt.figure()
    plot = plt.imshow(inverse_threshold_act.reshape(nx, ny), cmap = cmap)
    plt.title('Inverse threshold in state ' + str(k))
    plt.savefig('Inverse threshold in state ' + str(k) + '.pdf')

    fig = plt.figure()
    plot = plt.imshow(bandpass_act.reshape(nx, ny), cmap = cmap)
    plt.title('Bandpass in state ' + str(k))
    plt.savefig('Bandpass in state ' + str(k) + '.pdf')

    fig = plt.figure()
    plot = plt.imshow(inverse_bandpass_act.reshape(nx, ny), cmap = cmap)
    plt.title('Inverse bandpass in state ' + str(k))
    plt.savefig('Inverse bandpass in state ' + str(k) + '.pdf')

    plt.show()
    '''

plt.show()

threshold_acts = [np.zeros_like(AHL[-1])] + threshold_acts
inverse_threshold_acts = [np.ones_like(AHL[-1])] + inverse_threshold_acts
bandpass_acts = [np.zeros_like(AHL[-1])] + bandpass_acts
inverse_bandpass_acts = [np.ones_like(AHL[-1])] + inverse_bandpass_acts

# get the logic gates at each position for each activation function

all_logic_gates = []

for activation_func in [threshold_acts, inverse_threshold_acts, bandpass_acts, inverse_bandpass_acts]:
    logic_gates = []
    for n in range(nx*ny):
        logic_gate = (activation_func[0][n], activation_func[1][n], activation_func[2][n], activation_func[3][n])
        logic_gates.append(logic_gate)
    logic_gates = set(logic_gates)
    all_logic_gates.append(logic_gates)


if len(all_logic_gates[0]) > 5 or len(all_logic_gates[1]) > 5 or len(all_logic_gates[2]) > 8 or len(all_logic_gates[3]) > 8:
    print(i,j)
    print(len(all_logic_gates[0]))
    print(len(all_logic_gates[1]))
    print(len(all_logic_gates[2]))
    print(len(all_logic_gates[3]))
    #print(all_logic_gates[2])

    # logic gate possible with threshold and inverse threshold
    print(len(all_logic_gates[0].union(all_logic_gates[1])))

    # logic gate possible with bandpass and inverse bandpass
    print(len(all_logic_gates[2].union(all_logic_gates[3])))
    print()


# get logic gate maps

threshold_acts = np.array(threshold_acts)
inverse_threshold_acts = np.array(inverse_threshold_acts)
bandpass_acts = np.array(bandpass_acts)
inverse_bandpass_acts = np.array(inverse_bandpass_acts)

threshold_lgs = np.zeros_like(threshold_acts[0])

#base_two = np.array([1, 2, 4, 8])
base_two = np.array([8,4,2,1])
def map_decimals_to_LGS(decimals):
    mapping = {0: 'OFF', 1: 'AND', 2: 'A AND NOT B', 3: 'A', 4: 'B AND NOT A', 5: 'B', 6: 'XOR', 7: 'OR', 8: 'NOR', 9:'XNOR', 10: 'NOT B', 11: 'A OR NOT B', 12: 'NOT A', 13: 'B OR NOT A', 14: 'NAND', 15: 'ON'}
    m = []
    for d in decimals:
        m.append(mapping[d])

    print(m)
    return m


titles = ['Threshold', 'Inverse threshold', 'Bandpass', 'Inverse bandpass']
for j,acts in enumerate([threshold_acts, inverse_threshold_acts, bandpass_acts, inverse_bandpass_acts]):
    lgs = np.zeros_like(acts[0])
    for i in range(len(lgs)):
        lg = acts[:, i]
        decimal = np.sum(lg*base_two)
        lgs[i] = decimal

    unique_lgs = np.array(list(set(lgs)))

    print(unique_lgs)
    cmap = plt.cm.tab20c
    bounds = unique_lgs - 0.5
    bounds = np.append(bounds, max(unique_lgs) + 0.5)
    #bounds = unique_lgs
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    print(bounds)
    #bounds = [0., 0.5, 1.]
    #norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    #fig, axs = plt.subplots(1, 3)
    fig = plt.figure()
    plot = plt.imshow(lgs.reshape(nx, ny), cmap = cmap, norm = norm)
    plt.title(titles[j])

    cbar = plt.colorbar()

    ticks = []

    for i in range(len(bounds) - 1):
        ticks.append((bounds[i] + bounds[i+1])/2)

    cbar.set_ticks(ticks)

    cbar.set_ticklabels(map_decimals_to_LGS(unique_lgs))
    plt.savefig(titles[j] + '.pdf')
    plt.show()





sys.exit()
for i in range(25):
    print(i)
    for j in range(25):
        if not (i == j):
            threshold_acts = []
            inverse_threshold_acts = []
            bandpass_acts = []
            inverse_bandpass_acts = []

            for k in range(3):
                AHL = np.load('/home/neythen/Desktop/Projects/synbiobrain/maths_stuff/verify_comb_maths_results/' + str(i) + ',' + str(j) + ': ' + str(k) + '.npy')
                '''
                print(AHL.shape)
                print('max: ', np.max(AHL))
                print('min: ', np.min(AHL))
                print('mean: ', np.mean(AHL))
                print('median: ', np.median(AHL))
                print()
                '''
                threshold_act = get_activation_map(AHL[-1, :], lambda x: threshold(x, t))
                inverse_threshold_act = get_activation_map(AHL[-1, :], lambda x: inverse_threshold(x, t))
                bandpass_act = get_activation_map(AHL[-1, :], lambda x: bandpass(x, t1, t2))
                inverse_bandpass_act = get_activation_map(AHL[-1, :], lambda x: inverse_bandpass(x, t1, t2))

                threshold_acts.append(threshold_act)
                inverse_threshold_acts.append(inverse_threshold_act)
                bandpass_acts.append(bandpass_act)
                inverse_bandpass_acts.append(inverse_bandpass_act)

                fig = plt.figure()

                cmap = mpl.colors.ListedColormap(['r', 'g'])
                bounds = [0., 0.5, 1.]
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

                plot = plt.imshow(threshold_act.reshape(nx, ny), cmap = cmap)

                plt.show()



            threshold_acts = [np.zeros_like(AHL[-1])] + threshold_acts
            inverse_threshold_acts = [np.ones_like(AHL[-1])] + inverse_threshold_acts
            bandpass_acts = [np.zeros_like(AHL[-1])] + bandpass_acts
            inverse_bandpass_acts = [np.ones_like(AHL[-1])] + inverse_bandpass_acts

            # get the logic gates at each position for each activation function

            all_logic_gates = []

            for activation_func in [threshold_acts, inverse_threshold_acts, bandpass_acts, inverse_bandpass_acts]:
                logic_gates = []
                for n in range(nx*ny):
                    logic_gate = (activation_func[0][n], activation_func[1][n], activation_func[2][n], activation_func[3][n])
                    logic_gates.append(logic_gate)
                logic_gates = set(logic_gates)
                all_logic_gates.append(logic_gates)


            if len(all_logic_gates[0]) > 5 or len(all_logic_gates[1]) > 5 or len(all_logic_gates[2]) > 8 or len(all_logic_gates[3]) > 8:
                print(i,j)
                print(len(all_logic_gates[0]))
                print(len(all_logic_gates[1]))
                print(len(all_logic_gates[2]))
                print(len(all_logic_gates[3]))
                #print(all_logic_gates[2])

                # logic gate possible with threshold and inverse threshold
                print(len(all_logic_gates[0].union(all_logic_gates[1])))

                # logic gate possible with bandpass and inverse bandpass
                print(len(all_logic_gates[2].union(all_logic_gates[3])))
                print()

# assume a bandpass, inverse BP, threhold and inverse threshold
# get activation map for each state form each activation function
# overlay activation maps of each function and find the logic gates that are possible



'''
for i in range(25):
    for j in range(25):
        if not i == j:
            for k in range(3):
                AHL = np.load('/home/neythen/Desktop/Projects/synbiobrain/maths_stuff/verify_comb_maths_results/' + str(i) + ',' + str(j) + ': ' + str(k) + '.npy')
                print('max: ', np.max(AHL))
                print('min: ', np.min(AHL))
                print('mean: ', np.mean(AHL))
'''
