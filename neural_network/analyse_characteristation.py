
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

import numpy as np
import math
D = 5*10**(-10) #m^2/s

from scipy.optimize import curve_fit

def prob_dist(r, t): # from statistical physics
    '''
    one dimensional
    '''
    return 1/(4*math.pi * D * t) * math.exp(-(r**2/4*D*t))


def logit(x):
    # inverse of sigmoid

    return np.log(x/(1-x))


def hill(conc, n, kd, min, max):
    return min + (max-min)*(conc**n/(kd**n + conc**n))

def inverse_hill(AFU, n, kd, min, max):

    return ((kd**n*((AFU-min)/(max-min)))/(1-((AFU-min)/(max-min))))**(1/n)



if __name__ == '__main__':


    # load clemens data

    # reciever activation with AHL

    AHL_conc = []
    AFUs = []
    with open("/home/neythen/Desktop/Projects/synbiobrain/Sender_Receiver_Characterisation_Clemens/06_03_20/Concentration/Normalised Filtered Medians all Conc.csv") as file:
        file.readline()
        for line in file:
            print(line)
            line = line.split(',')

            try:
                AHL_conc.append(float(line[7]))
                if float(line[7]) < 12: # line up the data from the two seperate experiments
                    AFUs.append(float(line[5])/3.2)
                else:
                    AFUs.append(float(line[5]))
            except:
                pass

    print(len(AHL_conc))
    print(AFUs)








    #reciever activation against known AHL

    AHL_conc = np.array(AHL_conc)
    AFUs = np.array(AFUs)
    n, kd, min, max = curve_fit(hill, AHL_conc, AFUs, p0 = [2, 30, 0, 60000])[0]
    print('fitted n: ', n)
    print('fitted kd: ', kd)
    print('min: ', min)
    print('max: ', max)


    plt.title('Receiver response curve')
    #plt.plot(np.log10(AHL_conc), np.log10(np.array(AFUs)/(1-np.array(AFUs))))
    plt.scatter(AHL_conc, AFUs, label = 'actual')
    plt.plot(range(100), hill(np.array(range(100)), n, kd, min, max), label = 'hill with params n = ' + str(round(n, 2)) + ', kd = ' + str(round(kd, 2)) + ', min = ' + str(round(min,0))+ ', max = ' + str(round(max,0)))
    plt.xlabel('AHL conc (nM)')
    plt.ylabel('AFU')
    #plt.legend()

    plt.figure()
    plt.title('Receiver response curve log scale')
    plt.scatter(np.log(AHL_conc), np.log(AFUs), label='actual')
    plt.plot(np.log(range(100)), np.log(hill(np.array(range(100)), n, kd, min, max)),
                label = 'hill with params n = ' + str(round(n, 2)) + ', kd = ' + str(round(kd, 2)) + ', min = ' + str(round(min,0))+ ', max = ' + str(round(max,0)))
    plt.xlabel('log(AHL conc)')

    plt.ylabel('log(AFU)')
    #plt.legend()

    #plt.close('all')




    # sender receiver experiments
    times = []
    distances = []

    SRAFUs = []
    with open("/home/neythen/Desktop/Projects/synbiobrain/Sender_Receiver_Characterisation_Clemens/06_03_20/Timecourse 2/norm_median.csv") as file:

        file.readline()
        for line in file:
            line = line.split(',')
            try:
                distances.append(float(line[8]))
                times.append(float(line[5]))

                SRAFUs.append(float(line[7]))
            except:
                pass

    times = np.array(times)
    distances = np.array(distances)

    results_dict ={}
    for i in range(len(times)):
        if distances[i] in results_dict:
            if times[i] in results_dict[distances[i]]:
                results_dict[distances[i]][times[i]].append(SRAFUs[i])
            else:
                results_dict[distances[i]][times[i]] = []
                results_dict[distances[i]][times[i]].append(SRAFUs[i])
        else:
            results_dict[distances[i]] = {}
            results_dict[distances[i]][times[i]] = []

            results_dict[distances[i]][times[i]].append(SRAFUs[i])

    print(results_dict.keys())
    print(results_dict[list(results_dict.keys())[0]].keys())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(times, distances, SRAFUs)
    ax.set_xlabel('times')
    ax.set_ylabel('distances')
    ax.set_zlabel('AFU')

    distances = np.sort(np.array(list(results_dict.keys())))
    times = np.sort(np.array(list(results_dict[list(results_dict.keys())[0]].keys())))

    SRAFUs = []

    for d in distances:
        l =[]
        for t in times:
            l.append(np.mean(results_dict[d][t]))

        SRAFUs.append(l)


    X, Y = np.meshgrid(distances,times)


    Z = np.array(SRAFUs).T
    # normalise to lowest value
    Z -= np.min(Z)

    # normalise to the minimum of reciever exp as these are likely the same
    Z += min


    print(Z)
    print(X.shape, Y.shape, Z.shape)

    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_wireframe(X, Y, Z)

    ax.set_xlabel('distance (mm)')
    ax.set_ylabel('time (hours)')
    ax.set_zlabel('AFU')






    #min = 0 # normalise minimum, value to 0
    inferred_AHL = inverse_hill(Z, n, kd, min, max)
    print(np.max(Z))
    print(Z)
    print(inferred_AHL)
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_wireframe(X, Y, inferred_AHL)

    ax.set_xlabel('distance (mm)')
    ax.set_ylabel('time (hours)')
    ax.set_zlabel('Inferred AHL_conc')



    weights_from_mathemetica =[[0.021683922377034665, 0.030920133424316966, 0.034973368403054414, \
0.03871098809201532, 0.04217357871257692, 0.06909126248544832], \
[0.0015368210469670863, 0.0037760490692458925, 0.005089611778702592, \
0.006472892214514919, 0.007895974144804866, 0.022929686977473163], \
[0.000047178392792417286, 0.00026701234624318664, \
0.00046870021937784305, 0.0007332669552416473, 0.0010570961250979947, \
0.006944036132292156]]

    weights = np.array(weights_from_mathemetica).T
    print('inferred AHL: ', inferred_AHL)
    print('weight: ', weights)
    print('infA/w: ', inferred_AHL/weights) #CUT OFF THE ZEROS


    Z = weights

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_wireframe(X, Y, Z)

    ax.set_xlabel('distance (mm)')
    ax.set_ylabel('time (hours)')
    ax.set_zlabel('weight')

    Z = (inferred_AHL/weights)

    print(weights[1:, 1:])
    print(weights[:-1, :-1]) # cut off values with really low weights where noise will have a huge effect
    print(inferred_AHL[:, :-1]/weights[:, :-1])

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_wireframe(X[1:, :-1], Y[1:, :-1], Z[1:, :-1])

    ax.set_xlabel('distance (mm)')
    ax.set_ylabel('time (hours)')
    ax.set_zlabel('S')

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_wireframe(X, Y, Z)

    ax.set_xlabel('distance (mm)')
    ax.set_ylabel('time (hours)')
    ax.set_zlabel('S')

    #plt.show()
    print('mean S: ', np.mean(Z[1:, :-1]))
    print('std S: ', np.std(Z[1:, :-1]))
    plt.show()

