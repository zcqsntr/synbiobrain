import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.optimize import curve_fit
#sys.path.append('/home/neythen/Desktop/Projects/synbiobrain/')

from diffusion_sim import *
os.environ['PYOPENCL_CTX'] = '0'
grid_corners = np.array([[-10, 10], [-10, 10]])

nx = 300
ny = 300

node_dim = np.array([10, 10])

D = 3e-6 #cm^2/s

degradation_rates = ['4.16', '5', '10', '15', '20', '25', '50']


grid = SynBioBrainFD(grid_corners, nx, ny, 'float32')
vertices = np.array([i for i in range(grid.n_vertices) if abs(grid.get_vertex_position(i)[0]-1) < 0.034 ])
positions = np.array([grid.get_vertex_position(i)[1] for i in vertices])
print(positions)
node_radius = 1/2

end = np.where(np.isclose(positions, 1.0367893))[0][0] # get the node at the start of the part of the curve to fit
#print([grid.get_vertex_position(i)[0] for i in range(nx*ny)])

def inverse_x(x, a, b):
    return a/(b*(x-1)) # curve is shifted to the right by 1

def sigmoid(x, a, b, c, d):
    return a/(1+d*np.exp(-b*(x-c)))



fit_func = sigmoid
fig = plt.figure()


for i,rate in enumerate(degradation_rates):
    if i == 0:
        continue
    final_AHL = np.load('/home/neythen/Desktop/Projects/synbiobrain/finite_difference/unstable_AHL/' + rate + 'hours/output/final_AHL.npy')
    final_AHL = final_AHL[vertices]
    time = np.load('/home/neythen/Desktop/Projects/synbiobrain/finite_difference/unstable_AHL/' + rate + 'hours/output/stady_state_time.npy')[0]

    x = positions[:end]
    y = final_AHL[:end]

    def fit_func(x, b, d):
        #print(np.where(np.isclose(y, np.max(y)/2, atol = 0.01)))
        #midpoint_index = np.where(np.isclose(y, np.max(y)/2, atol = 0.01))[0][0]

        #midpoint = x[midpoint_index]
        #print('midpoint', midpoint)
        print(np.max(y))
        return sigmoid(x, np.max(y), b,0.5, d)
    fit_func = sigmoid
    print(len(x))
    print(len(y))

    if i == 5:
        break
    [a, b, c, d] = curve_fit(fit_func, x, y)[0]
    print(a, b, c, d)


    plt.subplot(2, 2, i)
    plt.title('AHL half life: ' + rate + ' hours')
    plt.plot(x, fit_func(x, a, b, c, d), label = 'Fit: {0}, {1}, {2}, {3}'.format(round(a, 2) ,round(b, 2), round(c, 2), round(d, 2)))
    plt.plot(x, y, label = 'Actual')

    plt.legend()


plt.show()
