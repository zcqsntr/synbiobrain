import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append('/home/neythen/Desktop/Projects/synbiobrain/')

from diffusion_sim import *
os.environ['PYOPENCL_CTX'] = '0'
grid_corners = np.array([[-10, 10], [-10, 10]])

nx = 300
ny = 300

node_dim = np.array([10, 10])

D = 3e-6 #cm^2/s

degradation_rates = ['-0.000005', '-0.000001', '-0.0000005', '-0.0000001', '-0.00000005']


grid = SynBioBrainFD(grid_corners, nx, ny, 'float32')
vertices = np.array([i for i in range(grid.n_nodes) if abs(grid.get_node_position(i)[0]-1) < 0.034 ])
positions = np.array([grid.get_node_position(i)[1] for i in vertices])
print(positions)
for rate in degradation_rates:

    print(vertices)
    print(len(vertices))
    print(grid.n_nodes)
    final_AHL = np.load('/home/neythen/Desktop/Projects/synbiobrain/finite_difference/degradation_coeff/' + rate + '/output/final_AHL.npy')
    time = np.load('/home/neythen/Desktop/Projects/synbiobrain/finite_difference/degradation_coeff/' + rate + '/output/stady_state_time.npy')[0]
    plt.figure()
    plt.plot(positions, final_AHL[vertices], label = "time = " + str(round(time/(60*60),2)) + ' hours')
    plt.xticks(range(-11, 11, 2)[1:])
    plt.legend()
    plt.title('Degradation rate: ' + rate)
    plt.xlabel("Position")
    plt.ylabel("Concentration")
    plt.savefig(rate + '.png')
plt.show()
# need to plot a 2d slice through the centre of node 45, node 45 at position (1., 1.)
# get nodes with x position = 1
