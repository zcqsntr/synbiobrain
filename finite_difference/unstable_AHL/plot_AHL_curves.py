import numpy as np
import matplotlib.pyplot as plt
import os, sys

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
for rate in degradation_rates:

    print(vertices)
    print(len(vertices))
    print(grid.n_vertices)
    final_AHL = np.load('/home/neythen/Desktop/Projects/synbiobrain/finite_difference/unstable_AHL/' + rate + 'hours/output/final_AHL.npy')
    time = np.load('/home/neythen/Desktop/Projects/synbiobrain/finite_difference/unstable_AHL/' + rate + 'hours/output/stady_state_time.npy')[0]
    plt.figure()
    plt.plot(positions, final_AHL[vertices], label = "time = " + str(round(time/(60*60),2)) + ' hours')
    plt.xticks(range(-11, 11, 2)[1:])
    plt.legend()
    plt.title('AHL half life: ' + rate + ' hours')
    plt.xlabel("Position")
    plt.ylabel("Concentration")
    plt.savefig(rate + 'hours.png')
plt.show()
# need to plot a 2d slice through the centre of node 45, node 45 at position (1., 1.)
# get nodes with x position = 1
