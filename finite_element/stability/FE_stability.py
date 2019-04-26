import os
import sys
os.environ['PYOPENCL_CTX'] = '0'
D = 3e-6 #mm^2/s
f = open('FE_stability_' + str(D) + '.txt', "w")
sys.path.append('/home/neythen/Desktop/Projects/synbiobrain/app')
from diffusion_sim import *

for mesh in ['/home/neythen/Desktop/Projects/synbiobrain/meshes/square.vtk', '/home/neythen/Desktop/Projects/synbiobrain/meshes/fine_square.vtk']:

    grid = Grid.from_file(mesh)

    global_A = grid.get_global_A(D)
    heated_element = np.zeros(len(grid.vertices))
    n_time_steps = 6000

    node_dim = np.array([11,11]) #dimensions of the grid, number of nodes in x and y directions
    grid_corners = np.array([[-1, 1], [-1, 1]])
    input_indeces = np.array([71])
    output_indeces = np.array(range(node_dim[0] * node_dim[1]))
    output_indeces = np.delete(output_indeces, input_indeces)

    all_node_positions = get_node_positions(node_dim, grid_corners)
    input_node_positions = [all_node_positions[i] for i in input_indeces]
    output_node_positions = [all_node_positions[i] for i in output_indeces]

    node_radius = 0.05
    output_vertices = assign_vertices(grid.vertices, output_node_positions, node_radius)
    input_vertices = assign_vertices(grid.vertices, input_node_positions, node_radius)

    ps = [5, 10, 5, 10]

    f.write('MESH: ' + mesh + '\n')
    f.write('-------------------------------------\n')
    for delta_t in [100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01]:
        print('delta_t: ', delta_t)
        f.write('delta_t: ' + str(delta_t) + '\n')
        inv_M = grid.synbio_brain(heated_element, delta_t, n_time_steps, input_vertices, output_vertices, ps, D, 1.)

        x = np.diag(np.ones(len(grid.vertices)))-delta_t*np.matmul(np.diag(inv_M), global_A)

        evs = np.linalg.eigvals(x)
        print('n negs: ', len(evs[evs<0]))
        f.write('n negs: ' + str(len(evs[evs<0]))+ '\n')
        abs_evs = np.absolute(evs)
        print('max abs ev: ' + str(np.max(abs_evs)))
        f.write('max abs ev: ' + str(np.max(abs_evs))+ '\n')
        f.write('\n')
        print()


f.close()
