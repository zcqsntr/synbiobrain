
import sys

sys.path.append('/home/neythen/Desktop/Projects/synbiobrain/')
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from diffusion_sim import *
import matplotlib.backends.backend_pdf
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys

os.environ['PYOPENCL_CTX'] = '0'
grid_corners = np.array([[-10, 10], [-10, 10]])

nx = 300
ny = 300

node_dim = np.array([10, 10])

D = 3e-6 #cm^2/s

grid = SynBioBrainFD(grid_corners, nx, ny, 'float32')
#output_indeces = np.array(range(node_dim[0] * node_dim[1]))
output_indeces = np.array([0])
#input_indeces = np.array([30, 32, 48, 50])

#input_indeces = np.random.choice(np.array(range(node_dim[0] * node_dim[1])), size = (50,))
input_indeces = np.array([45])
#input_indeces = np.array([0,1,2])

#input_indeces = np.array([1374, 1375, 1376, 1377, 1378])

one_hot_in = np.zeros(grid.n_nodes)

one_hot_in[input_indeces] = 1
node_radius = 20/40

try:
    one_hot_in[input_indeces] = 1
except:
    pass

#output_indeces = np.delete(output_indeces, input_indeces)
with Timer() as t:
    all_node_positions = get_node_positions(node_dim, grid_corners)
    input_node_positions = [all_node_positions[i] for i in input_indeces]
    print('input:', input_node_positions)
    output_node_positions = [all_node_positions[i] for i in output_indeces]
    vertex_positions = np.array([grid.get_node_position(i) for i in range(grid.n_nodes)])
    print('positions:', len(vertex_positions))
print(t.interval)
with Timer() as t:
    output_vertices, one_hot_out = assign_vertices(vertex_positions, output_node_positions, node_radius)
    print(output_vertices.shape)
    input_vertices, one_hot_in = assign_vertices(vertex_positions, input_node_positions, node_radius)
    barrier_vertices, one_hot_barrier = get_barrier_vertices(vertex_positions, grid_corners, node_dim, nx, ny)

print('ins:', input_vertices)
#one_hot_out = np.ones(grid.n_nodes)

#one_hot_out = np.zeros(grid.n_nodes)


#one_hot_out[4354] = 1



production_rate = 0.00001
#ps = [10**(-1.9),10**(-0.6) ,10**(-2), 10**(-0.4)]
ps = [10**(-1.9),10**(-0.6) ,10**(-2), 10]
print(ps)

delta_t = 100
n_time_steps = 2000



# calculate how much memory will be needed and loop however many times so it fits on GPU

n_loops = 1
heated_element = np.zeros(grid.n_nodes)
time_elapsed = 0

while True:


    overall_Us,overall_activated = grid.synbio_brain(heated_element, delta_t, n_time_steps, one_hot_in, one_hot_out, one_hot_barrier, ps, D, production_rate, n_loops, boundary_cond = 'insulating')
    print('max: ',np.max( overall_Us))
    print('mean: ',np.mean( overall_Us))

    print(np.all(overall_Us[-1] == overall_Us[-2]))
    if np.all(overall_Us[-1] == overall_Us[-2]):
        break


    time_elapsed += delta_t*n_time_steps

    heated_element = overall_Us[-1]

for i in range(len(overall_Us)):
    if np.all(overall_Us[i] == overall_Us[i+1]):
        time_elapsed += i*delta_t
        break
print(time_elapsed)

#plt.plot([overall_Us[i][49063] for i in range(len(overall_Us))])
#plt.show()

np.save("output/final_AHL.npy", overall_Us[-1])
np.save("output/stady_state_time.npy", np.array([time_elapsed]))
# plot the anmations

np.save("output/final_GFP.npy", overall_activated)
print(np.load("output/stady_state_time.npy"))
