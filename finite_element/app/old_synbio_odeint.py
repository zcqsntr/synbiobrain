from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import numpy as np

from diffusion_sim import *


os.environ['PYOPENCL_CTX'] = '0'

grid = Grid.from_file("structured_11by11.vtk")
D = 3e-6 #cm^2/s
#D = 0.2
production_rate = 1.

node_dim = np.array([11,11]) #dimensions of the grid, number of nodes in x and y directions
grid_corners = np.array([[-1, 1], [-1, 1]])
input_indeces = np.array([70])
output_indeces = np.array(range(node_dim[0] * node_dim[1]))
output_indeces = np.delete(output_indeces, input_indeces)

all_node_positions = get_node_positions(node_dim, grid_corners)
print(len(all_node_positions))
input_node_positions = [all_node_positions[i] for i in input_indeces]
output_node_positions = [all_node_positions[i] for i in output_indeces]

node_radius = 2/23 * 1/2
output_vertices = assign_vertices(grid.vertices, output_node_positions, node_radius)
input_vertices = assign_vertices(grid.vertices, input_node_positions, node_radius)

ps = [5, 10, 5, 10]

heated_element = np.zeros(len(grid.vertices))
A = grid.get_global_A(D)
inv_M = grid.synbio_brain(heated_element, 1, 500, input_vertices, output_vertices, ps, D, production_rate)

def dydt(t, y):
    print(t)
    return - np.matmul(inv_M,np.matmul(A, y))

for i, point in enumerate(grid.vertices):

    if 0 < point[0] < 0.5 and 0<point[1] <0.5:
        heated_element[i] = 1


bunch = solve_ivp(dydt,  (0, 50), heated_element)
print(bunch.t)
Us = np.array(bunch.y.T)
print(Us.shape)
frame_skip = 5
# plot the anmations

fig = plt.figure()
ims = []
for i in range(0,len(Us), frame_skip):
    print(i)
    ax = fig.gca(projection = '3d')

    plot = ax.plot_trisurf(grid.vertices[:,0], grid.vertices[:,1], Us[i], shade=False, cmap=cm.coolwarm, linewidth=0)
    ims.append([plot])

anim = animation.ArtistAnimation(fig, ims, interval = 200, blit = True, repeat_delay = 100)

plt.show()
