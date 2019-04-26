import sys
sys.path.append('/home/neythen/Desktop/Projects/synbiobrain/finite_element/app')

from diffusion_sim import *
import numpy as np
from mesh_funcs import *
import sys



node_dim = np.array([9,9]) #dimensions of the grid, number of nodes in x and y directions
grid_corners = np.array([[-1, 1], [-1, 1]])
#input_indeces = np.array([30, 32, 48, 50])
input_indeces = np.array([30])



node_radius = 2/19 * 1/2 + 0.04
#create_circles_2("empty_square.geo", node_dim, node_radius, grid_corners, "2by2.geo")
clear_fields("/home/neythen/Desktop/Projects/synbiobrain/finite_element/meshes/9by9_buffer.geo")
create_node_fields("/home/neythen/Desktop/Projects/synbiobrain/finite_element/meshes/9by9_buffer.geo", node_dim, node_radius, grid_corners, 0.15, 1)
