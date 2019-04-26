
import sys
import matplotlib.backends.backend_pdf
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
sys.path.append('/home/neythen/Desktop/Projects/synbiobrain/')
from diffusion_sim import *
os.environ['PYOPENCL_CTX'] = '0'
nx = 300
ny = 300
node_radius = 20/40
node_dim = np.array([10, 10])
grid_corners = np.array([[-10, 10], [-10, 10]])

grid = SynBioBrainFD(grid_corners, nx, ny, 'float32')
vertex_positions = np.array([grid.get_node_position(i) for i in range(grid.n_nodes)])
barriers = ['1', '0.8', '0.6', '0.4', '0.2', '0.15', '0.1', '0.05', '0.01']
all_cohesive_ts = []
for barrier in barriers:
    print(barrier)
    activated_ts = np.load('/home/neythen/Desktop/Projects/synbiobrain/finite_difference/results/diffusion_factor/'+ barrier +'_barrier/output/GFP_ts.npy')
    cohesive_ts = count_cohesive_nodes_FD(activated_ts, vertex_positions, node_dim, node_radius, grid_corners)
    all_cohesive_ts.append(cohesive_ts)
all_cohesive_ts = np.array(all_cohesive_ts)

np.save('all_cohesive_ts.npy', all_cohesive_ts)
