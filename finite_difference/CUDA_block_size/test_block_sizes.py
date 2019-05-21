
import sys

sys.path.append('/home/neythen/Desktop/Projects/synbiobrain/')
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from diffusion_sim_CUDA import *
import matplotlib.backends.backend_pdf
import matplotlib as mpl
import matplotlib.pyplot as plt


os.environ['PYOPENCL_CTX'] = '0'
grid_corners = np.array([[-10, 10], [-10, 10]])

nx = 300
ny = 300

node_dim = np.array([1, 10])

D = 3e-6 #cm^2/s

grid = SynBioBrainCUDA(grid_corners, nx, ny, 'float32')
output_indeces = np.array(range(node_dim[0] * node_dim[1]))
#output_indeces = np.array([0])
#input_indeces = np.array([30, 32, 48, 50])

#input_indeces = np.random.choice(np.array(range(node_dim[0] * node_dim[1])), size = (50,))
input_indeces = np.array([9])
#input_indeces = np.array([0,1,2])

#input_indeces = np.array([1374, 1375, 1376, 1377, 1378])


node_radius = 20/40

try:
    one_hot_in[input_indeces] = 1
except:
    pass

#output_indeces = np.delete(output_indeces, input_indeces)

#one_hot_out[4354] = 1

production_rate = 0.00001
ps = [10**(-1.9),10**(-0.6) ,10**(-2), 10**(-0.4)]
#ps = [10**(-1.9),10**(-0.6) ,10**(-2), 10]
print(ps)

delta_t = 100
n_time_steps = 2000
frame_skip = 150
X = grid.vertex_positions[:,0]
Y = grid.vertex_positions[:,1]
X = np.arange(-1, 1 + grid.dx, grid.dx)
Y = np.arange(-1, 1 + grid.dx, grid.dx)
X, Y = np.meshgrid(X, Y)
x, y= X.ravel(), Y.ravel()
'''
plt.figure()
plot = plt.imshow(one_hot_barrier.reshape(grid.nx, grid.ny), cmap='plasma')
plt.show()
'''

# calculate how much memory will be needed and loop however many times so it fits on GPU

n_loops = 3
checkerboard = False
boundary_cond = 'periodic'
params = [delta_t, n_time_steps, n_loops, ps, D, production_rate, boundary_cond, node_radius]
heated_element = np.zeros(grid.n_vertices)
block_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
times = np.zeros(len(block_sizes))
for i in range(10):
    for j,block_size in enumerate(block_sizes):
        with Timer() as t:
            print()
            print()
            print('----------------------------------------------------------')
            print(block_size)
            '''
            overall_Us = []
            overall_activated = []
            for i in range(n_loops):
                print(i)

                #heated_element[all_node_vertices[39] + all_node_vertices[40] + all_node_vertices[41]] = 3

                Us, activated_ts = grid.synbio_brain(heated_element, delta_t, n_time_steps, one_hot_in, one_hot_out, ps, D, production_rate)
                one_hot_in = np.zeros(grid.n_nodes)
                heated_element = Us[-1] # next IC is end of previous
                overall_Us.append(Us)
                overall_activated.append(activated_ts)
            '''
            grid.blockDim = (block_size, 1,1)
            overall_Us,overall_activated = grid.synbio_brain(heated_element, grid_corners, node_dim, input_indeces, output_indeces, params)
            print('max: ',np.max( overall_Us))
            print('mean: ',np.mean( overall_Us))


            #for 300x300 grid
            '''
            print('where a', np.where(overall_activated[:, 49063] == 1))
            print('where a', np.where(overall_activated[:, 49063] == 0))
            '''
        print(np.all(overall_Us[-1] == overall_Us[-2]))
        #plt.plot([overall_Us[i][49063] for i in range(len(overall_Us))])
        #plt.show()

        print('X: ', X.shape)
        print('nx: ', grid.nx)
        fast_time = t.interval
        print(fast_time)

        times[j] += fast_time

print(times)
plt.plot(times)
plt.show()
