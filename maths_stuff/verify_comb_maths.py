
import sys

sys.path.append('/home/neythen/Desktop/Projects/synbiobrain/')
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from diffusion_sim_CUDA import *
import matplotlib.backends.backend_pdf
import matplotlib as mpl
import matplotlib.pyplot as plt
import math



grid_corners = np.array([[-1, 1], [-1, 1]]) # in cm

nx = 200
ny = 200

node_dim = np.array([5, 5])

D = 3e-6 #cm^2/s

grid = SynBioBrainCUDA(grid_corners, nx, ny, checkerboard = False, dtype = 'float32')
print('vertices: ', grid.n_vertices)
output_indeces = np.array(range(node_dim[0] * node_dim[1]))
#output_indeces = np.array([0])
#input_indeces = np.array([30, 32, 48, 50])

#input_indeces = np.random.choice(np.array(range(node_dim[0] * node_dim[1])), size = (50,))
input_indeces = np.array([12])
#input_indeces = np.array([0,1,2])

#input_indeces = np.array([1374, 1375, 1376, 1377, 1378])


node_radius = 1/30

try:
    one_hot_in[input_indeces] = 1
except:
    pass

#output_indeces = np.delete(output_indeces, input_indeces)

#one_hot_out[4354] = 1

#production_rate = 0.000001
production_rate = 0.00000001
ps = [10**(-1.9),10**(-0.6) ,10**(-1.9), 10**(-0.6)] # thresholds in micro molar
#ps = [10**(-1.9),10**(-0.6) ,10**(-2), 10]
print(ps)

delta_t = 100
delta_t = (grid.dx)**2/(2*D) * 0.5 # maximum stable timestep with 50% safety
print('delta_t (seconds): ', delta_t)

simulation_time = 10 #number of real hours to simulate

total_timesteps = int(simulation_time*60*60/delta_t)
print('total_timesteps: ', total_timesteps)
# calculate how much memory will be needed and loop however many times so it fits on GPU
bits = nx*ny*32*total_timesteps
bytes = bits/8
print('number of bytes: ', bytes)

# assume 2.5gb of v mem free for sim
n_loops = math.ceil(bytes/(2.2*1e9))
print('n_loops: ', n_loops)
n_time_steps = int(total_timesteps/n_loops) # n_timesteps per loop
print('n_timesteps per loop: ', n_time_steps)

n_frames = 70
frame_skip = math.ceil(total_timesteps/n_frames) # for plotting animations
print(total_timesteps)
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



print('n_loops: ', n_loops)

boundary_cond = 'insulating'
params = [delta_t, n_time_steps, n_loops, ps, D, production_rate, boundary_cond, node_radius]
print('SIMULATION TIME (hours): ', simulation_time)
heated_element = np.zeros(grid.n_vertices)
all_node_positions = grid.get_node_positions(node_dim, grid_corners)

vertices, one_hot_vertices = grid.assign_vertices(grid.vertex_positions, np.transpose(all_node_positions[4]).reshape(1,2), node_radius)

#heated_element[vertices] = 90

for i in range(25):
    for j in range(25):
        print('----------------------------------------------------------------', i,j)
        if not i == j:
            i = 6
            j = 18
            for k, input_indeces in enumerate([np.array([i]), np.array([j]), np.array([i, j])]):

                with Timer() as t:

                    overall_Us, overall_activated = grid.synbio_brain(heated_element, grid_corners, node_dim, input_indeces, output_indeces, params)
                    print('max: ',np.max( overall_Us))
                    print('mean: ',np.mean( overall_Us))

                np.save(str(i) + ',' + str(j) + ': ' + str(k) + '.npy', np.array(overall_Us[::10]))
                    #for 300x300 grid


                print(np.all(overall_Us[-1] == overall_Us[-2]))
                #plt.plot([overall_Us[i][49063] for i in range(len(overall_Us))])
                #plt.show()

                print('X: ', X.shape)
                print('nx: ', grid.nx)
                fast_time = t.interval
                print('real time taken (seconds) : ', fast_time)
                bounds = np.linspace(0, 2, 5)
                norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)


                fig = plt.figure()
                ax = fig.gca(projection = '3d')
                ax.view_init(elev = 0,azim=0)
                ax.set_xlabel('X position (cm)')
                ax.set_ylabel('[AHL]')
                plot = ax.plot_trisurf(grid.vertex_positions[:,0], grid.vertex_positions[:,1], overall_Us[-1], shade=False, cmap=cm.coolwarm, linewidth=0)
                #plot = plt.imshow(overall_Us[-1].reshape(grid.nx, grid.ny), cmap='plasma')
                np.save("output/AHL_ts.npy", overall_Us)

                plt.show()


                # plot the anmations
                pdf = matplotlib.backends.backend_pdf.PdfPages("output/AHL.pdf")
                fig = plt.figure()
                ims = []
                for i in range(0,len(overall_Us), frame_skip):
                    print(i)
                    #ax = fig.gca(projection = '3d')
                    #plot = ax.bar3d(x, y, bottom, width, depth, Us[i], shade=True, color = 'b')
                    fig.suptitle('time: ' + str(delta_t*i))
                    #plot = ax.plot_surface(X, Y, Us[i].reshape(grid.nx, grid.ny), rstride=1, cstride=1, cmap=cm.coolwarm,
                    #    linewidth=0, antialiased=False)
                    plot = plt.imshow(overall_Us[i].reshape(grid.nx, grid.ny), cmap='plasma', vmin = 0)

                    ims.append([plot])
                    pdf.savefig(fig)



                anim = animation.ArtistAnimation(fig, ims, interval = 200, blit = True, repeat_delay = 100)

                pdf.close()
                plt.title('AHL_conc')
                plt.show()

                plt.close()
                pdf = matplotlib.backends.backend_pdf.PdfPages("output/GFP.pdf")
                fig = plt.figure()
                ims = []
                cmap = mpl.colors.ListedColormap(['g', 'k'])
                bounds = [0., 0.5, 1.]
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

                for i in range(0,len(overall_activated), frame_skip):
                    print(i)
                    fig.suptitle('time: ' + str(delta_t*i))


                    plot = plt.imshow(overall_activated[i].reshape(grid.nx, grid.ny), cmap = 'inferno')

                    ims.append([plot])
                    pdf.savefig(fig)

                anim = animation.ArtistAnimation(fig, ims, interval = 200, blit = True, repeat_delay = 100)
                plt.show()
                pdf.close()

                np.save("output/GFP_ts.npy", overall_activated)
