
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

grid = SynBioBrainCUDA(grid_corners, nx, ny, checkerboard = False, dtype = 'float32')
output_indeces = np.array(range(node_dim[0] * node_dim[1]))
#output_indeces = np.array([0])
#input_indeces = np.array([30, 32, 48, 50])

#input_indeces = np.random.choice(np.array(range(node_dim[0] * node_dim[1])), size = (50,))
input_indeces = np.array([0])
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

boundary_cond = 'periodic'
params = [delta_t, n_time_steps, n_loops, ps, D, production_rate, boundary_cond, node_radius]
heated_element = np.zeros(grid.n_vertices)

with Timer() as t:

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
    overall_Us, overall_activated = grid.synbio_brain(heated_element, grid_corners, node_dim, input_indeces, output_indeces, params)
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
    plot = plt.imshow(overall_Us[i].reshape(grid.nx, grid.ny), cmap='plasma', vmin = 0, vmax = 0.25)

    ims.append([plot])
    pdf.savefig(fig)



anim = animation.ArtistAnimation(fig, ims, interval = 200, blit = True, repeat_delay = 100)

pdf.close()
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
