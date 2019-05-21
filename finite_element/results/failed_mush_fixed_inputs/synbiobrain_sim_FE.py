
import sys

sys.path.append('/home/neythen/Desktop/Projects/synbiobrain/finite_element/app/mesh_creation')
sys.path.append('/home/neythen/Desktop/Projects/synbiobrain')
from mesh_funcs import *
import matplotlib.backends.backend_pdf
from diffusion_sim import *

os.environ['PYOPENCL_CTX'] = '0'

def I(vertice): # identity function for test
    return 1

'''
actual positions are done in cartesian

but the abstract grid is treated as a matrix
'''


grid_corners = np.array([[-1, 1], [-1, 1]])
grid = SynBioBrainFE.from_file("../meshes/9by9.vtk",grid_corners, 'float32')
D = 3e-6 #cm^2/s
#D = 0.2

delta_t = 0.9
n_time_steps = 30000
frame_skip = 1000

X = grid.mesh['vertices'][:,0]
Y = grid.mesh['vertices'][:,1]

node_dim = np.array([9,9]) #dimensions of the grid, number of nodes in x and y directions

#input_indeces = np.array([30, 32, 48, 50])
input_indeces = np.array([39, 40, 49])
#input_indeces = np.array([])

node_radius = 2/23 * 1/2

output_indeces = np.array(range(node_dim[0] * node_dim[1]))
output_indeces = np.delete(output_indeces, input_indeces)
#output_indeces = np.array([])
all_node_positions = get_node_positions(node_dim, grid_corners)
print(len(all_node_positions))
input_node_positions = [all_node_positions[i] for i in input_indeces]
output_node_positions = [all_node_positions[i] for i in output_indeces]

all_node_vertices, one_hot_all, all_node_counts = assign_vertices(grid.vertices, all_node_positions, node_radius)
output_vertices, one_hot_out, output_counts = assign_vertices(grid.vertices, output_node_positions, node_radius)
input_vertices, one_hot_in, input_counts = assign_vertices(grid.vertices, input_node_positions, node_radius)

#print('length: ',len(input_vertices[0]))

print(all_node_vertices[0])
production_rate = 0.01
ps = [1.5, 3.5, 1., 5.5]

with Timer() as t:

    heated_element = np.zeros(len(grid.vertices))
    '''
    for i, point in enumerate(grid.vertices):

        if -1/3 < point[0] < 1/3 and -node_radius < point[1] < node_radius:
            heated_element[i] = 30
    '''
    #heated_element[all_node_vertices[39] + all_node_vertices[40] + all_node_vertices[41]] = 3
    print('02')
    Us, activated_ts = grid.synbio_brain(heated_element, delta_t, n_time_steps, one_hot_in, one_hot_out, ps, D, production_rate)
    #Us_cheat, activated_ts_cheat = grid.synbio_brain_cheating(heated_element, 0.00005, n_time_steps, input_vertices, output_vertices, threshold)
    print('12')

fast_time = t.interval
print(fast_time)
bounds = np.linspace(0, 2, 5)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)


fig = plt.figure()

ax = fig.gca(projection = '3d')
ax.view_init(elev  = 90,azim=-90)
ax.plot_trisurf(grid.vertices[:,0], grid.vertices[:,1], Us[-1],cmap=cm.coolwarm)
np.save("output/AHL_ts.npy", Us)
plt.show()

'''
fig = plt.figure()

ax = fig.gca(projection = '3d')
ax.plot_trisurf(grid.vertices[:,0], grid.vertices[:,1], input_vertices, shade=False, cmap=cm.coolwarm, linewidth=0)
plt.show()
'''


# plot the anmations
pdf = matplotlib.backends.backend_pdf.PdfPages("output/AHL.pdf")

ims = []
fig = plt.figure()
for i in range(0,len(Us), frame_skip):
    print(i)

    ax = fig.gca(projection = '3d')
    fig.suptitle('time: ' + str(delta_t*i))
    plot = ax.plot_trisurf(grid.vertices[:,0], grid.vertices[:,1], Us[i], shade=False, cmap=cm.coolwarm, linewidth=0)
    ims.append([plot])
    pdf.savefig(fig)
anim = animation.ArtistAnimation(fig, ims, interval = 200, blit = True, repeat_delay = 100)
pdf.close()
plt.show()

'''
anim = animation.ArtistAnimation(fig, ims, interval = 50, blit = True, repeat_delay = 100)
plt.show()

fig = plt.figure()
ims = []
for i in range(0,len(Us_cheat), 100):
    print(i)
    ax = fig.gca(projection = '3d')
    ax.view_init(elev = 90,azim=0)

    ims.append([ax.plot_trisurf(grid.vertices[:,0], grid.vertices[:,1], Us_cheat[i], shade=False, cmap=cm.coolwarm, linewidth=0)])
'''

# ANIMATE WITH MOVIEPY (UPDATE THE CURVE FOR EACH t). MAKE A GIF.

def make_frame_mpl(t):
     # <= Update the curve
    fig_mpl = ims[t]
    return mplfig_to_npimage(fig_mpl) # RGB image of the figure
'''
animation =mpy.VideoClip(make_frame_mpl, duration=5)
animation.write_gif("AHL.gif", fps=20)
'''

pdf = matplotlib.backends.backend_pdf.PdfPages("output/GFP.pdf")
fig = plt.figure()
ims = []
for i in range(0,len(activated_ts), frame_skip):
    print(i)

    fig.suptitle('time: ' + str(delta_t*i))
    ax = fig.gca(projection = '3d')
    ax.view_init(elev = 90,azim=-90)
    plot = ax.plot_trisurf(grid.vertices[:,0], grid.vertices[:,1], activated_ts[i], shade=False, cmap=cm.coolwarm, linewidth=0)
    ims.append([plot])
    pdf.savefig(fig)

anim = animation.ArtistAnimation(fig, ims, interval = 200, blit = True, repeat_delay = 100)
pdf.close()
plt.show()
np.save("output/GFP_ts.npy", activated_ts)
def make_frame_mpl(t):
     # <= Update the curve
    fig_mpl = ims[t]
    return mplfig_to_npimage(fig_mpl) # RGB image of the figure
'''
animation =mpy.VideoClip(make_frame_mpl, duration=5)
animation.write_gif("GFP.gif", fps=20)
'''
