from diffusion_sim import *

node_dim = np.array([9,9]) #dimensions of the grid, number of nodes in x and y directions
grid_corners = np.array([[-1, 1], [-1, 1]])
#input_indeces = np.array([30, 32, 48, 50])
input_indeces = np.array([30, 32, 48])

os.environ['PYOPENCL_CTX'] = '0'

node_radius = 2/23 * 1/2

one_root = 'grid/one_source/'
two_root = 'grid/two_sources/'
three_root = 'grid/three_sources/'
four_root = 'grid/four_sources/'

root = four_root

final_AHL = np.load(root + 'AHL_ts.npy')[-1]
grid = Grid.from_file("9by9.vtk", 'float32')

ps = [2,5.5,0,10]
activated = get_activated_from_AHL(grid, final_AHL, ps, node_dim, node_radius, grid_corners, input_indeces)
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.view_init(elev = 90,azim=-90)
plot = ax.plot_trisurf(grid.vertices[:,0], grid.vertices[:,1], activated, shade=False, cmap=cm.coolwarm, linewidth=0)

plt.savefig(root + 'between ' +str(ps[0]).replace('.', '-') + ' and ' + str(ps[1]).replace('.', '-'))
plt.show()
