import numpy as np
from scipy.sparse import diags
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.backends.backend_pdf
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from get_grid_config import plot_grid
import copy
from matplotlib.animation import FFMpegWriter
from scipy import optimize



class FasterFFMpegWriter(FFMpegWriter):
    '''FFMpeg-pipe writer bypassing figure.savefig.'''
    def __init__(self, **kwargs):
        '''Initialize the Writer object and sets the default frame_format.'''
        super().__init__(**kwargs)
        self.frame_format = 'argb'

    def grab_frame(self, **savefig_kwargs):
        '''Grab the image information from the figure and save as a movie frame.

        Doesn't use savefig to be faster: savefig_kwargs will be ignored.
        '''
        try:
            # re-adjust the figure size and dpi in case it has been changed by the
            # user.  We must ensure that every frame is the same size or
            # the movie will not save correctly.
            self.fig.set_size_inches(self._w, self._h)
            self.fig.set_dpi(self.dpi)
            # Draw and save the frame as an argb string to the pipe sink
            self.fig.canvas.draw()
            self._frame_sink().write(self.fig.canvas.tostring_argb())
        except (RuntimeError, IOError) as e:
            out, err = self._proc.communicate()
            raise IOError('Error saving animation to file (cause: {0}) '
                      'Stdout: {1} StdError: {2}. It may help to re-run '
                      'with --verbose-debug.'.format(e, out, err))
def plot():

    fig = plt.figure()
    ims = []
    #cmap = mpl.colors.ListedColormap(['g', 'k'])
    #bounds = [0., 1.]
    #norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    writer = FasterFFMpegWriter()

    for i in range(0,len(all_activated), frame_skip):
        #print(i)
        #print(np.max(all_activated))
        plot = plt.imshow((all_activated[i, 0, :] + all_activated[i, 1, :] + all_activated[i, 2, :]).reshape(nx, ny), vmax=np.max(all_activated), vmin=0, cmap = 'plasma')

        ims.append([plot])


    anim = animation.ArtistAnimation(fig, ims, interval = 200, blit = True, repeat_delay = 100)

    anim.save('network_out/activation.mp4', writer = writer)
    '''
    X = np.arange(nx)
    Y = np.arange(ny)
    X, Y = np.meshgrid(X, Y)
    x, y= X.ravel(), Y.ravel()


    fig = plt.figure()
    ims = []

    print(X.shape, Y.shape)
    for i in range(0,len(all_us), frame_skip):
        #print(i)
        ax = fig.gca(projection = '3d')
        #plot = ax.bar3d(x, y, us[i], 0, 1, 1, shade=True, color = 'b')
        #ax.set_zlim(0,5)
        plot = ax.plot_surface(X, Y, all_us[i, 0, :, :].reshape(nx, ny), rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        #plot = plt.imshow(us[i].reshape(N, N), cmap='plasma')
        #plot = plt.pcolormesh(X, Y, us[i].reshape(N, N), cmap='plasma')
        ims.append([plot])


    anim = animation.ArtistAnimation(fig, ims, interval = 200, blit = True, repeat_delay = 100)
    #anim.save('network_out/AHL_1.mp4', writer = writer)

    fig = plt.figure()
    ims = []

    print(X.shape, Y.shape)
    for i in range(0,len(all_us), frame_skip):
        #print(i)
        ax = fig.gca(projection = '3d')
        #plot = ax.bar3d(x, y, us[i], 0, 1, 1, shade=True, color = 'b')
        #ax.set_zlim(0,5)
        plot = ax.plot_surface(X, Y, all_us[i, 1, :, :].reshape(nx, ny), rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        #plot = plt.imshow(us[i].reshape(N, N), cmap='plasma')
        #plot = plt.pcolormesh(X, Y, us[i].reshape(N, N), cmap='plasma')
        ims.append([plot])


    anim = animation.ArtistAnimation(fig, ims, interval = 200, blit = True, repeat_delay = 100)

    #anim.save('network_out/AHL_2.mp4', writer = writer)
    plt.show()
    '''

def plot_one_AHL():

    fig = plt.figure()
    ims = []
    #cmap = mpl.colors.ListedColormap(['g', 'k'])
    #bounds = [0., 1.]
    #norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    writer = FasterFFMpegWriter()

    for i in range(0,len(all_activated), frame_skip):
        #print(i)
        #print(np.max(all_activated))
        plot = plt.imshow((all_activated[i, 0, :] + all_activated[i, 1, :]).reshape(nx, ny), vmax=np.max(all_activated), vmin=0, cmap = 'plasma')

        ims.append([plot])


    anim = animation.ArtistAnimation(fig, ims, interval = 200, blit = True, repeat_delay = 100)

    anim.save('network_out/activation.mp4', writer = writer)
    '''
    X = np.arange(nx)
    Y = np.arange(ny)
    X, Y = np.meshgrid(X, Y)
    x, y= X.ravel(), Y.ravel()


    fig = plt.figure()
    ims = []

    print(X.shape, Y.shape)
    for i in range(0,len(all_us), frame_skip):
        #print(i)
        ax = fig.gca(projection = '3d')
        #plot = ax.bar3d(x, y, us[i], 0, 1, 1, shade=True, color = 'b')
        #ax.set_zlim(0,5)
        plot = ax.plot_surface(X, Y, all_us[i, 0, :, :].reshape(nx, ny), rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        #plot = plt.imshow(us[i].reshape(N, N), cmap='plasma')
        #plot = plt.pcolormesh(X, Y, us[i].reshape(N, N), cmap='plasma')
        ims.append([plot])


    anim = animation.ArtistAnimation(fig, ims, interval = 200, blit = True, repeat_delay = 100)
    #anim.save('network_out/AHL_1.mp4', writer = writer)

    fig = plt.figure()
    ims = []

    print(X.shape, Y.shape)
    for i in range(0,len(all_us), frame_skip):
        #print(i)
        ax = fig.gca(projection = '3d')
        #plot = ax.bar3d(x, y, us[i], 0, 1, 1, shade=True, color = 'b')
        #ax.set_zlim(0,5)
        plot = ax.plot_surface(X, Y, all_us[i, 1, :, :].reshape(nx, ny), rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        #plot = plt.imshow(us[i].reshape(N, N), cmap='plasma')
        #plot = plt.pcolormesh(X, Y, us[i].reshape(N, N), cmap='plasma')
        ims.append([plot])


    anim = animation.ArtistAnimation(fig, ims, interval = 200, blit = True, repeat_delay = 100)

    #anim.save('network_out/AHL_2.mp4', writer = writer)
    plt.show()
    '''

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def normalise_positions(node_positions):
    # normalise positions so that the grid starts at 0,0
    min_x, min_y = node_positions[0][0]
    max_x, max_y = node_positions[0][1]

    for layer in node_positions:
        for node in layer:
            if node[0] < min_x:
                min_x = node[0]
            if node[1] < min_y:
                min_y = node[1]
            if node[0] > max_x:
                max_x = node[0]
            if node[1] > max_y:
                max_y = node[1]

    min_x -= node_radius
    min_y -= node_radius

    for i, layer in enumerate(node_positions):
        for j, node in enumerate(layer):
            node_positions[i][j][0] -= min_x
            node_positions[i][j][1] -= min_y


    return max_x-min_x+node_radius, max_y-min_y+node_radius

class NodeSimPython(object):
    '''
    MASK FUNCTIONALITY BROKEM FOR CHECKERBOARD: FIX

    '''
    def __init__(self, grid_corners, nx, ny, node_positions, node_radius, D, dtype='float64'):
        """
        Initialize a grid.

        Parameters
        ---------------------
        vertices: (N, 3) Array
            Vertices are stored in an (N, 3) array of floating point numbers.
        elements: (N, 3) Array
            Elements are stored in an (N, 3) array of type 'np.int32'.
        dtype: string
          dtype is either 'float64' or 'float32'. Internally, all structures and the
          vertices are converted to the format specified by 'dtype'. This is useful
          for OpenCL computations on GPUs that do not properly support double precision
          ('float64') types

        """


        '''

        THESE MUST BE SET SO THE dx = dy
        '''
        self.grid_corners = np.array(grid_corners)
        self.nx = nx
        self.ny = ny
        self.dx = (grid_corners[0,1] - grid_corners[0,0]) / (nx-1)
        self.dy = (grid_corners[1,1] - grid_corners[1,0]) / (ny-1)
        self.D = D

        if not (self.dx == self.dy):
            raise Exception('dx should equal dy')

        self.dtype = dtype
        self.iters = 0

        # calculate inverse_jacobian_transpose and integration elements in parrellel

        # use relevant precision
        if self.dtype == 'float64':
            dtype = 'double'

        elif self.dtype == 'float32':
            dtype = 'float'

        self.iterates = []
        self.nIters = 0
        self.callIters = 0
        self.n_vertices = self.nx * self.ny

        self.node_positions = node_positions
        self.node_radius = node_radius
        self.vertex_positions = np.array([self.get_vertex_position(i) for i in range(self.n_vertices)])

    def get_vertex_position(self, node_number):
        node_coordinates = (node_number//self.nx, node_number % self.nx)

        node_position = (self.grid_corners[1, 0] + node_coordinates[0] * self.dy, self.grid_corners[0, 0] + node_coordinates[1] * self.dx)

        return node_position

    def assign_vertices(self, node_positions):
        '''
        assigns mesh vertices to be part of input or output nodes and returns the
        number of vertices inside each node
        '''
        vertices = np.array(self.vertex_positions)[:,:2]


        one_hot_vertices = np.zeros(len(vertices))
        #print('asdfsdafsadfasdf',node_positions[:, None].shape)
        if node_positions[0] is not None:
            node_positions = np.array(node_positions)
            differences = vertices-node_positions[:, None]

            vertices = np.where(np.linalg.norm(differences, axis = 2) < self.node_radius)[1]
            one_hot_vertices[vertices] = 1

        one_hot_vertices = np.array(one_hot_vertices, dtype = np.int32)

        return vertices, one_hot_vertices

    def get_boundary_nodes(self):

        boundaries = np.array(self.grid_corners)
        boundary_nodes = []
        for i in range(self.n_vertices):
            position = self.get_vertex_position(i)

            if np.any(np.abs(boundaries.T - np.array(position)) < self.dx/5): #will get all nodes on the square
                boundary_nodes.append(i)

        return np.array(boundary_nodes)

    def get_A(self):
        N = self.nx # for now only works for square grid
        k = np.array([np.ones(N**2-1),-4*np.ones(N**2),np.ones(N**2-1)])
        offset = [-1,0,1]
        A = diags(k,offset).toarray()

        for i in range(len(A)):
            try:
                A[i][i+N] = 1
            except:
                pass

            try:
                if i-N >= 0:
                    A[i][i-N] = 1
            except:
                pass

        return A*self.D/self.dx**2

    def threshold_on(self, x):
        return sigmoid(x-5)

    def threshold_off(self, x):
        return 1 - sigmoid(x-5)

    def off(self, x):
        return 0*x

    def get_act(self, u):
        AHL1 = u[0,:,:]
        AHL2 = u[1,:,:]

        # layers = [in, h1, h2, out]

        h1_mask, h2_mask, out_mask = self.layer_masks[1:]

        h1_act = self.threshold_on(AHL1).flatten() * h1_mask
        h2_act = self.threshold_off(AHL1).flatten() * h2_mask
        out_act = self.threshold_on(AHL2).flatten() * out_mask
        #print()
        #print(np.sum(h1_act), np.sum(h2_act), np.sum(out_act))

        return np.array([self.input_OH, h1_act + h2_act, out_act])

    def get_act_one_AHL(self, u):
        AHL1 = u[0,:,:]
        AHL2 = u[1,:,:]

        # layers = [in, h1, h2, out]

        h1_mask, h2_mask, out_mask = self.layer_masks[1:]

        h1_act = self.threshold_on(AHL1).flatten() * h1_mask
        h2_act = self.threshold_off(AHL1).flatten() * h2_mask
        out_act = self.threshold_on(AHL1).flatten() * out_mask
        #print()
        #print(np.sum(h1_act), np.sum(h2_act), np.sum(out_act))

        return np.array([self.input_OH + h1_act + h2_act, out_act])

    def simulate(self):

        u = u0

        #

        A = self.get_A() # need to scale A by D and dx, dy to change unit


        input_verts, self.input_OH = self.assign_vertices(self.node_positions[0])


        h1_verts, h1_OH = self.assign_vertices(self.node_positions[1])
        h2_verts, h2_OH = self.assign_vertices(self.node_positions[2])
        output_verts, output_OH = self.assign_vertices(self.node_positions[3])

        self.layer_masks = [self.input_OH, h1_OH, h2_OH, output_OH]

        u = np.zeros((2,self.nx,self.ny))



        # simulation parameters
        dt = 0.1 * self.dx**2/ D #stable up to 0.2

        t_steps = int(tmax//dt)

        print('t_steps: ', t_steps)
        print('dt: ', dt)

        thresh_min = 0.01
        thresh_max = 0.6
        prod_rate = 10000
        deg_rate = 0.000001 * prod_rate
        frame_skip = 1
        all_activated = []
        all_us = [copy.copy(u)]

        # do simulation
        for t in range(t_steps):
            if t%100 == 0:
                print(t)
            activated_nodes = self.get_act_one_AHL(u)
            #print(np.sum(activated_nodes))
            all_activated.append(activated_nodes)

            for i in range(1):#2 AHLs
                du = (A.dot(u[i,:,:].flatten()) + activated_nodes[i] * prod_rate - u[i,:,:].flatten() * deg_rate)
                #print('du: ',np.sum(du))
                u[i,:,:] += dt*du.reshape(nx,ny)
            #print(u)

            u = u.reshape(2,nx,ny)
            # apply insulating boundary conditions
            u[0,0,:] = u[0,1,:]
            u[0,-1,:] = u[0,-2,:]
            u[0,:,0] = u[0,:,1]
            u[0,:,-1] = u[0,:,-2]

            u[1,0,:] = u[1,1,:]
            u[1,-1,:] = u[1,-2,:]
            u[1,:,0] = u[1,:,1]
            u[1,:,-1] = u[1,:,-2]


            all_us.append(copy.copy(u))
        all_activated = np.array(all_activated)
        all_us = np.array(all_us)
        return all_us, all_activated


tmax = int(3) # hours

nx = ny = 100
n_AHLs = 2
D = 3e-6 #cm^2/s
D = 0.03 #mm^2/min LUCA'S ONE

D = D/6000 *60**2 #cm^2/h
node_radius = 0.01
u0 = np.zeros((n_AHLs, nx, ny))

node_positions = np.load('network_out/one_AHL/node_positions.npy', allow_pickle = True)
minimal_model = np.load('network_out/minimal_model.npy', allow_pickle = True)
print(node_positions)
print(minimal_model)
n_h1 = sum(minimal_model[0][:5])
n_h2 = sum(minimal_model[0][5:])

max_x, max_y = normalise_positions(node_positions)
print('normalised grid: ', max_x, max_y)
#grid_corners = np.array([[-0.5,1],[-0.5,1]])
grid_corners = np.array([[-0.2,0.5],[-0.2,0.5]]) #cm

#plot_grid(node_positions, 0, n_h1)
#plt.show()


node_positions = [[node_positions[0][0]], node_positions[1][:n_h1], node_positions[1][n_h2:], node_positions[2]]

grid = NodeSimPython(grid_corners, nx, ny, node_positions, node_radius, D)

all_us, all_activated = grid.simulate()

#print(all_us.shape, all_activated.shape)
frame_skip = 200
plot_one_AHL()
plt.close('all')


X = np.arange(nx)
Y = np.arange(ny)
X, Y = np.meshgrid(X, Y)
x, y= X.ravel(), Y.ravel()


fig = plt.figure()

#print(i)
ax = fig.gca(projection = '3d')
#plot = ax.bar3d(x, y, us[i], 0, 1, 1, shade=True, color = 'b')
#ax.set_zlim(0,5)
plot = ax.plot_surface(X, Y, all_us[-1, 0, :, :].reshape(nx, ny), rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#plot = plt.imshow(us[i].reshape(N, N), cmap='plasma')
#plot = plt.pcolormesh(X, Y, us[i].reshape(N, N), cmap='plasma')
#anim.save('network_out/AHL_1.mp4', writer = writer)
plt.savefig('AHL.png')
fig = plt.figure()

ax = fig.gca(projection = '3d')
#plot = ax.bar3d(x, y, us[i], 0, 1, 1, shade=True, color = 'b')
#ax.set_zlim(0,5)
plot = ax.plot_surface(X, Y, all_us[-1, 1, :, :].reshape(nx, ny), rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#plot = plt.imshow(us[i].reshape(N, N), cmap='plasma')
#plot = plt.pcolormesh(X, Y, us[i].reshape(N, N), cmap='plasma')

#anim.save('network_out/AHL_2.mp4', writer = writer)
plt.show()
'''
def inverse_r(x, a, b):
    return a/(x**b)

func_to_fit = np.load('diffusion_curve.npy')
#func_to_fit = all_us[-1, 0, 49,55:]
#np.save('diffusion_curve.npy', func_to_fit)

x = np.array(range(1,46,1))*grid.dx

params, params_covariance = optimize.curve_fit(inverse_r, x, func_to_fit, p0=[350,0.3])
print(params)

plt.figure()
plt.plot(x, func_to_fit, label = 'diffusion curve')
plt.plot(x,inverse_r(x, 47, 2), label = '1/r')
plt.legend()

plt.show()
'''
