import numpy as np
from scipy.sparse import diags
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.backends.backend_pdf
import matplotlib as mpl
import matplotlib.pyplot as plt
from phi_empirical_x1 import *

node_dim = 3 # dimension in terms of nodes
buffer = 2 # number of elemnets between grid and boundary of simulation
N = node_dim + 2*buffer
def create_A():

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
    return A

def send_func(t):
    if (t//1)%1 == 0:
        return 1
    else:
        return 0

def get_act(u):
    rec_prod = np.ones_like(u)
    rec_prod[u > thresh_max] = 0
    rec_prod[u < thresh_min] = 0
    rec_prod *= rec_mask

    send_prod = send_func(t)*send_mask

    return rec_prod + send_prod


def plot():

    fig = plt.figure()
    ims = []
    cmap = mpl.colors.ListedColormap(['g', 'k'])
    bounds = [0., 1.]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    for i in range(0,len(all_activated), frame_skip):
        #print(i)
        plot = plt.imshow(all_activated[i].reshape(N, N), cmap = 'inferno')

        ims.append([plot])


    anim = animation.ArtistAnimation(fig, ims, interval = 200, blit = True, repeat_delay = 100)
    plt.show()



    X = np.arange(N)
    Y = np.arange(N)
    X, Y = np.meshgrid(X, Y)
    x, y= X.ravel(), Y.ravel()


    fig = plt.figure()
    ims = []

    print(X.shape, Y.shape)
    for i in range(0,len(us), frame_skip):
        #print(i)
        ax = fig.gca(projection = '3d')
        #plot = ax.bar3d(x, y, us[i], 0, 1, 1, shade=True, color = 'b')
        ax.set_zlim(0,5)
        plot = ax.plot_surface(X, Y, us[i].reshape(N, N), rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        #plot = plt.imshow(us[i].reshape(N, N), cmap='plasma')
        #plot = plt.pcolormesh(X, Y, us[i].reshape(N, N), cmap='plasma')
        ims.append([plot])


    anim = animation.ArtistAnimation(fig, ims, interval = 200, blit = True, repeat_delay = 100)

    plt.show()


A = create_A()
print(A)

# dU = Au + prod + deg


rec_mask = np.zeros((N,N))
rec_mask[buffer: buffer + node_dim, buffer: buffer + node_dim] = 1

rec_mask[buffer,buffer] = 0
print(rec_mask)
rec_mask = rec_mask.flatten()



send_mask = np.zeros((N,N))
send_mask[buffer,buffer] = 1
print(send_mask)
send_mask = send_mask.flatten()

u0 = np.zeros((N,N))
u0[buffer, buffer] = 1
u = u0.flatten()
us = [u]
deg_rate = 0.1

dt = 0.1 #stable up to 0.2
tmax = 10000
thresh_min = 0.01
thresh_max = 0.6
prod_rate = 1
frame_skip = 1
all_activated = []

for t in range(tmax):
    activated_nodes = get_act(u)
    all_activated.append(activated_nodes.reshape(N,N)[buffer: buffer + node_dim, buffer: buffer + node_dim].flatten())


    u = u + dt*(A.dot(u) + activated_nodes * prod_rate - u * deg_rate)

    u = u.reshape(N,N)
    #insulating boundary conditions
    u[0,:] = u[1,:]
    u[-1,:] = u[-2,:]
    u[:,0] = u[:,1]
    u[:,-1] = u[:,-2]
    u = u.flatten()
    us.append(u)
all_activated = np.array(all_activated)
print(int_inf(all_activated, 1, np.arange(tmax), 'binary', 0.5))
