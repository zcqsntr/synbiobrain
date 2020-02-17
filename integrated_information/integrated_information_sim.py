import numpy as np
from scipy.sparse import diags
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.backends.backend_pdf
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from phi_empirical_x1 import *

node_dim = 3 # dimension in terms of number of  nodes
buffer = 2 # number of elemnets between grid and the boundary of simulation
N = node_dim + 2*buffer # we simulate an N*N grid



def create_A(N): # the diffusion matrix

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
    # function of the sender node, for now all senders have the same input function, but this could be changed
    if (t//1)%2 == 0:
        return 1
    else:
        return 0

def send_func_sin(t):
    # function of the sender node, for now all senders have the same input function, but this could be changed
    return np.abs(np.sin(t/scale))


def get_act(u):
    # gets the activated nodes
    rec_prod = np.ones_like(u)
    #print(u)

    rec_prod[u > thresh_max] = 0
    rec_prod[u < thresh_min] = 0
    rec_prod *= rec_mask

    send_prod = send_func_sin(t)*send_mask

    return rec_prod + send_prod


def plot():

    fig = plt.figure()
    ims = []
    cmap = mpl.colors.ListedColormap(['g', 'k'])
    bounds = [0., 1.]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    for i in range(0,len(all_activated), frame_skip):
        #print(i)
        plot = plt.imshow(all_activated[i].reshape(node_dim, node_dim), cmap = 'inferno')

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

for scale in [100000, 1000000]:

    D = 3e-6 #cm^2/s


    A = create_A(N)*D # get the diffusion matrix
    print(A)

    # dU = Au + prod + deg

    # this tells which nodes are recievers
    rec_mask = np.zeros((N,N))
    rec_mask[buffer: buffer + node_dim, buffer: buffer + node_dim] = 1

    rec_mask[buffer,buffer] = 0
    print(rec_mask)
    rec_mask = rec_mask.flatten()


    #this tells which nodes are senders
    send_mask = np.zeros((N,N))
    send_mask[buffer,buffer] = 1
    print(send_mask)
    send_mask = send_mask.flatten()

    u0 = np.zeros((N,N))

    u = u0.flatten()
    us = [u]
    deg_rate = 0.1*D

    # simulation parameters
    dt = 0.02/ D #stable up to 0.2

    print(' delta t (hours)', dt /(60**2))
    tmax = int(1e4)
    thresh_min = 0.01
    thresh_max = 0.6
    prod_rate = 1*D
    frame_skip = 1
    all_activated = []

    # do simulation
    for t in range(tmax):
        activated_nodes = get_act(u)
        all_activated.append(activated_nodes.reshape(N,N)[buffer: buffer + node_dim, buffer: buffer + node_dim].flatten())
        u = u + dt*(A.dot(u) + activated_nodes * prod_rate - u * deg_rate)
        #print(u)

        u = u.reshape(N,N)
        # apply insulating boundary conditions
        u[0,:] = u[1,:]
        u[-1,:] = u[-2,:]
        u[:,0] = u[:,1]
        u[:,-1] = u[:,-2]
        u = u.flatten()
        us.append(u)
    all_activated = np.array(all_activated)
    # all_activated.shape = (tmax, N)
    print(all_activated.shape)

    #plot() # comment this to stop plotting


    taus = range(10)
    IIs = []
    MIs = []
    II_modified = []
    for i, tau in enumerate(taus):
        print(i)
        results = int_inf(all_activated, tau, np.arange(tmax), 'binary', 0.5)
        IIs.append(results[4])
        MIs.append(results[5])
        II_modified.append(results[6])
        print()

    np.save('IIs_' + str(scale) + '.npy', np.array(IIs))
    np.save( 'MIs_' + str(scale) + '.npy', np.array(MIs))
    np.save( 'IIs_modified_' + str(scale) + '.npy', np.array(II_modified))
    plt.figure()
    taus = np.array(taus)*dt/(60**2)
    plt.plot(taus, IIs, label = 'II')
    plt.plot(taus, MIs, label = 'MI')
    plt.plot(taus, II_modified, label = 'II mod')
    plt.legend()
    plt.xlabel('tau (hours)')
    plt.savefig('Int_inf_' + str(scale) + '.png')
        #plt.show()

# vary the tau and look at the pattern
# could try e.g. sin(x), lots of ones
# decoder based II in a paper, eventually move up to this
# often use first minimum of mutual information or a region around first minimum
# amount of data can matter so play around with this start at 10^6 also check how it varies with number of data choose number once the MI is smooth
#  another method (Queyrannes algrithm )  to make it faster, prevents exhaustive search of partitions, in references
