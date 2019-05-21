import sys
sys.path.append('/home/neythen/Desktop/Projects/synbiobrain')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
from diffusion_sim import FasterFFMpegWriter


GFP_ts = np.load('/home/neythen/Desktop/Projects/synbiobrain/finite_difference/results/cool_square/output/GFP_ts.npy')
AHL_ts = np.load('/home/neythen/Desktop/Projects/synbiobrain/finite_difference/results/cool_square/output/AHL_ts.npy')

ims = []
cmap = mpl.colors.ListedColormap(['g', 'k'])
bounds = [0., 0.5, 1.]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
writer = FasterFFMpegWriter(fps=8, metadata=dict(artist='Me'), bitrate=1800)
delta_t = 5
frame_skip = 500

fig = plt.figure()
for i in range(0,len(GFP_ts), frame_skip):
    print(i)
    fig.suptitle('time: ' + str(delta_t*i))


    plot = plt.imshow(GFP_ts[i].reshape(10, 10), cmap = 'inferno')

    ims.append([plot])


anim = animation.ArtistAnimation(fig, ims, interval = 200, blit = True, repeat_delay = 100)


plt.show()
anim.save('cool_square/finite_difference_GFP.mp4', writer = writer)



fig = plt.figure()
ims = []
for i in range(0,len(AHL_ts), frame_skip):
    print(i)
    #ax = fig.gca(projection = '3d')
    #plot = ax.bar3d(x, y, bottom, width, depth, Us[i], shade=True, color = 'b')
    fig.suptitle('time: ' + str(delta_t*i))
    #plot = ax.plot_surface(X, Y, Us[i].reshape(grid.nx, grid.ny), rstride=1, cstride=1, cmap=cm.coolwarm,
    #    linewidth=0, antialiased=False)
    plot = plt.imshow(AHL_ts[i].reshape(10, 10), cmap='plasma')

    ims.append([plot])

anim = animation.ArtistAnimation(fig, ims, interval = 200, blit = True, repeat_delay = 100)
plt.show()
anim.save('cool_square/finite_difference_AHL.mp4', writer = writer)
