import sys
sys.path.append('/home/neythen/Desktop/Projects/synbiobrain')
from diffusion_sim import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
os.environ['PYOPENCL_CTX'] = '0'

GFP_ts = np.load('/home/neythen/Desktop/Projects/synbiobrain/finite_element/results/failed_mush/failed_mush/GFP_ts.npy')
AHL_ts = np.load('/home/neythen/Desktop/Projects/synbiobrain/finite_element/results/failed_mush/failed_mush/AHL_ts.npy')
writer = FasterFFMpegWriter(fps=8, metadata=dict(artist='Me'), bitrate=1800)

grid = SynBioBrainFE.from_file("/home/neythen/Desktop/Projects/synbiobrain/finite_element/meshes/9by9_buffer.vtk", 'float32')


ims = []
frame_skip = 2000
print(len(GFP_ts))
delta_t = 5
'''
fig = plt.figure()
for i in range(0,len(GFP_ts), frame_skip):
    print(i)

    fig.suptitle('time: ' + str(delta_t*i))
    ax = fig.gca(projection = '3d')
    ax.view_init(elev = 90,azim=-90)
    plot = ax.plot_trisurf(grid.vertices[:,0], grid.vertices[:,1], GFP_ts[i], shade=False, cmap=cm.coolwarm, linewidth=0)
    ims.append([plot])


anim = animation.ArtistAnimation(fig, ims, interval = 200, blit = True, repeat_delay = 100)

#anim.save('/home/neythen/Desktop/Projects/synbiobrain/finite_element/results/failed_mush/failed_mush/finite_element_GFP.mp4', writer = writer)
plt.show()

'''


fig = plt.figure()
ax = fig.gca(projection = '3d')

plot = ax.plot_trisurf(grid.vertices[:,0], grid.vertices[:,1], AHL_ts[-1], shade=False, cmap=cm.coolwarm, linewidth=0)
plt.show()
ims = []
fig = plt.figure()
for i in range(0,len(AHL_ts), frame_skip):
    print(i)

    ax = fig.gca(projection = '3d')
    fig.suptitle('time: ' + str(delta_t*i))
    plot = ax.plot_trisurf(grid.vertices[:,0], grid.vertices[:,1], AHL_ts[i], shade=False, cmap=cm.coolwarm, linewidth=0)
    ims.append([plot])

anim = animation.ArtistAnimation(fig, ims, interval = 200, blit = True, repeat_delay = 100)
#anim.save('/home/neythen/Desktop/Projects/synbiobrain/finite_element/results/failed mush/finite_element_GFP.mp4', writer = writer)
plt.show()
