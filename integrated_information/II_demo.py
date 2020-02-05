import numpy as np
from scipy.sparse import diags
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.backends.backend_pdf
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from phi_empirical_x1 import *
import math




n_points = int(1e6)


IIs = []
taus =  np.arange(0.1, 2*math.pi, 0.5)
print(taus)
for tau in taus:

    tspan = np.arange(0, tau*n_points, tau)[:n_points]

    s = np.sin(tspan).reshape(n_points,1)
    c = np.cos(tspan).reshape(n_points,1)
    ones = np.ones((n_points, 1))
    y = np.hstack((s,c))
    #y = np.hstack((ones, ones, ones))
    #y = s
    print(y.shape) # needs to be (n_tsteps, n_channels)
    result = int_inf(y, 1, tspan, 'binary', 0.5)
    IIs.append(result[4])

plt.plot(taus, IIs)
plt.show()
