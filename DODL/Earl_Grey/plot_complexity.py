import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt


SMALL_SIZE = 11
MEDIUM_SIZE = 14
BIGGER_SIZE = 17

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

width = 0.1
n_inputs = np.arange(1, 12)

n_funcs = 2**2**n_inputs

n_cells_ours = n_inputs + 4

n_colonies = n_inputs + np.ceil((n_inputs-1)/2) + 1
n_output_colonies =  np.ceil((n_inputs-1)/2) + 1

n_modules = 2**(n_inputs -1)

n_cells = 2*n_inputs + 1

n_modulators = 3*n_inputs

n_branches = 2**(n_inputs -1)

plt.bar(n_inputs-3.5*width, n_cells_ours, width, label = 'Cell types $= n+ 4$', color='blue')
plt.bar(n_inputs - 2.5*width, n_colonies, width, label = 'Colonies $=n + \\lceil{\\frac{n-1}{2}}\\rceil + 1$', color='lightblue')

plt.bar(n_inputs - 0.5*width, n_modulators, width, label = 'Cell types $=3n$',color='green')
plt.bar(n_inputs + 0.5*width, n_branches, width, label = 'Branches $=2^{n-1}$', color='lightgreen')

plt.bar(n_inputs + 2.5*width, n_cells, width, label = 'Cell types $=2n+1$', color='red')
plt.bar(n_inputs + 3.5*width, n_modules, width, label = 'Modules $=2^{n-1}$', color='pink')




#plt.bar(n_inputs, n_funcs)
#plt.legend()
plt.xlabel('Number of inputs $(n)$')
plt.ylabel('Count')

plt.savefig('complexity1.png', dpi = 300)
plt.show()