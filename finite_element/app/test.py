from diffusion_sim import *
import numpy as np

import matplotlib.pyplot as plt
initial_pop = 1
max_growth_rate = 0.1
carrying_capacity = 50
tmax = 100
dt = 0.8


time_series = get_population_timeseries(initial_pop, max_growth_rate, carrying_capacity, tmax, dt)
print(time_series)

plt.plot(time_series)
plt.show()
