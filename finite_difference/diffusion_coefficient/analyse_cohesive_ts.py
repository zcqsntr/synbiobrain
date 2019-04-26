import numpy as np
import matplotlib.pyplot as plt
cohesive_ts = np.load('all_cohesive_ts.npy')
proportions = []
barriers = ['1', '0.8', '0.6', '0.4', '0.2', '0.15', '0.1', '0.05', '0.01']

barriers = np.array([float(b) for b in barriers])
for diff_coeff in cohesive_ts:
    un_cohesive_steps = 0
    for proportion in diff_coeff:
        if proportion != 100:
            un_cohesive_steps += 1

    proportions.append(un_cohesive_steps/len(diff_coeff))

plt.plot(1-barriers, proportions)
plt.ylabel('Proportion of time spent switching')
plt.xlabel('1 - P')
plt.ylim(bottom = 0)

plt.show()
