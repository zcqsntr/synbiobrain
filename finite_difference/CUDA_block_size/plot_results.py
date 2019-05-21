import matplotlib.pyplot as plt


average_times = [11.532449937,  8.680184841,  7.296907139,  6.583113742,  6.182232356, 6.03160305,  6.041518283,  6.022027588,  6.03643806,  6.057223225,  6.052840853]
block_sizes = [1,2,4,8,16,32,64,128,256,512,1024]

plt.semilogx(block_sizes, average_times, basex = 2)

plt.xlabel('Block size')
plt.ylabel('Average time taken (s)')
plt.show()
