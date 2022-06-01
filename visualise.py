import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

perfdir = './performance-and-animations/'
targetdir = './performance-and-animations/'

#Rewards 
rewards_list = pd.read_csv(perfdir + 'rewards_list.dat', sep="\s+", names=['rewards'])
rewards_list.describe()

#Steps
nr_steps = pd.read_csv(perfdir + '/nr_steps_list.dat', sep="\s+", names=['nr_steps'])
nr_steps.describe()

'''
Steps hist
'''
steps_hist = pd.read_csv(perfdir + 'steps_histogram.dat', sep="\s+", names=['steps_bin_edges', 'steps_hst'])
fig = plt.step(steps_hist.loc[:, 'steps_bin_edges'],
         steps_hist.loc[:, 'steps_hst'], 
         where='pre',
         color='b',
        #  linestyle='.', 
         )

plt.savefig(targetdir + 'steps_histogram_dense.png')
plt.close()

'''
Rewards hist
Was created as follows:
    rewards_hst, rewards_bin_edges = np.histogram(rewards_lst, bins=50, density=True)
    rewards_hst = np.reshape(rewards_hst, (-1, 1))
    rewards_bin_edges = np.reshape(rewards_bin_edges[:-1], (-1, 1))

    steps_hst, steps_bin_edges = np.histogram(nr_steps_lst, bins=50, density=True)
    steps_hst = np.reshape(steps_hst, (-1, 1))
    steps_bin_edges = np.reshape(steps_bin_edges[:-1], (-1, 1))

    perfdir = './performance-and-animations/'
    np.savetxt(perfdir + 'rewards_histogram.dat',
            np.concatenate((rewards_bin_edges, rewards_hst), axis=1)
            )
    np.savetxt(perfdir + 'steps_histogram.dat',
            np.concatenate((steps_bin_edges, steps_hst), axis=1)
            )
    np.savetxt(perfdir + 'rewards_list.dat',
            rewards_lst
            )
    np.savetxt(perfdir + 'nr_steps_list.dat',
            nr_steps_lst
            )
'''
rewards_hist = pd.read_csv(perfdir + 'rewards_histogram.dat', sep="\s+", names=['rewards_bin_edges', 'rewards_hst'])
fig = plt.step(rewards_hist.loc[:, 'rewards_bin_edges'],
         rewards_hist.loc[:, 'rewards_hst'], 
         where='pre',
         color='g',
        #  linestyle='.', 
         )

plt.savefig(targetdir + 'rewards_histogram_dense.png')
plt.close()



