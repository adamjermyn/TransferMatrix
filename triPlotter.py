import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import sys

modelSTR = 'Output/' + sys.argv[1] + '_' + sys.argv[2]

samples = np.loadtxt(modelSTR)
samples *= -1

fig, _ = plt.subplots(nrows=3, ncols=3, figsize=(5,6))
corner(samples, labels=["$Q$", "$W$","$J$", "$\ln\,f$"], fig=fig)
fig.savefig(modelSTR+'_'+'triangle'+sys.argv[2]+'.pdf')

