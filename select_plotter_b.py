from data import data
from scipy.optimize import brentq
from numpy import exp,log,real
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import transferMatrixClass as tmc
import models
import sympy as sp
import sys

# Can specify s = 'h' or 'd' for the two kinds of actin
# The models we use are:
# short - eFuncTwoPlaneVeryShortActinModel
# medium - eFuncTwoPlaneShortActinModel
# long - eFuncTwoPlaneActinModel

s = 'b'

lefts = [-1,-2,-1,0,-4]
rights = [1,1,3,2,0]


# Build the models
acts = []
for i in range(len(lefts)):
	model, blockSize = models.exclusiveActinModelRuleFactory(lefts[i], rights[i], sides='left')
	act = models.actin(model, [0,1], sp.symbols('a b c'), blockSize=blockSize)
	acts.append(act)

strs = ['Output/' + s + '_' + str(l) + '_' + str(r) for l,r in zip(*(lefts,rights))]
labels = [str(r) + ',' + str(l) for l,r in zip(*(lefts,rights))]

fits = []
for st in strs:
	f = np.loadtxt(st + '_inclusive_summary.txt')
	q = f[0][0]
	w = f[1][0]
	j = f[2][0]
	fits.append((q,w,j))

c,bindingF,length,dl,name = data(s)

fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)

ax.scatter(bindingF,length)
ax.errorbar(bindingF,length,yerr=dl, linestyle='None')

NUM_COLORS = len(lefts)
cm = plt.get_cmap('winter')

ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
for st, params, act, label in zip(*(strs, fits, acts, labels)):
	q, w, j = params
	bFhighRes = np.linspace(min(bindingF),max(bindingF),num=100,endpoint=True)
	fHR = np.zeros(100)
	for i,x in enumerate(bFhighRes):
		p = act.bindingFinder((0,q,w),x)[0]
		fHR[i] = act.fN((p,q,w))[1]
		if i%10==0:
			print(p)
	mlHighRes = models.meanL(c,fHR,j)
	line, = ax.plot(bFhighRes,mlHighRes, linewidth=1, label=label)

ax.set_xlim([0,1.3])
ax.set_xlabel('Cofilin Binding Fraction')
ax.set_ylabel('Filament length ($\mu m$)')
ax.legend()
plt.tight_layout()
plt.savefig('selected_'+s+'.pdf')
