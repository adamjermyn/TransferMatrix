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

s = sys.argv[1]

lefts = [0,0,0,0,0]
rights = [0,1,4,3,2]

# Build the models
acts = []
for i in range(len(lefts)):
	model, blockSize = models.exclusiveActinModelRuleFactory(lefts[i], rights[i], sides='left')
	act = models.actin(model, [0,1], sp.symbols('a b c'), blockSize=blockSize)
	acts.append(act)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
del colors[4]
colors[4], colors[3] = colors[3], colors[4]


strs = ['Output/' + s + '_' + str(l) + '_' + str(r) for l,r in zip(*(lefts,rights))]
styles = ['-.','--','-',':',(0, (3, 1, 1, 1))]

fits = []
for st in strs:
	f = np.loadtxt(st + '_exclusive_left_summary.txt')
	q = f[0][0]
	w = f[1][0]
	j = f[2][0]
	fits.append((q,w,j))

def rn(x):
	return round(x,1)

labels = ['Range = ' + str(r-l) + ', J = ' + str(rn(f[2])) + \
			', Q = ' + str(rn(f[0])) + ', W = ' + str(rn(f[1]))\
		  for l,r,f in zip(*(lefts,rights,fits))]


c,bindingF,length,dl,name = data(s)

plt.figure(figsize=(5,4))

plt.scatter(bindingF,length)
plt.errorbar(bindingF,length,yerr=dl, linestyle='None')

counter = 0
for st, params, act, style, label in zip(*(strs, fits, acts, styles, labels)):
	q, w, j = params
	bFhighRes = np.linspace(min(bindingF),max(bindingF),num=100,endpoint=True)
	fHR = np.zeros(100)
	for i,x in enumerate(bFhighRes):
		p = act.bindingFinder((0,q,w),x)[0]
		_, fHR[i], _, _, interQ, d_lnZ_dQ_intensive, interW, d_lnZ_dW_intensive = act.fN((p,q,w))
		print(x, interQ, d_lnZ_dQ_intensive, interW, d_lnZ_dW_intensive)
	mlHighRes = models.meanL(c,fHR,j)
	line, = plt.plot(bFhighRes,mlHighRes, linewidth=3, label=label)
	plt.setp(line, linestyle=style, c=colors[counter])
	counter += 1

handles, labels = plt.gca().get_legend_handles_labels()
order = [0,1,4,3,2]
handles = list(handles[order[i]] for i in range(len(order)))
labels = list(labels[order[i]] for i in range(len(order)))
plt.xlabel('Cofilin Binding Fraction')
plt.ylabel('Filament length ($\mu m$)')
plt.legend(handles,labels)
#plt.tight_layout()
plt.savefig('fig_combined_'+s+'.pdf')
