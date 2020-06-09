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
postfix = sys.argv[2]

lefts, rights = np.meshgrid(range(-4,1), range(0,5), indexing='ij')

lefts = np.reshape(lefts, (-1,))
rights = np.reshape(rights, (-1,))

ls = []
rs = []
for i in range(len(lefts)):
	l = lefts[i]
	r = rights[i]
	st = 'Output/' + s + '_' + str(l) + '_' + str(r)
	try:
		np.loadtxt(st + '_' + postfix + '_summary.txt')
		ls.append(l)
		rs.append(r)
	except:
		print('File not found for: ',l,r)

lefts = ls
rights = rs


# Build the models
acts = []
for i in range(len(lefts)):
	model, blockSize = models.actinModelRuleFactory(lefts[i], rights[i])
	act = models.actin(model, [0,1], sp.symbols('a b c'), blockSize=blockSize)
	acts.append(act)

strs = ['Output/' + s + '_' + str(l) + '_' + str(r) for l,r in zip(*(lefts,rights))]
labels = [str(l) + ',' + str(r) for l,r in zip(*(lefts,rights))]

fits = []
for st in strs:
	f = np.loadtxt(st + '_' + postfix + '_summary.txt')
	q = f[0][0]
	w = f[1][0]
	j = f[2][0]
	fits.append((q,w,j))

c,bindingF,length,dl,name = data(s)

fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)

ax.scatter(bindingF,length)
ax.errorbar(bindingF,length,yerr=dl, linestyle='None')

NUM_COLORS = len(ls)
cm = plt.get_cmap('winter')

counter = 0
ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
for st, params, act, label, l, r in zip(*(strs, fits, acts, labels, lefts, rights)):
	if l != -3 or r != 0:
		continue
	q, w, j = params
	bFhighRes = np.linspace(min(bindingF),max(bindingF),num=100,endpoint=True)
	fHR = np.zeros(100)
	for i,x in enumerate(bFhighRes):
		p = act.bindingFinder((0,q,w),x)[0]
		fHR[i] = act.fN((p,q,w))[1]
		print(x,p,q,w, act.fN((p,q,w)))
		if i%10==0:
			print(p)
	mlHighRes = models.meanL(c,fHR,j)
	if abs(l) < abs(r):
		line, = ax.plot(bFhighRes,mlHighRes, linewidth=1, label=label, linestyle='-')
	elif abs(l) == abs(r):
		line, = ax.plot(bFhighRes,mlHighRes, linewidth=1, label=label, linestyle='-.')		
	else:
		line, = ax.plot(bFhighRes,mlHighRes, linewidth=1, label=label, linestyle=':')
	counter += 1

ax.set_xlim([0,1.3])
ax.set_xlabel('Cofilin Binding Fraction')
ax.set_ylabel('Filament length ($\mu m$)')
ax.legend()
plt.tight_layout()
plt.savefig('all_combined_'+s+'_' + postfix + '.pdf')
