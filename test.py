import transferMatrixClass as tmc
import models
import itertools as it
import numpy as np

model = models.actinModelFactory([-1,0,1],[0])
#model = models.actinModelFactory([0,1],[-1,0,1])
#model = models.eFuncLongIsing
#model = models.eFuncTwoPlaneActinModel
pvals = np.array([0.5,1.,2.])


stateRange = [0,1]
params = tmc.sp.symbols('p q w')
blockSize = 3
#tm,left,right = tmc.transferMatrix(model,stateRange,params,blockSize)
tm,left,right,blockSize = tmc.transferMatrixVariableSize(model,stateRange,params)
print(blockSize)
exit()
T,eL,eR,dT = tmc.wrapper(tm,left,right,params)

def transfer(n):
	return tmc.fN(T, eL, eR, dT, pvals, blockSize, n)

def explicit(m, n):
	vals = []
	for i in it.product([0,1],repeat=n):
		vals.append(-m(i,pvals))
	s = np.sum(np.exp(vals))
	return -np.log(s)

e1 = explicit(model, 9)
e2 = explicit(model, 10)
print(transfer(10))
print(e1, e2, e2 - e1)

print('----')

print(blockSize)
print(transfer(2*blockSize+2))
print(explicit(model,2*blockSize+2))

print('----')

print(pvals)

params = pvals

# Evaluate endcaps at partition sizes
e1 = eL[0](*params)
e2 = eR[0](*params)

# Evaluate transfer matrix and derivative
tm = T(*params)
dt = dT(*params)

# Cast to numpy
e1 = np.array(e1.tolist(), dtype=float)
e2 = np.array(e2.tolist(), dtype=float)
tm = np.array(tm.tolist(), dtype=float)
dt = np.array(dt.tolist(), dtype=float)

print(e1.shape, e2.shape)

from collections import Counter
cnt1 = Counter()
cnt2 = Counter()
for i in it.product([0,1],repeat=8):
	ev = model(i,pvals)
	tv = -np.log(e1[i[0],4*i[1]+2*i[2]+i[3]]*tm[4*i[1]+2*i[2]+i[3],4*i[4]+2*i[5]+i[6]]*e2[4*i[4]+2*i[5]+i[6],i[7]])
	cnt1[ev] += 1
	cnt2[tv] += 1
	print(i,ev-tv,ev,tv)

print(cnt1)
print(cnt2)