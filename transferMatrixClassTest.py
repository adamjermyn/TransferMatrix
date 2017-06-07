import transferMatrixClass as tmc
import transferFast
import models
import itertools as it
import numpy as np

stateRange = [0,1]
params = tmc.sp.symbols('p q w')
tm,left,right,blockSize = tmc.transferMatrixVariableSize(models.eFuncOnePlaneActinModel,stateRange,params)
T,eL,eR = tmc.wrapper(tm,left,right,params)
print tmc.fN(T,eL,eR,[0,4,0],blockSize,6)

s = 0
for i in it.product([0,1],repeat=6):
	s += np.exp(-models.eFuncOnePlaneActinModel(i,[0,4,0]))
print -np.log(s)
