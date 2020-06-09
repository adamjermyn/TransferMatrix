import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import transferMatrixClass as tmc
import models


m1 = models.eFuncTwoPlaneActinModel
#m2 = models.actinModelFactory([0,1],[-1,0,1,2])
m2 = models.actinModelFactory([-1,0],[-3,-2,-1,0])

for _ in range(100):
	s = np.random.randint(0, high=2,size=10)
	params = [0, np.random.randn(), np.random.randn()]
	print(s,params,m1(s,params),m2(s,params))
	assert m1(s,params) == m2(s,params)