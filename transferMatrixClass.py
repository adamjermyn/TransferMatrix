import numpy as np
import sympy as sp
from sympy import exp
import itertools as it
from numpy import real,log

def partition(n, blockSize):
	'''
	This function takes as input:
		n 			-	The size of the system of interest.
		blockSize	-	The size of the blocks of the transfer matrix.

	It returns a tuple containing the size of the left endcap, the number of
	blocks, and the size of the right endcap in that order. The right endcap
	is always chosen to have size 1.
	'''
	if n >= 2+blockSize:
		m = n - 1
		q = m % blockSize
		if q == 0:
			q += blockSize
		return q,(n-1-q)/blockSize,1
	else:
		return None


def transferMatrix(eFunc, stateRange, params, blockSize,check=True):
	# eFunc must accept an array of states, each drawn from stateRange

	# Calculate transfer matrix
	tMat = sp.zeros(len(stateRange)**blockSize,len(stateRange)**blockSize)
	for i,x in enumerate(it.product(stateRange,repeat=blockSize)):
		print(i)
		for j,y in enumerate(it.product(stateRange,repeat=blockSize)):
			tMat[i,j] = exp(eFunc(y+y,params) - eFunc(x+y+y,params))
			if check:
				for z in it.product(stateRange,repeat=blockSize):
					if exp(eFunc(y+z,params) - eFunc(x+y+z,params)) != tMat[i,j]:
						return None
	# Calculate left end cap
	leftEnds = []
	for l in range(1,blockSize+1):
		print(l)
		end = sp.zeros(len(stateRange)**l,len(stateRange)**blockSize)
		for i,x in enumerate(it.product(stateRange,repeat=l)):
			for j,y in enumerate(it.product(stateRange,repeat=blockSize)):
				end[i,j] = exp(eFunc(y+y,params) - eFunc(x+y+y,params))
		leftEnds.append(end)

	# Calculate right end cap
	rightEnds = []
	for l in range(1,blockSize+1):
		print(l)
		end = sp.zeros(len(stateRange)**blockSize,len(stateRange)**l)
		for i,x in enumerate(it.product(stateRange,repeat=blockSize)):
			for j,y in enumerate(it.product(stateRange,repeat=l)):
				end[i,j] = exp(-eFunc(x+y,params))
		rightEnds.append(end)

	return tMat,leftEnds,rightEnds

def transferMatrixVariableSize(eFunc, stateRange, params):
	blockSize = 1
	while True:
		tm = transferMatrix(eFunc, stateRange, params, blockSize)
		if tm is not None:
			print('Final Size:',blockSize)
			return tm[0],tm[1],tm[2],blockSize
		blockSize += 1
		print(blockSize)

def wrapper(tm,leftEnds,rightEnds,params):
	T = sp.lambdify(params,tm,modules='mpmath')
	eL = [sp.lambdify(params,l,modules='mpmath') for l in leftEnds]
	eR = [sp.lambdify(params,r,modules='mpmath') for r in rightEnds]
	return T,eL,eR

def fN(T,eL,eR,params,blockSize,n):
	# Inputs must be wrapped already
	n1,n2,n3 = partition(n,blockSize)
	e1 = eL[n1-1](*params)
	e2 = eR[n3-1](*params)
	tm = T(*params)
	e1m = sp.Max(e1)
	e2m = sp.Max(e2)
	tmm = sp.Max(tm)
	e1 /= e1m
	e2 /= e2m
	tmm /= tmm
	e1 = np.array(eL[n1-1](*params))
	e2 = np.array(eR[n3-1](*params))
	tm = np.array(T(*params))
	m = max(np.linalg.eigvals(tm))
	z = np.sum(e1.dot(np.linalg.matrix_power(tm/m,n2-1)).dot(e2))
	f = -real(log(e1m) + log(e2m) + log(z) + (log(m)+log(tmm))*(n2-1))
	slope = -real(log(m)/blockSize)
	inter = f - slope*n
	return slope,inter,f
