import numpy as np
import sympy as sp
from sympy import exp
import itertools as it
from numpy import real, log
from scipy.linalg import eig
import mpmath as mp

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
		return q,(n-1-q)//blockSize,1
	else:
		return None

def transferMatrix(eFunc, stateRange, params, blockSize, check=True):
	'''
	This function computes the transfer matrix associated with a Hamiltonian.
	The input parameters are:
		eFunc 		-	The Hamiltonian. This function must take as input a list of
						items drawn from stateRange and a list of parameters given
						by params.
		stateRange	-	This is a list specifying the allowed states on a site.
		params 		-	This is a list of the parameters in the Hamiltonian.
						These are sympy variables.
		blockSize	-	Transfer matrices generally require blocking together
						neighbouring sites. This is an integer specifying
						how many sites to block together. If this is set
						incorrectly the result will generically be wrong.
						This will never exceed the maximum range of the
						interactions in the system.
		check		-	An optional argument which defaults to True. If True
						the code will perform a basic check to see if the
						block size is correct. This is done just by examining the
						energy associated with a block which is 50% larger.

	This method implements a brute-force approach to finding the transfer matrix
	for a system. This works by constructing three neighbouring blocks of size blockSize.
	The states of these blocks are x, y, and y. For each (x,y) we compute eFunc(x,y,y).
	We also compute eFunc(y,y), the energy of two blocks of size blockSize in the
	same state. We subtract these and exponentiate, and take this to be the value of the
	transfer matrix for the state pair (x,y).

	The same method is then used to compute so-called endcap matrices. These matrices
	give the contribution of blocks smaller than blockSize which are attached to the
	ends of the system, so that we can deal with system sizes which are not integer
	multiples of the block size. The left endcap is computed with precisely the same
	logic, while the right endcap just returns exp(-eFunc(x + y)) with no subtraction.
	This has to be done at one endcap to anchor the energies, and we choose to do it 
	at the right one.

	This method relies on the range of interactions being at most blockSize. This is so
	that eFunc(x, y, y) - eFunc(y, y) is precisely the contribution to the energy
	associated with adding on the block x to a block y, irrespective of the rest of the
	system. As a result if there are interactions which are longer-ranged than blockSize
	then this result will be invalid

	The mathematical operations are all performed in sympy so that this expensive operation
	need only be performed once.
	'''

	# Generate block states
	states = list(it.product(stateRange, repeat=blockSize))

	# Initialize matrix
	tMat = sp.zeros(len(states), len(states))

	# Compute transfer matrix
	for i, x in enumerate(states):
		print(i)
		for j, y in enumerate(states):
			tMat[i,j] = exp(eFunc(y + y, params) - eFunc(x + y + y, params))

			# Test block size (not guaranteed to catch errors in pathological cases)
			if check:
				for z in states:
					if exp(eFunc(y+z,params) - eFunc(x+y+z,params)) != tMat[i,j]:
						raise ValueError('Error: Block size too small.')

	# Initialize endcap states
	endStates = list([list(it.product(stateRange, repeat=l)) for l in range(1, blockSize + 1)])

	# Compute left end cap
	leftEnds = []
	for l in range(blockSize):
		print(l)
		end = sp.zeros(len(endStates[l]),len(states))
		for i,x in enumerate(endStates[l]):
			for j,y in enumerate(states):
				end[i,j] = exp(eFunc(y+y,params) - eFunc(x+y+y,params))
		leftEnds.append(end)

	# Compute right end cap
	rightEnds = []
	for l in range(blockSize):
		print(l)
		end = sp.zeros(len(states),len(endStates[l]))
		for i,x in enumerate(states):
			for j,y in enumerate(endStates[l]):
				# No subtraction here, as we want to set the energies in the case
				# of a single block with a right endcap (what we called anchoring the
				# energy in the docs above).
				end[i,j] = exp(-eFunc(x+y,params))
		rightEnds.append(end)

	return tMat,leftEnds,rightEnds

def transferMatrixVariableSize(eFunc, stateRange, params):
	'''
	This method takes as input:
		eFunc 		-	The Hamiltonian. This function must take as input a list of
						items drawn from stateRange and a list of parameters given
						by params.
		stateRange	-	This is a list specifying the allowed states on a site.
		params 		-	This is a list of the parameters in the Hamiltonian.
						These are sympy variables.


	This method attempts to find the correct block size for the transfer matrix.
	It does this by starting with a small block and increasing the size by one site
	until the check procedure does not fail. This is not guaranteed to work in pathological
	cases, but in cases with simple interacting Hamiltonians should be fine.

	'''

	blockSize = 1

	while True:

		try:
			tm = transferMatrix(eFunc, stateRange, params, blockSize, check=True)
			print('Final Size:',blockSize)
			return tm[0],tm[1],tm[2],blockSize
		except ValueError:
			print('Tried size',blockSize,'. Incrementing.')
			blockSize += 1


def wrapper(tm,leftEnds,rightEnds,params):
	'''
	This method takes as input a transfer matrix as well as the associated
	endcaps and parameters and wraps them so that they may be called as regular
	numeric functions rather than returning symbolic expressions.
	'''

	print('Wrapping transfer matrix...')
	T = sp.lambdify(params, tm, "mpmath")
	print('Wrapping transfer matrix derivative...')
	dP = sp.lambdify(params, tm.diff(params[0]), "mpmath")
	dQ = sp.lambdify(params, tm.diff(params[1]), "mpmath")
	dW = sp.lambdify(params, tm.diff(params[2]), "mpmath")
	print('Wrapping left end caps...')
	eL = list([sp.lambdify(params, l, "mpmath") for l in leftEnds])
	dQL = list([sp.lambdify(params, l.diff(params[1]), "mpmath") for l in leftEnds])
	dWL = list([sp.lambdify(params, l.diff(params[2]), "mpmath") for l in leftEnds])
	print('Wrapping right end caps...')
	eR = list([sp.lambdify(params, r, "mpmath") for r in rightEnds])
	dQR = list([sp.lambdify(params, r.diff(params[1]), "mpmath") for r in rightEnds])
	dWR = list([sp.lambdify(params, r.diff(params[2]), "mpmath") for r in rightEnds])
	print('Done!')
	return T,eL,eR,dP,dQ,dW,dQL,dWL,dQR,dWR

def fN(T,eL,eR,dP,dQ,dW,dQL,dWL,dQR,dWR,params,blockSize,n):
	'''
	This method takes as input:
		T 			-	The wrapped (numerical) transfer matrix.
		eL 			-	The wrapped (numerical) left endcap list.
		eR 			-	The wrapped (numerical) right endcap list.
		params 		-	The numerical values of the parameters.
		blockSize 	-	The block size used to construct T.
		n 			-	The size of the system of interest.

	This method computes the slope and intercept of the free energy form

	F(L) = a + b L

	b is computed using the maximal eigenvalue of T. The free energy is then computed
	for the system size n and this is used to determine a.

	'''

	# Partition the system size into left cap, blocks, right cap
	n1,n2,n3 = partition(n,blockSize)

	# Evaluate endcaps at partition sizes
	e1 = eL[n1-1](*params)
	e2 = eR[n3-1](*params)
	de1_dQ = dQL[n1-1](*params)
	de2_dQ = dQR[n3-1](*params)
	de1_dW = dWL[n1-1](*params)
	de2_dW = dWR[n3-1](*params)

	# Evaluate transfer matrix and derivative
	tm = T(*params)
	dp = dP(*params)
	dt_dQ = dQ(*params)
	dt_dW = dW(*params)



	# Normalize
	sh = np.array(tm.tolist(), dtype=float).shape
	y = -1e100
	for i in range(sh[0]):
		for j in range(sh[1]):
			if abs(tm[i,j]) > y:
				y = abs(tm[i,j])

	tm /= y
	dp /= y
	dt_dQ /= y
	dt_dW /= y


	# Cast to numpy
	e1 = np.array(e1.tolist(), dtype=float)
	e2 = np.array(e2.tolist(), dtype=float)
	tm = np.array(tm.tolist(), dtype=float)
	dp = np.array(dp.tolist(), dtype=float)
	dt_dQ = np.array(dt_dQ.tolist(), dtype=float)
	dt_dW = np.array(dt_dW.tolist(), dtype=float)
	de1_dQ = np.array(de1_dQ.tolist(), dtype=float)
	de2_dQ = np.array(de2_dQ.tolist(), dtype=float)
	de1_dW = np.array(de1_dW.tolist(), dtype=float)
	de2_dW = np.array(de2_dW.tolist(), dtype=float)

	# Compute slope and slope derivative
	vals, invvecs, vecs = eig(tm, left=True, right=True)

	ind = np.argmax(np.real(vals))
	val = np.real(vals[ind])
	logy = float(mp.log(y))
	logm = logy + log(val)
	slope = -logm / blockSize

	# ds = -d log m / blockSize
	# d log m = dm / m = d (m/y) / (m/y)
	# What we've written below as dm is actually d(m/y), so we divide it by val (which is just m/y).
	dm = np.dot(np.conj(invvecs[:,ind]), np.dot(dp, vecs[:,ind])) / np.dot(np.conj(invvecs[:,ind]), vecs[:,ind])
	ds = -np.real(dm) / (val * blockSize)

	d_lnZ_dQ_intensive = np.dot(np.conj(invvecs[:,ind]), np.dot(dt_dQ, vecs[:,ind])) / np.dot(np.conj(invvecs[:,ind]), vecs[:,ind])
	d_lnZ_dQ_intensive = -np.real(d_lnZ_dQ_intensive) / (val * blockSize)

	d_lnZ_dW_intensive = np.dot(np.conj(invvecs[:,ind]), np.dot(dt_dW, vecs[:,ind])) / np.dot(np.conj(invvecs[:,ind]), vecs[:,ind])
	d_lnZ_dW_intensive = -np.real(d_lnZ_dW_intensive) / (val * blockSize)

	# Compute free energy
	z = np.sum(np.dot(e1, np.dot(np.linalg.matrix_power(tm, n2 - 1), e2)))
	f = -real(log(z)) - (n2 - 1)*logy # logy provides the correction because we normalized tm

	# Compute dlnZ/dQ
	dlnZ = 0
	dlnZ += np.sum(np.dot(de1_dQ, np.dot(np.linalg.matrix_power(tm, n2 - 1), e2)))
	dlnZ += np.sum(np.dot(e1, np.dot(np.linalg.matrix_power(tm, n2 - 1), de2_dQ)))
	dlnZ += (n2 - 1) * np.sum(np.dot(e1, np.dot(dt_dQ, np.dot(np.linalg.matrix_power(tm, n2 - 2), e2))))
	dlnZ /= z
	num_Q = -dlnZ

	# Compute dlnZ/dW
	dlnZ = 0
	dlnZ += np.sum(np.dot(de1_dW, np.dot(np.linalg.matrix_power(tm, n2 - 1), e2)))
	dlnZ += np.sum(np.dot(e1, np.dot(np.linalg.matrix_power(tm, n2 - 1), de2_dW)))
	dlnZ += (n2 - 1) * np.sum(np.dot(e1, np.dot(dt_dW, np.dot(np.linalg.matrix_power(tm, n2 - 2), e2))))
	dlnZ /= z
	num_W = -dlnZ

	# Evaluate intercepts
	inter = f - slope*n
	interQ = num_Q - d_lnZ_dQ_intensive * n
	interW = num_W - d_lnZ_dW_intensive * n

	return slope,inter,f,ds, interQ, d_lnZ_dQ_intensive, interW, d_lnZ_dW_intensive
