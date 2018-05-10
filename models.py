import pickle
import numpy as np
import transferMatrixClass as tmc
from scipy.optimize import brentq
from numpy import log

# Helpers

def brentqWrapper(func):
	init = 3

	while init < 1e10 and np.sign(func(init)) == np.sign(func(-init)):
		init *= 2

	a = brentq(func,-init,init)
	return a

def meanL(c,inter,j):
	b = inter - j
	x = c*np.exp(b)
	ret = np.zeros(x.shape)
	ret = x * (-1 + np.sqrt(1 + 4*x))/(2 * x + 1 - np.sqrt(1 + 4*x))
	return ret

# Models

def eFuncIsing(state, params): # Accepts binary inputs
	e = 0
	for i in range(len(state)):
		e += state[i]*params[0]
	for i in range(len(state)-1):
		e += state[i]*state[i+1]*params[1]
	return e

def eFuncSymmetricTwoPlaneActinModel(state, params): # Accepts binary inputs
	e = 0
	# Binding energy
	for i in range(len(state)):
		e += params[0]*state[i]
	# In-plane bonds
	for i in range(len(state)):
		if sum(state[max(0,i-2):min(len(state),i+4):2]) > 0:
			e += params[1]
	# Out-of-plane bonds
	for i in range(len(state)-1):
		if sum(state[i:min(len(state),i+4)]) > 0:
			e += params[2]
	return e

def eFuncTwoPlaneActinModel(state, params): # Accepts binary inputs
	e = 0
	# Binding energy
	for i in range(len(state)):
		e += params[0]*state[i]
	# In-plane bonds
	for i in range(2,len(state)):
		if state[i] == 1 or state[i-2] == 1:
			e += params[1]
	# Out-of-plane bonds
	for i in range(len(state)-1):
		if sum(state[i:min(len(state),i+4)]) > 0:
			e += params[2]
	return e

def eFuncTwoPlaneShortActinModel(state, params): # Accepts binary inputs
	e = 0
	# Binding energy
	for i in range(len(state)):
		e += params[0]*state[i]
	# In-plane bonds
	for i in range(2,len(state)):
		if state[i] == 1 or state[i-2] == 1:
			e += params[1]
	# Out-of-plane bonds
	for i in range(len(state)-1):
		if sum(state[i:min(len(state),i+3)]) > 0:
			e += params[2]
	return e

def eFuncTwoPlaneVeryShortActinModel(state, params): # Accepts binary inputs
	e = 0
	# Binding energy
	for i in range(len(state)):
		e += params[0]*state[i]
	# In-plane bonds
	for i in range(2,len(state)):
		if state[i] == 1 or state[i-2] == 1:
			e += params[1]
	# Out-of-plane bonds
	for i in range(len(state)-1):
		if sum(state[i:min(len(state),i+2)]) > 0:
			e += params[2]
	return e

def actinModelFactory(inPlane, outPlane):
	'''
	Generates the energy function for an Actin binding model.
	
	inPlane is a list of integers specifying which in-plane bonds contribute
	to the energy. This is indexed so that zero means the bond the Cofilin is attached
	to.

	outPlane is a list of integers specifying which out-of-plane bonds contribute to the
	energy. This is indexed so that zero is not used, one and later mean the bonds
	to the right of that the Cofilin is attached to and negative one and earlier are
	those to the left.
	'''
	def energy(state, params):
		# State and params are binary inputs
		

def eFuncSymmetricTwoPlaneVeryShortActinModel(state, params): # Accepts binary inputs
	e = 0
	# Binding energy
	for i in range(len(state)):
		e += params[0]*state[i]
	# In-plane bonds
	for i in range(2,len(state)):
		if state[i] == 1 or state[i-2] == 1:
			e += params[1]
	# Out-of-plane bonds
	for i in range(len(state)-1):
		if sum(state[i:min(len(state),i+2)]) > 0:
			e += params[2]
	return e


# Actin class

class actin:
	def __init__(self,model,stateRange,params,blockSize=None):
		self.model = model
		self.stateRange = stateRange
		self.params = params

		# Initialize symbolic matrices
		print('Constructing matrices...')

		if blockSize is None:
			self.tm,self.left,self.right,self.blockSize = tmc.transferMatrixVariableSize(self.model,stateRange,params)
		else:
			self.blockSize = blockSize
			self.tm,self.left,self.right = tmc.transferMatrix(self.model,stateRange,params,blockSize,check=False)

		# Wrapped matrices
		print('Wrapping...')
		self.T,self.eL,self.eR,self.dT = tmc.wrapper(self.tm,self.left,self.right,params)
		print('Done.')

	def fN(self,params,n=50):
		return tmc.fN(self.T,self.eL,self.eR,self.dT,params,self.blockSize,n)

	def cofilinBindingFrac(self,params):
		# Assumes that params[0] is the binding energy
		return self.fN(params)[3]

	def bindingFinder(self,params,bf):
		# Assumes that params[0] is the binding energy
		def ff(bindingE):
			pcopy = np.copy(params)
			pcopy[0] = bindingE
			return self.cofilinBindingFrac(pcopy)-bf

		bindingE = brentqWrapper(ff)
		if ff(bindingE)>1e-8:
			print('Error:',bindingE,ff(bindingE),ff(bindingE-1e-3),ff(bindingE+1e-3))
		pcopy = np.copy(params)
		pcopy[0] = bindingE	
		return pcopy

