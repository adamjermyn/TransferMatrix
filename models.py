import pickle
import numpy as np
import transferMatrixClass as tmc
from scipy.optimize import brentq
from numpy import log
# Helpers

def brentqWrapper(func):
	init = 10.
	while init < 1e5 and func(-init) >= func(init):
		init *= 2
	
	a = brentq(func,-init,init)
	return a

def meanL(c,inter,j):
	b = inter - j
	x = c*np.exp(b)
	ret = np.zeros(x.shape)
	ret = 0.5*(-1 + np.sqrt(1 + 4*x))
	return ret

# Models

def eFuncIsing(state, params): # Accepts binary inputs
	e = 0
	for i in range(len(state)):
		e += state[i]*params[0]
	for i in range(len(state)-1):
		e += state[i]*state[i+1]*params[1]
	return e

def eFuncOnePlaneActinModel(state, params): # Accepts binary inputs
	e = 0
	# Binding energy
	for i in range(len(state)):
		e += params[0]*state[i]
	# Bonds:
	x = list(state) + [0]
	for i in range(1,len(state)):
		if x[i] == 1:
			e += params[1]
		elif x[i-1] == 1 and x[i+1] == 0:
			e += params[2]
		elif x[i-1] == 0 and x[i+1] == 1:
			e += params[2]
		elif x[i-1] == 1 and x[i+1] == 1:
			e += params[1]
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

def eFuncLongTwoPlaneActinModel(state, params): # Accepts binary inputs
	e = 0
	# Binding energy
	for i in range(len(state)):
		e += params[0]*state[i]
	# In-plane bonds
	for i in range(len(state)):
		if sum(state[max(0,i-4):min(len(state),i+4):2]) > 0:
			e += params[1]
	# Out-of-plane bonds
	for i in range(len(state)-1):
		if sum(state[i:min(len(state),i+6)]) > 0:
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


# Actin class

class actin:
	def __init__(self,model,stateRange,params,blockSize=None):
		self.model = model
		self.stateRange = stateRange
		self.params = params

		# Initialize symbolic matrices
		if blockSize is None:
			self.tm,self.left,self.right,self.blockSize = tmc.transferMatrixVariableSize(self.model,stateRange,params)
		else:
			self.blockSize = blockSize
			self.tm,self.left,self.right = tmc.transferMatrix(self.model,stateRange,params,blockSize,check=False)
		import pickle
#		self.tm = pickle.load(open('actTM'))
#		self.left = pickle.load(open('actLeft'))
#		self.right = pickle.load(open('actRight'))
#		self.blockSize = pickle.load(open('block'))
		pickle.dump(self.tm,open('actTM','w+'))
		pickle.dump(self.left,open('actLeft','w+'))
		pickle.dump(self.right,open('actRight','w+'))
		pickle.dump(self.blockSize,open('block','w+'))
		# Wrapped matrices
		self.T,self.eL,self.eR = tmc.wrapper(self.tm,self.left,self.right,params)

	def fN(self,params,n=50):
		return tmc.fN(self.T,self.eL,self.eR,params,self.blockSize,n)

	def cofilinBindingFrac(self,params):
		# Assumes that params[0] is the binding energy
		p0 = np.zeros(params.shape)
		p1 = np.zeros(params.shape)
		p0 += params
		p1 += params
		p0[0] -= 1e-8
		p1[0] += 1e-8
		sM,_,_ = self.fN(p0)
		sP,_,_ = self.fN(p1)
		return ((sP-sM)/(2e-8))

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

