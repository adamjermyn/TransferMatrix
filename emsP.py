from data import data
from scipy.optimize import brentq
from numpy import exp,log,real
import numpy as np
import itertools as it
import transferMatrixClass as tmc
import models
import emcee as em
import sympy as sp
import sys

# The first input is a string specifying the kind of Actin, either 'h' or 'd'.
s = sys.argv[1] # must be 'd' or 'h'

# The next two inputs specify the model range
left = int(sys.argv[2])
right = int(sys.argv[3])

# For file names
fname = s + '_' + sys.argv[2] + '_' + sys.argv[3]
modelSTR = 'Output/' + fname

# Threads
threads = int(sys.argv[4])

# Build the model
model, blockSize = models.actinModelRuleFactory(left, right)
act = models.actin(model, [0,1], sp.symbols('a b c'), blockSize=blockSize)

def evaluate(theta, x, y, c):
	bindingF = x
	q,w,j= theta
	pVals = np.zeros(bindingF.shape)
	fVals = np.zeros(bindingF.shape)
	for i,bf in enumerate(bindingF):
		p,_,_ = act.bindingFinder((0,q,w),bf)
		pVals[i] = p
		fVals[i] = act.fN((p,q,w))[1]
	model = models.meanL(c,fVals,j)
	return model

def reducedChiSquared(theta, x, y, yerr, c):
	model = evaluate(theta,x,y,c)
	inv_sigma2 = 1.0/(yerr**2)
	return np.sum((y-model)**2*inv_sigma2)/(len(x)-3-1)

def lnlike(theta, x, y, yerr, c):
	model = evaluate(theta,x,y,c)
	print(model)
	inv_sigma2 = 1.0/(yerr**2)
	return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior(theta):
	q,w,j = theta
	if -6.0 < q < 6.0 and -6.0 < w < 6.0 and -17.0 < j < 0.0:
		return 0.0
	return -np.inf

def lnprob(theta, x, y, yerr,c):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	print('Likelihood:',lp + lnlike(theta, x, y, yerr,c))
	print('Params:',theta)
	return lp + lnlike(theta, x, y, yerr,c)

ndim, nwalkers, nper = 3, 32, 20

c,bindingF,length,dl,name = data(s)

print('Beginning '+name+' search...')

pos = [2*np.random.randn(ndim)+np.array([0,0,-7.5]) for i in range(nwalkers)]
sampler = em.EnsembleSampler(nwalkers, ndim, lnprob, args=(bindingF, length, dl, c), threads=threads)

for i in range(1000):
	print(i)
	pos, prob, state = sampler.run_mcmc(pos, nper)
	print('Sampled.')
	samples = sampler.chain[:, nper*i//2:, :].reshape((-1, ndim))
	sampless = np.copy(samples)
	q_mcmc, w_mcmc, j_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(sampless, [16, 50, 84],
                                                axis=0)))
	print('PARAMSIGMAS:',q_mcmc,w_mcmc,j_mcmc,reducedChiSquared((q_mcmc[0],w_mcmc[0],j_mcmc[0]),bindingF,length,dl,c))
	np.savetxt(modelSTR, samples)
	np.savetxt(modelSTR+'_summary.txt', np.array([q_mcmc,w_mcmc,j_mcmc,[0,0,reducedChiSquared((q_mcmc[0],w_mcmc[0],j_mcmc[0]),bindingF,length,dl,c)]]))
	bFhighRes = np.linspace(min(bindingF),max(bindingF),num=100,endpoint=True)
	fHR = np.zeros(100)

exit()
