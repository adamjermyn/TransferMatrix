from data import data
from scipy.optimize import brentq
from numpy import exp,log,real
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import transferMatrixClass as tmc
import models
import emcee as em
from corner import corner
import sympy as sp
import sys

# Can specify s = 'h' or 'd' for the two kinds of actin
# The models we use are:
# short - eFuncTwoPlaneVeryShortActinModel
# medium - eFuncTwoPlaneShortActinModel
# long - eFuncTwoPlaneActinModel

if sys.argv[1] == 'short':
	model = models.eFuncTwoPlaneVeryShortActinModel
elif sys.argv[1] == 'medium':
	model = models.eFuncTwoPlaneShortActinModel
elif sys.argv[1] == 'long':
	model = models.eFuncTwoPlaneActinModel
elif sys.argv[1] == 'sym':
	model = models.eFuncSymmetricTwoPlaneActinModel

s = sys.argv[2] # must be 'd' or 'h'

modelSTR = sys.argv[1] + '_' + s

_, _, _, blockSize = tmc.transferMatrixVariableSize(model, [0,1], sp.symbols('a b c'))
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

ndim, nwalkers, nper = 3, 8, 40

c,bindingF,length,dl,name = data(s)

print('Beginning '+name+' search...')

pos = [2*np.random.randn(ndim)+np.array([0,0,-7.5]) for i in range(nwalkers)]
sampler = em.EnsembleSampler(nwalkers, ndim, lnprob, args=(bindingF, length, dl, c), threads=1)

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
	np.savetxt(modelSTR+'_summary', np.array([q_mcmc,w_mcmc,j_mcmc,[0,0,reducedChiSquared((q_mcmc[0],w_mcmc[0],j_mcmc[0]),bindingF,length,dl,c)]]))
	fig = corner(samples, labels=["$Q$", "$W$","$J$", "$\ln\,f$"])
	fig.savefig(modelSTR+'_'+'triangle'+s+'.png')
	plt.close('all')
	plt.scatter(bindingF,length)
	plt.errorbar(bindingF,length,yerr=dl)
	bFhighRes = np.linspace(min(bindingF),max(bindingF),num=100,endpoint=True)
	fHR = np.zeros(100)
	for i,x in enumerate(bFhighRes):
		p = act.bindingFinder((0,q_mcmc[0],w_mcmc[0]),x)[0]
		fHR[i] = act.fN((p,q_mcmc[0],w_mcmc[0]))[1]
		if i%10==0:
			print(p)
	mlHighRes = models.meanL(c,fHR,j_mcmc[0])
	plt.plot(bFhighRes,mlHighRes)
	plt.xlabel('Cofilin Binding Fraction')
	plt.ylabel('Filament length ($\mu m$)')
	plt.title(name)
	plt.savefig(modelSTR+'_'+'fig'+s+'.png',dpi=100)

exit()
