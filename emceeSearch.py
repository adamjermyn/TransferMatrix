from data import data
from scipy.optimize import brentq
from scipy.special import hyp2f1
from numpy import exp,log,real
import numpy as np
import pickle
import itertools as it
import matplotlib.pyplot as plt
import transferMatrixClass as tmc
import models
import emcee as em
from corner import corner
import sympy as sp
import dill as pickle

# Can specify s = 'h' or 'd' for the two kinds of actin
# The models we use are:
# short - eFuncTwoPlaneVeryShortActinModel
# medium - eFuncTwoPlaneShortActinModel
# long - eFuncTwoPlaneActinModel

act = pickle.load(open('model.dat','rb'))

s = 'd'

def evaluate(theta, x, y, c):
	bindingF = x
	q,w,j= theta
	pVals = np.zeros(bindingF.shape)
	fVals = np.zeros(bindingF.shape)
	for i,bf in enumerate(bindingF):
		p,_,_ = act.bindingFinder((0,q,w),bf)
		print(p)
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
	if -3.0 < q < 3.0 and -3.0 < w < 3.0 and -3.0 < j < 3.0:
		return 0.0
	return -np.inf

def lnprob(theta, x, y, yerr,c):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, x, y, yerr,c)

ndim, nwalkers, nper = 3, 100, 2

c,bindingF,length,dl,name = data(s)

print('Beginning '+name+' search...')

pos = [2*np.random.randn(ndim)+np.array([0,0,0]) for i in range(nwalkers)]
sampler = em.EnsembleSampler(nwalkers, ndim, lnprob, args=(bindingF, length, dl,c))

for i in range(1000):
	print(i)
	pos, prob, state = sampler.run_mcmc(pos, nper)
	print('Sampled.')
	samples = sampler.chain[:, nper*i/2:, :].reshape((-1, ndim))
	sampless = np.copy(samples)
	q_mcmc, w_mcmc, j_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(sampless, [16, 50, 84],
                                                axis=0)))
	print(q_mcmc,w_mcmc,j_mcmc,reducedChiSquared((q_mcmc[0],w_mcmc[0],j_mcmc[0]),bindingF,length,dl,c))

	fig = corner(samples, labels=["$Q$", "$W$","$J$", "$\ln\,f$"])
	fig.savefig('triangle'+s+'.png')
	plt.close('all')
	plt.scatter(bindingF,length/100)
	plt.errorbar(bindingF,length/100,yerr=dl/100)
	bFhighRes = np.linspace(min(bindingF),max(bindingF),num=100,endpoint=True)
	fHR = np.zeros(100)
	for i,x in enumerate(bFhighRes):
		p = act.bindingFinder((0,q_mcmc[0],w_mcmc[0]),x)[0]
		fHR[i] = act.fN((p,q_mcmc[0],w_mcmc[0]))[1]
		if i%10==0:
			print(p)
	mlHighRes = models.meanL(c,fHR,j_mcmc[0])
	plt.plot(bFhighRes,mlHighRes/100)
	plt.xlabel('Cofilin Binding Fraction')
	plt.ylabel('Filament length ($\mu m$)')
	plt.title(name)
	plt.savefig('fig'+s+'.png',dpi=100)

exit()
