import numpy as np
import models
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from corner import hist2d
import data
from data import molarConv
import sympy as sp

# Long PRL

model, blockSize = models.exclusiveActinModelRuleFactory(0, 4, sides='left')
act = models.actin(model, [0,1], sp.symbols('a b c'), blockSize=blockSize)

# Read samples

samples = np.loadtxt('Output/d_0_4_exclusive_left')

# Cut out J

samples = samples[:,(0,1)]

# Make convention match PRL.

samples *= -1

# Make corner plot

fig = plt.figure(figsize=(5,7))

gs = GridSpec(3,3,figure=fig,height_ratios=[0.1,1,1], width_ratios=[1,1,0.15])

ax = [[],[],[],[]]
ax[0].append(plt.subplot(gs[0,0]))
ax[0].append(plt.subplot(gs[0,1]))
ax[0].append(plt.subplot(gs[0,2]))
ax[1].append(plt.subplot(gs[1,0]))
ax[1].append(plt.subplot(gs[1,1]))
ax[1].append(plt.subplot(gs[1,2]))
ax[2].append(plt.subplot(gs[2,:]))

qRan = (-2,2)
wRan = (-2,2)

hist2d(samples[:,0],samples[:,1],ax=ax[1][0],bins=30,plot_datapoints=False)
ax[1][0].set_xlabel('$Q$')
ax[1][0].set_ylabel('$W$')
ax[1][0].set_xlim((0,2))
ax[1][0].set_ylim((-2,-0.5))

hist2d(samples[:,0],samples[:,1],ax=ax[1][1],bins=100, plot_datapoints=False,range=[qRan, wRan])
ax[1][1].set_xlabel('$Q$')
ax[1][1].set_ylabel('$W$')

lims = [ax[1][0].get_xlim(), ax[1][0].get_ylim()]
rect = Rectangle((lims[0][0], lims[1][0]), lims[0][1]-lims[0][0], lims[1][1]-lims[1][0], linestyle='dashed',fill=False)
ax[1][1].add_patch(rect)
ax[1][1].scatter((1,),(1,))
ax[1][1].scatter((1,),(-1,))
ax[1][1].scatter((-1,),(1,))
ax[1][1].scatter((-1,),(-1,))
ax[1][1].scatter((0,),(0,))


ax[0][1].hist(samples[:,0], bins=100, density=True, range=qRan, color='k',histtype='stepfilled')
ax[0][1].get_yaxis().set_visible(False)
ax[0][1].get_xaxis().set_visible(False)
ax[0][1].spines['top'].set_visible(False)
ax[0][1].spines['bottom'].set_visible(False)
ax[0][1].spines['left'].set_visible(False)
ax[0][1].spines['right'].set_visible(False)

ax[1][2].hist(samples[:,1], bins=100, density=True, range=wRan, color='k',histtype='stepfilled', orientation='horizontal')
ax[1][2].get_xaxis().set_visible(False)
ax[1][2].get_yaxis().set_visible(False)
ax[1][2].spines['top'].set_visible(False)
ax[1][2].spines['left'].set_visible(False)
ax[1][2].spines['bottom'].set_visible(False)
ax[1][2].spines['right'].set_visible(False)

ax[0][0].get_yaxis().set_visible(False)
ax[0][0].get_xaxis().set_visible(False)
ax[0][0].spines['top'].set_visible(False)
ax[0][0].spines['bottom'].set_visible(False)
ax[0][0].spines['left'].set_visible(False)
ax[0][0].spines['right'].set_visible(False)

ax[0][2].get_yaxis().set_visible(False)
ax[0][2].get_xaxis().set_visible(False)
ax[0][2].spines['top'].set_visible(False)
ax[0][2].spines['bottom'].set_visible(False)
ax[0][2].spines['left'].set_visible(False)
ax[0][2].spines['right'].set_visible(False)

ax[2][0].set_xlabel('Cofilin Binding Fraction')
ax[2][0].set_ylabel('Filament length ($\mu m$)')


# Plot models

c,bindingF,length,dl,name = data.data('d')

# Binding fraction range
minB = 1e-3
maxB = 0.97

def plotterModels(ax, minB,maxB,model,c,qR,wR,jR,modelStr,bindF):
	for aa in range(len(qR)):
		q = qR[aa]
		j = jR[aa]
		w = wR[aa]
		q *= -1
		w *= -1
		bFhighRes = np.linspace(minB,maxB,num=100,endpoint=True)
		fHR = np.zeros(100)
		for i,x in enumerate(bFhighRes):
			p = model.bindingFinder((0,q,w),x)[0]
			fHR[i] = model.fN((p,q,w))[1]
		mlHighRes = models.meanL(c*molarConv,fHR,j)
		ax.plot(bFhighRes,mlHighRes,label=modelStr[aa])

q = [1.,1.,-1.,-1.,0.]
w = [1.,-1.,1.,-1.,0.]
j = [-5, -9, -8, -10, -8]
#j = [-11.,-10.,-9.,-7.,-10.]
plotterModels(ax[2][0], minB, maxB, act, 2e-6, q, w, j, ['$Q=1,W=1$','$Q=1,W=-1$','$Q=-1,W=1$','$Q=-1,W=-1$','$Q=0,W=0$'], bindingF)

# Finalise layout

plt.tight_layout()
#fig.subplots_adjust(wspace=0.4,hspace=0.3)
from matplotlib.transforms import Affine2D

scaler = Affine2D().scale(sx=1.25, sy=1.0)
ax[1][0].set_position(ax[1][0].get_position().transformed(scaler))
ax[1][2].set_position(ax[1][2].get_position().transformed(scaler))

ax[1][1].set_position(ax[1][1].get_position().translated(0.1,0))
ax[0][1].set_position(ax[0][1].get_position().translated(0.1,-0.07))
ax[1][2].set_position(ax[1][2].get_position().translated(-0.27,-0.007))


#fig = corner(samples, bins=30, labels=["$Q$", "$W$"], range=[(-3,3),(-3,3)])

plt.savefig('test.pdf')
