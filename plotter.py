import numpy as np
import models
import matplotlib
import matplotlib.pyplot as plt
from data import molarConv
import data
import sympy as sp

#matplotlib.style.use('http://plotornot.chrisbeaumont.org/matplotlibrc')

def plotterCran(minB,maxB,model,cRan,q,w,j):
	for c in cRan:
		bFhighRes = np.linspace(minB,maxB,num=100,endpoint=True)
		fHR = np.zeros(100)
		for i,x in enumerate(bFhighRes):
			p = model.bindingFinder((0,q,w),x)[0]
			fHR[i] = model.fN((p,q,w))[1]
		mlHighRes = models.meanL(c*molarConv,fHR,j)
		plt.plot(bFhighRes,mlHighRes/100,label=str(c)+'$\mu M$')
	plt.xlabel('Cofilin Binding Fraction')
	plt.ylabel('Filament length ($\mu m$)')
	plt.legend(loc=0)
	plt.show()

def plottermodelRan(minB,maxB,model,c,qR,wR,jR):
	for aa in range(len(qR)):
		q = qR[aa]
		j = jR[aa]
		w = wR[aa]
		bFhighRes = np.linspace(minB,maxB,num=100,endpoint=True)
		fHR = np.zeros(100)
		for i,x in enumerate(bFhighRes):
			p = model.bindingFinder((0,q,w),x)[0]
			fHR[i] = model.fN((p,q,w))[1]
		mlHighRes = models.meanL(c*molarConv,fHR,j)
		plt.plot(bFhighRes,mlHighRes/100,label='('+str(q)+','+str(w)+','+str(j)+')')
	plt.xlabel('Cofilin Binding Fraction')
	plt.ylabel('Filament length ($\mu m$)')
	plt.legend(loc=0)
	plt.show()

def plotterModels(minB,maxB,modelss,c,qR,wR,jR,modelStr,bindF,length,dl):
	for aa in range(len(qR)):
		q = qR[aa]
		j = jR[aa]
		w = wR[aa]
		bFhighRes = np.linspace(minB,maxB,num=100,endpoint=True)
		fHR = np.zeros(100)
		for i,x in enumerate(bFhighRes):
			p = modelss[aa].bindingFinder((0,q,w),x)[0]
			fHR[i] = modelss[aa].fN((p,q,w))[1]
		mlHighRes = models.meanL(c*molarConv,fHR,j)
		plt.plot(bFhighRes,mlHighRes/100,label=modelStr[aa]+' ('+str(q)+','+str(w)+','+str(j)+')')
	plt.scatter(bindingF,length/100)
	plt.errorbar(bindingF,length/100,yerr=dl/100,ls='dotted')
	plt.xlabel('Cofilin Binding Fraction')
	plt.ylabel('Filament length ($\mu m$)')
	plt.legend(loc=0)
	plt.show()	

# D Mid:
# (-0.30156024324897013, 0.13636134861603974, 0.069553728797884018) (0.97202457554124932, 1.3981449108974584, 0.12401812887914399) (-15.238251635899056, 0.095724814105128786, 0.7984433598849332) 1.26070004381
# H Mid:
# (-0.71303185019882953, 0.10034016135370605, 0.087605271510944549) (1.6044046429350651, 0.19528508206599127, 0.12997412197620806) (-14.594844213604459, 0.16568211565285473, 0.22456225476814851) 7.05428848056
# D Short:
# (-0.81848475057743431, 0.19322606944533649, 0.15272810007342807) (1.9343216366420968, 0.48891481849954466, 0.2697622007677245) (-15.201160260614268, 0.10152183223045164, 3.1717203932552938) 2.35932251346
# H Short:
# (-2.0699434208117005, 0.30206785380482581, 0.44125973515527228) (3.804652744160832, 0.72799600070329662, 0.5412495730753708) (-14.578079079601105, 0.25425175222379259, 0.24168369612567275) 9.9719247644
# D Very short:
# (-2.7778276553682879, 1.3933165876805416, 1.3652241985016333) (5.9759339096931043, 2.7459065608908411, 2.6822565439039168) (-14.962564878661105, 0.1082529317901102, 0.097084792644004381) 10.9310736979
# H Very short:
# (0.4404530721227507, 0.31321383099634831, 0.46065851204450009) (-1.7137759363737357, 0.73415646709898585, 0.39436798351898283) (-13.018707314496369, 0.20382472858603506, 0.19962160272523377) 22.771184622

# h fit parameters
jH = [-14.595,-14.578,-13.019]
qH = [-0.713,-2.070,0.440]
wH = [1.604,3.805,-1.714]

# d fit parameters
jD = [-15.238,-15.201,-14.963]
qD = [-0.302,-0.818,-2.778]
wD = [0.972,1.934,5.976]

# Binding fraction range
minB = 1e-3
maxB = 0.97

# Actin models
act1 = models.actin(models.eFuncTwoPlaneActinModel,[0,1],sp.symbols('a b c'))
act2 = models.actin(models.eFuncTwoPlaneShortActinModel,[0,1],sp.symbols('a b c'))
act3 = models.actin(models.eFuncTwoPlaneVeryShortActinModel,[0,1],sp.symbols('a b c'))
modelss = [act1,act2,act3]

c,bindingF,length,dl,name = data.data('h')
plotterModels(minB,maxB,modelss,3e-6,qH,wH,jH,['Long','Medium','Short'],bindingF,length,dl)
c,bindingF,length,dl,name = data.data('d')
plotterModels(minB,maxB,modelss,2e-6,qD,wD,jD,['Long','Medium','Short'],bindingF,length,dl)

# Concentration range
#cRan = [1e-6,2e-6,3e-6,5e-6,1e-5,3e-5 ]

#plotterCran(minB,maxB,act,cRan,qD,wD,jD)

#df = 0.5

#plottermodelRan(minB,maxB,act,3e-6,[qD,qD-df,qD+df],[wD,wD,wD],[jD,jD,jD])
#plottermodelRan(minB,maxB,act,3e-6,[qD,qD,qD],[wD,wD-df,wD+df],[jD,jD,jD])
#plottermodelRan(minB,maxB,act,3e-6,[qD,qD,qD],[wD,wD,wD],[jD,jD-df,jD+df])
