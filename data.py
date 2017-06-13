import numpy as np

molarConv = 600 # Converts 1M/L to occupied volume fraction, assuming cubic 10nm monomers

def data(name):
	if name=='h':
		# Model parameters for hCof_A167EyActin:
		c = 3e-6*molarConv
		bindingF = np.array([1e-3,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
		length = np.array([4.6, 3.19, 2.61, 1.61, 1.2, 1.69, 1.68, 2.49, 3.16, 3.46])
		dl = np.array([0.69, 0.29, 0.34, 0.13, 0.11, 0.13, 0.42, 0.13, 0.21, 0.34])
		name = "hCof_A167EyActin"
		return c,bindingF,length,dl,name
	elif name=='d':
		# Model parameters for D34C_yCof_RSKactin:
		c = 2e-6*molarConv
		bindingF = np.array([1e-3,0.1,0.2,0.4,0.5,0.6,0.75,0.85,0.98])
		length = np.array([5.56,3.85,2.94,2.73,2.68,3.24,4.03,4.16,4.09])
		dl = np.array([0.45,0.26,0.37,0.3,0.22,0.52,0.27,0.23,0.38])
		name = "D34C_yCof_RSKactin"
		return c,bindingF,length,dl,name
