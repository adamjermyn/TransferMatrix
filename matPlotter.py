import numpy as np
import sys

postfix = sys.argv[1]

left = list(range(-4,1))
right = list(range(0,5))
left = left[::-1]

chis = np.zeros((len(left), len(right)))
js = np.zeros((len(left), len(right)))
qs = np.zeros((len(left), len(right)))
ws = np.zeros((len(left), len(right)))

outs = np.zeros((len(left), len(right),4,3))

prefix = sys.argv[2]

for i,l in enumerate(left):
	for j,r in enumerate(right):
		if r - l < 5:
			try:
				params = np.loadtxt('Output/' + prefix + '_' + str(l) + '_' + str(r) + '_' + postfix +  '_summary.txt')
				outs[i,j] = params
			except:
				print('Error.',l,r,'not found!')

# Correct decoupled Q models

if postfix == 'inclusive':
	outs[0,0,3,2] *= 1. * (9 - 3 - 1) / (9 - 1 - 1)
	outs[1,0,3,2] *= 1. * (9 - 3 - 1) / (9 - 2 - 1)
	outs[1,1,3,2] *= 1. * (9 - 3 - 1) / (9 - 2 - 1)
	outs[0,1,3,2] *= 1. * (9 - 3 - 1) / (9 - 2 - 1)
elif postfix == 'exclusive_left':
	outs[0,0,3,2] *= 1. * (9 - 3 - 1) / (9 - 1 - 1)
	outs[0,2,3,2] *= 1. * (9 - 3 - 1) / (9 - 1 - 1)
	outs[0,1,3,2] *= 1. * (9 - 3 - 1) / (9 - 2 - 1)
	outs[1:3,0:2,3,2] *= 1. * (9 - 3 - 1) / (9 - 2 - 1)
elif postfix == 'exclusive_both':
	outs[:3,:3,3,2] *= 1. * (9 - 3 - 1) / (9 - 2 - 1)
	outs[0,0,3,2] *= 1. * (9 - 2 - 1) / (9 - 1 - 1)
else:
	raise ValueError


# Account for convention on partition function
outs[:,:,:3,:3] *= -1

outs[outs==0] = np.nan

js = outs[:,:,2,0]
qs = outs[:,:,0,0]
ws = outs[:,:,1,0]
chis = outs[:,:,3,2]

print(left[1:3], right[0:2])

best = np.argmin(chis)

best = (-1,-1,1e100)
for i in range(len(chis)):
	for j in range(len(chis[0])):
		if chis[i,j] < best[2]:
			best = (j,i,chis[i,j])
			print(i,j,chis[i,j],best[2])

print(chis)

print(best)

import matplotlib.pyplot as plt

plt.figure(figsize=(5.9,5))
plt.subplot(221)
plt.imshow(chis, origin='upper', vmax=5, cmap='cool')
plt.annotate('X', xy=(best[0]-0.2,best[1]+0.15), xycoords='data', fontsize=14)
plt.xlabel('$\\mathcal{R}$')
plt.ylabel('$\\mathcal{L}$', rotation='horizontal')
plt.title('Reduced $\\chi^2$')
plt.xticks(range(len(right)), right)
plt.yticks(range(len(left)), left)
plt.colorbar()
plt.subplot(222)
plt.imshow(js, origin='upper', cmap='coolwarm', vmin=-11, vmax=11)
plt.annotate('X', xy=(best[0]-0.2,best[1]+0.15), xycoords='data', fontsize=14)
plt.xlabel('$\\mathcal{R}$')
plt.ylabel('$\\mathcal{L}$', rotation='horizontal')
plt.xticks(range(len(right)), right)
plt.yticks(range(len(left)), left)
plt.title('J')
plt.colorbar()
plt.subplot(223)
plt.imshow(qs, origin='upper', cmap='coolwarm', vmin=-1.5, vmax=1.5)
plt.annotate('X', xy=(best[0]-0.2,best[1]+0.15), xycoords='data', fontsize=14)
plt.xlabel('$\\mathcal{R}$')
plt.ylabel('$\\mathcal{L}$', rotation='horizontal')
plt.xticks(range(len(right)), right)
plt.yticks(range(len(left)), left)
plt.title('Q')
plt.colorbar()
plt.subplot(224)
plt.imshow(ws, origin='upper', cmap='coolwarm', vmin=-1.5, vmax=1.5)
plt.annotate('X', xy=(best[0]-0.2,best[1]+0.15), xycoords='data', fontsize=14)
plt.xlabel('$\\mathcal{R}$')
plt.ylabel('$\\mathcal{L}$', rotation='horizontal')
plt.title('W')
plt.xticks(range(len(right)), right)
plt.yticks(range(len(left)), left)
plt.colorbar()
plt.tight_layout()
plt.savefig(prefix + '_matPlot_' + postfix + '.pdf')
