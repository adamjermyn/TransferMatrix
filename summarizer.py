import numpy as np

left = list(range(-3,1))
right = list(range(1,5))

chis = np.zeros((len(left), len(right)))
js = np.zeros((len(left), len(right)))
qs = np.zeros((len(left), len(right)))
ws = np.zeros((len(left), len(right)))

outs = np.zeros((len(left), len(right),4,3))

for i,l in enumerate(left):
	for j,r in enumerate(right):
		try:
			params = np.loadtxt('Output/d_' + str(l) + '_' + str(r) + '_summary.txt')
			outs[i,j] = params
		except:
			print('Error.',i,j,'not found!')

# Correct decoupled Q models

outs[1,0,3,2] *= 1. * (9 - 3 - 1) / (9 - 2 - 1)
outs[2,0,3,2] *= 1. * (9 - 3 - 1) / (9 - 2 - 1)
outs[3,0,3,2] *= 1. * (9 - 3 - 1) / (9 - 2 - 1)
outs[3,1,3,2] *= 1. * (9 - 3 - 1) / (9 - 2 - 1)

# Account for convention on partition function
outs[:,:,:3,:3] *= -1

# Print chis

print('&&$\\frac{\\chi^2}{N-1}$&&\\\\')
print('L,R &',end='')
for r in right[:-1]:
	print(r,'&',end='')
print(right[-1])
print('\\\\')
print('\\hline')
for i in range(len(left)):
	print(left[i],'&',end='')
	for j in range(len(right)-1):
		print(round(outs[i,j,3,2],2),'&',end='')
	print(round(outs[i,-1,3,2],2),end='')
	print('\\\\')
print('\\hline')

# Print js

print('&&$J$&&\\\\')
print('L,R &',end='')
for r in right[:-1]:
	print(r,'&',end='')
print(right[-1])
print('\\\\')
print('\\hline')
for i in range(len(left)):
	print(left[i],'&',end='')
	for j in range(len(right)-1):
		print('$',round(outs[i,j,2,0],2),'^{',round(outs[i,j,2,1],2),'}_{',round(outs[i,j,2,2],2),'}$&',end='')
	print('$',round(outs[i,j,2,0],2),'^{',round(outs[i,j,2,1],2),'}_{',round(outs[i,j,2,2],2),'}$',end='')
	print('\\\\')
print('\\hline')

# Print qs

print('&&$Q$&&\\\\')
print('L,R &',end='')
for r in right[:-1]:
	print(r,'&',end='')
print(right[-1])
print('\\\\')
print('\\hline')
for i in range(len(left)):
	print(left[i],'&',end='')
	for j in range(len(right)-1):
		print('$',round(outs[i,j,0,0],2),'^{',round(outs[i,j,0,1],2),'}_{',round(outs[i,j,0,2],2),'}$&',end='')
	print('$',round(outs[i,j,0,0],2),'^{',round(outs[i,j,0,1],2),'}_{',round(outs[i,j,0,2],2),'}$',end='')
	print('\\\\')
print('\\hline')

# Print ws

print('&&$W$&&\\\\')
print('L,R &',end='')
for r in right[:-1]:
	print(r,'&',end='')
print(right[-1])
print('\\\\')
print('\\hline')
for i in range(len(left)):
	print(left[i],'&',end='')
	for j in range(len(right)-1):
		print('$',round(outs[i,j,1,0],2),'^{',round(outs[i,j,1,1],2),'}_{',round(outs[i,j,1,2],2),'}$&',end='')
	print('$',round(outs[i,j,1,0],2),'^{',round(outs[i,j,1,1],2),'}_{',round(outs[i,j,1,2],2),'}$',end='')
	print('\\\\')
print('\\hline')




#			chis[i+1,j+1] = params[-1,-1]
#			qs[i+1,j+1] = params[0,0]
#			ws[i+1,j+1] = params[1,0]
#			js[i+1,j+1] = params[2,0]
exit()



from tabulate import tabulate


chis[0,1:] = right
chis[1:,0] = left
js[0,1:] = right
js[1:,0] = left
qs[0,1:] = right
qs[1:,0] = left
ws[0,1:] = right
ws[1:,0] = left

print(tabulate(chis, tablefmt='latex', floatfmt='.2f').split('\n',2)[2].rsplit('\n',1)[0])
print(tabulate(js, tablefmt='latex', floatfmt='.2f').split('\n',2)[2].rsplit('\n',1)[0])
print(tabulate(qs, tablefmt='latex', floatfmt='.2f').split('\n',2)[2].rsplit('\n',1)[0])
print(tabulate(ws, tablefmt='latex', floatfmt='.2f').split('\n',2)[2].rsplit('\n',1)[0])
