import numpy as np
from os import listdir
from os.path import isfile, join

mypath = 'Output'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
files = [f for f in onlyfiles if 'summary' in f]

left = list(range(-4,1))
right = list(range(0,5))
left = left[::-1]

fi = open('out.csv','w')

# Note that the +/- errors are swapped in order because we negate the parameters.
fi.write('Actin,Cofilin,L,R,Both Inclusive?,Q,dQ-,dQ+,W,dW-,dW+,J,dJ-,dJ+,Reduced Chi^2,\n')

for f in files:
	if '_h' not in f and 'summary' in f and 'right' not in f and 'both' not in f and 'prl' not in f and 'sym' not in f and 'short' not in f and 'long' not in f and 'med' not in f:
		if 'exclusive' in f or 'inclusive' in f:
			fname = 'Output/' + f
			data = np.loadtxt(fname)

			description = f.split('_')


			s = ''

			if  f[:2] == 'hy':
				s = s + 'A167EyActin'
			elif f[0] == 'd' or f[0] == 'b':
				s = s + 'RSKactin'

			if f[:2] == 'hy' or f[0] == 'b':
				s = s + ',' + 'hCof'
			else:
				s = s + ',' + 'D34C_yCof'

			s = s + ',' + description[1]
			s = s + ',' + description[2]


			L = int(description[1])
			R = int(description[2])
			chi = data[-1][-1]

			if 'inclusive' in fname:
				s = s + ',' + 'True'
				if L == left[0] and R == right[0]:
					chi *= (9 - 3 - 1) / (9 - 1 - 1)
				elif L == left[1] and R == right[0]:
					chi *= (9 - 3 - 1) / (9 - 2 - 1)
				elif L == left[1] and R == right[1]:
					chi *= (9 - 3 - 1) / (9 - 2 - 1)
				elif L == left[0] and R == right[1]:
					chi *= (9 - 3 - 1) / (9 - 2 - 1)
			elif 'exclusive' in fname and 'both' not in fname:
				s = s + ',' + 'False'
				if L == left[0] and R == right[0]:
					chi *= (9 - 3 - 1) / (9 - 1 - 1)
				elif L == left[0] and R == right[2]:
					chi *= (9 - 3 - 1) / (9 - 1 - 1)
				elif L == left[0] and R == right[1]:
					chi *= (9 - 3 - 1) / (9 - 2 - 1)
				elif (L == left[1] or L == left[2]) and (R == right[0] or R == right[1]):
					chi *= (9 - 3 - 1) / (9 - 2 - 1)
			else:
				print(f)

			for i in range(len(data)-1):
				for j in range(len(data[i])):
					if j == 0:					
						s = s + ',' + str(-round(data[i][j], 2))
					else:
						s = s + ',' + str(round(data[i][j], 2))
			s = s + ',' + str(round(chi,2))


			fi.write(s + '\n')

fi.close()
