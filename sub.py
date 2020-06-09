import time
import subprocess

for ran in range(0,5):
	for i in range(-ran, 1):
		left = i
		right = i + ran

		for string in ['exclusive', 'inclusive']:
			if string == 'inclusive':
				name = str(abs(left)) + '_' + str(right) + '_job_' + string
				fi = open(name, 'w+')

				s = '#!/bin/bash -l\n#SBATCH -t 15:00:00\n#SBATCH --nodes=1 --ntasks-per-node=20\n#SBATCH --mail-type ALL\ncd $SLURM_SUBMIT_DIR\nsrun -n 1 -c 20 python emsP.py d' + ' ' + str(left) + ' ' + str(right) + ' 20' + ' ' + string
				fi.write(s)
				fi.close()
				subprocess.Popen('sbatch ' + name, shell=True)
				time.sleep(2)
			else:
				for sides in ['both', 'left']:
					name = str(abs(left)) + '_' + str(right) + '_job_' + string + '_' + sides
					fi = open(name, 'w+')

					s = '#!/bin/bash -l\n#SBATCH -t 15:00:00\n#SBATCH --nodes=1 --ntasks-per-node=20\n#SBATCH --mail-type ALL\ncd $SLURM_SUBMIT_DIR\nsrun -n 1 -c 20 python emsP.py d' + ' ' + str(left) + ' ' + str(right) + ' 20' + ' ' + string + ' ' + sides
					fi.write(s)
					fi.close()
					subprocess.Popen('sbatch ' + name, shell=True)
					time.sleep(2)
