from data import data
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)

c,bindingF,length,dl,name = data('d')
ax.scatter(bindingF,length, s=54)
ax.errorbar(bindingF,length, markersize=54,yerr=dl, linestyle='None', label=name)
c,bindingF,length,dl,name = data('h')
ax.scatter(bindingF,length, s=54, c='silver')
ax.errorbar(bindingF,length, markersize=54,yerr=dl, c='silver', linestyle='None', label=name)
c,bindingF,length,dl,name = data('b')
ax.scatter(bindingF,length, s=54, c='darkseagreen')
ax.errorbar(bindingF,length, markersize=54,yerr=dl, c='darkseagreen', linestyle='None', label=name)

ax.set_xlim([-0.1,1.1])
ax.set_xlabel('Cofilin Binding Fraction')
ax.set_ylabel('Filament length ($\mu m$)')
ax.legend()
plt.tight_layout()
plt.savefig('data.pdf')
