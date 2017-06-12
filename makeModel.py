import sympy as sp
import models
import dill as pickle

act = models.actin(models.eFuncTwoPlaneShortActinModel,[0,1],sp.symbols('a b c'),blockSize=8)
pickle.dump(act, open('model.dat','wb+'))