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
import triangle
import sympy as sp

act = models.actin(models.eFuncLongTwoPlaneActinModel,[0,1],sp.symbols('a b c'),blockSize=8)
