import numpy as np
import transferMatrixClass as tmc
from scipy.optimize import brentq

# Helpers


def brentqWrapper(func):
    init = 3

    while init < 1e10 and np.sign(func(init)) == np.sign(func(-init)):
        init *= 2

    a = brentq(func, -init, init)
    return a


def meanL(c, inter, j):
    b = inter - j
    x = c * np.exp(b)
    ret = np.zeros(x.shape)
    ret = x * (-1 + np.sqrt(1 + 4 * x)) / (2 * x + 1 - np.sqrt(1 + 4 * x))
    return ret

# Models


def eFuncIsing(state, params):  # Accepts binary inputs
    e = 0
    for i in range(len(state)):
        e += state[i] * params[0]
    for i in range(len(state) - 1):
        e += state[i] * state[i + 1] * params[1]
    return e


def eFuncLongIsing(state, params):  # Accepts binary inputs
    e = 0
    for i in range(len(state)):
        e += state[i] * params[0]
    for i in range(len(state) - 2):
        e += state[i] * state[i + 2] * params[1]
    return e


def eFuncSymmetricTwoPlaneActinModel(state, params):  # Accepts binary inputs
    e = 0
    # Binding energy
    for i in range(len(state)):
        e += params[0] * state[i]
    # In-plane bonds
    for i in range(len(state)):
        if sum(state[max(0, i - 2):min(len(state), i + 4):2]) > 0:
            e += params[1]
    # Out-of-plane bonds
    for i in range(len(state) - 1):
        if sum(state[i:min(len(state), i + 4)]) > 0:
            e += params[2]
    return e


def eFuncTwoPlaneActinModel(state, params):  # Accepts binary inputs
    e = 0
    # Binding energy
    for i in range(len(state)):
        e += params[0] * state[i]
    # In-plane bonds
    for i in range(2, len(state)):
        if state[i] == 1 or state[i - 2] == 1:
            e += params[1]
    # Out-of-plane bonds
    for i in range(len(state) - 1):
        if sum(state[i:min(len(state), i + 4)]) > 0:
            e += params[2]
    return e


def eFuncTwoPlaneShortActinModel(state, params):  # Accepts binary inputs
    e = 0
    # Binding energy
    for i in range(len(state)):
        e += params[0] * state[i]
    # In-plane bonds
    for i in range(2, len(state)):
        if state[i] == 1 or state[i - 2] == 1:
            e += params[1]
    # Out-of-plane bonds
    for i in range(len(state) - 1):
        if sum(state[i:min(len(state), i + 3)]) > 0:
            e += params[2]
    return e


def eFuncTwoPlaneVeryShortActinModel(state, params):  # Accepts binary inputs
    e = 0
    # Binding energy
    for i in range(len(state)):
        e += params[0] * state[i]
    # In-plane bonds
    for i in range(2, len(state)):
        if state[i] == 1 or state[i - 2] == 1:
            e += params[1]
    # Out-of-plane bonds
    for i in range(len(state) - 1):
        if sum(state[i:min(len(state), i + 2)]) > 0:
            e += params[2]
    return e

def actinModelRuleFactory(left, right):
    '''
    Generates the energy function for an Actin binding model.

    left and right specify Cofilin sites that are allosterically affected.

    outPlane is as defined in actinModelFactory and equals list(range(left+1, right+1)),
    so that (-1,0] includes just the outPlane bond between the -1 and 0 sites.

    The inPlane bonds we use are just the even-numbered affected Cofilin sites.

    Hence

    inPlane = (even numbers in range(left,right+1)) / 2

    (we divide by 2 because there's a factor of 2 put back in in actinModelFactory)
    '''

    outPlane = list(range(left + 1, right + 1))
    inPlane = list(i//2 for i in range(left, right + 1) if i%2 == 0)
    return actinModelFactory(inPlane, outPlane)

def actinModelInhomogeneousRuleFactory(left, right):
    '''
    Generates the energy function for an Actin binding model.

    left and right specify Cofilin sites that are allosterically affected.

    outPlane is as defined in actinModelFactory and equals list(range(left+1, right+1)),
    so that (-1,0] includes just the outPlane bond between the -1 and 0 sites.

    The inPlane bonds we use are just the even-numbered affected Cofilin sites.

    Hence

    inPlane = (even numbers in range(left,right+1)) / 2

    (we divide by 2 because there's a factor of 2 put back in in actinModelFactory)
    '''

    outPlane = list(range(left + 1, right + 1))
    inPlane = list(i//2 for i in range(left, right + 1) if i%2 == 0)
    return actinModelInhomoheneousFactory(inPlane, outPlane)


def exclusiveActinModelRuleFactory(left, right, sides='both'):
    '''
    Generates the energy function for an Actin binding model.

    left and right specify Cofilin sites that are allosterically affected.

    outPlane is as defined in actinModelFactory and equals list(range(left+1, right+1)),
    so that [-1,0] includes just the outPlane bond between the -1 and 0 sites.

    The inPlane bonds we use are just the even-numbered affected Cofilin sites,
    excluding those on the ends, if any.

    Hence

    inPlane = (even numbers in range(left + 1,right)) / 2

    (we divide by 2 because there's a factor of 2 put back in in actinModelFactory)
    '''

    outPlane = list(range(left + 1, right + 1))
    if sides=='both':
        inPlane = list(i//2 for i in range(left + 1, right) if i%2 == 0)
    elif sides == 'left':
        inPlane = list(i//2 for i in range(left + 1, right + 1) if i%2 == 0)
    elif sides == 'right':
        inPlane = list(i//2 for i in range(left, right) if i%2 == 0)

    return actinModelFactory(inPlane, outPlane)

def actinModelFactory(inPlane, outPlane):
    '''
    Generates the energy function for an Actin binding model.

    inPlane is a list of integers specifying which in-plane bonds contribute
    to the energy. This is indexed so that zero means the bond the Cofilin is attached
    to. Incrementing one such index by one moves the state-space index by 2 because we
    skip the out-of-plane bonds.

    outPlane is a list of integers specifying which out-of-plane bonds contribute to the
    energy. This is indexed so that zero means the bond left of the in-plane one the Cofilin
    is attached to. Incrementing one such index by one moves the state-space index
    by one as well.
    '''

    # Construct energy function
    def energy(state, params):
        # State and params are binary inputs

        e = 0
        # Binding energy
        for i in range(len(state)):
            e += params[0] * state[i]

        # In-plane bonds
        for i in range(len(state)): # There are as many in-plane bonds as there are Cofilin binding sites
            isAffected = False
            for j in inPlane:
                ind = i - 2 * j  # Minus so that j > 0 means the Cofilin is to the left of the bond
                if ind >= 0 and ind < len(state):
                    if state[ind] == 1:
                        isAffected = True
            if isAffected:
                e += params[1]

        # Out-of-plane bonds
        for i in range(len(state) - 1):
            isAffected = False
            for j in outPlane:
                ind = i - j  # Minus so that j > 0 means the Cofilin is to the left of the bond
                if ind >= 0 and ind < len(state):
                    if state[ind] == 1:
                        isAffected = True
            if isAffected:
                e += params[2]

        return e

    print(inPlane, outPlane)
    if len(inPlane) > 0:
        inPlaneMax = max(inPlane)
        inPlaneMin = min(inPlane)
    else:
        inPlaneMax = 0
        inPlaneMin = 0
    if len(outPlane) > 0:
        outPlaneMax = max(outPlane)
        outPlaneMin = min(outPlane)
    else:
        outPlaneMax = 0
        outPlaneMin = 0    

    inPlaneMax *= 2
    inPlaneMin *= 2
    blockSize = max(inPlaneMax, outPlaneMax) - min(inPlaneMin, outPlaneMin)

    blockSize = max(1, blockSize)

    return energy, blockSize

def actinModelInhomoheneousFactory(inPlane, outPlane):
    '''
    Generates the energy function for an Actin binding model.

    inPlane is a list of integers specifying which in-plane bonds contribute
    to the energy. This is indexed so that zero means the bond the Cofilin is attached
    to. Incrementing one such index by one moves the state-space index by 2 because we
    skip the out-of-plane bonds.

    outPlane is a list of integers specifying which out-of-plane bonds contribute to the
    energy. This is indexed so that zero means the bond left of the in-plane one the Cofilin
    is attached to. Incrementing one such index by one moves the state-space index
    by one as well.
    '''

    # Construct energy function
    def energy(state, params):
        # State and params are binary inputs

        e = 0
        # Binding energy
        for i in range(len(state)):
            e += params[0][i] * state[i]

        # In-plane bonds
        for i in range(len(state)): # There are as many in-plane bonds as there are Cofilin binding sites
            isAffected = False
            for j in inPlane:
                ind = i - 2 * j  # Minus so that j > 0 means the Cofilin is to the left of the bond
                if ind >= 0 and ind < len(state):
                    if state[ind] == 1:
                        isAffected = True
            if isAffected:
                e += params[1][i]

        # Out-of-plane bonds
        for i in range(len(state) - 1):
            isAffected = False
            for j in outPlane:
                ind = i - j  # Minus so that j > 0 means the Cofilin is to the left of the bond
                if ind >= 0 and ind < len(state):
                    if state[ind] == 1:
                        isAffected = True
            if isAffected:
                e += params[2][i]

        return e

    print(inPlane, outPlane)
    if len(inPlane) > 0:
        inPlaneMax = max(inPlane)
        inPlaneMin = min(inPlane)
    else:
        inPlaneMax = 0
        inPlaneMin = 0
    if len(outPlane) > 0:
        outPlaneMax = max(outPlane)
        outPlaneMin = min(outPlane)
    else:
        outPlaneMax = 0
        outPlaneMin = 0    

    inPlaneMax *= 2
    inPlaneMin *= 2
    blockSize = max(inPlaneMax, outPlaneMax) - min(inPlaneMin, outPlaneMin)

    blockSize = max(1, blockSize)

    return energy, blockSize


# Actin class


class actin:

    def __init__(self, model, stateRange, params, blockSize=None):
        self.model = model
        self.stateRange = stateRange
        self.params = params

        # Initialize symbolic matrices
        print('Constructing matrices...')

        if blockSize is None:
            self.tm, self.left, self.right, self.blockSize = tmc.transferMatrixVariableSize(
                self.model, stateRange, params)
        else:
            self.blockSize = blockSize
            self.tm, self.left, self.right = tmc.transferMatrix(
                self.model, stateRange, params, blockSize, check=False)

        # Wrapped matrices
        print('Wrapping...')
        self.T, self.eL, self.eR, self.dP, self.dQ, self.dW, self.dQL, self.dWL, self.dQR, self.dWR = tmc.wrapper(
            self.tm, self.left, self.right, params)
        print('Done.')

    def fN(self, params, n=50):
        return tmc.fN(self.T, self.eL, self.eR, self.dP, self.dQ, self.dW, self.dQL, self.dWL, self.dQR, self.dWR, params, self.blockSize, n)

    def cofilinBindingFrac(self, params):
        # Assumes that params[0] is the binding energy
        return self.fN(params)[3]

    def bindingFinder(self, params, bf):
        # Assumes that params[0] is the binding energy
        def ff(bindingE):
            pcopy = np.copy(params)
            pcopy[0] = bindingE
            return self.cofilinBindingFrac(pcopy) - bf

        bindingE = brentqWrapper(ff)
        if ff(bindingE) > 1e-8:
            print('Error:', bindingE, ff(bindingE), ff(
                bindingE - 1e-3), ff(bindingE + 1e-3))
        pcopy = np.copy(params)
        pcopy[0] = bindingE
        return pcopy
