import numpy as np
from math import *

def norm(v1):
    return sqrt(np.sum(np.power(v1, 2)))

def maxIgnoreNan(v):
    maxVal = -inf
    maxInd = -1
    for i in range(len(v)):
        if not v[i] == nan and v[i] > maxVal:
            maxVal = v[i]
            maxInd = i
    return maxVal, maxInd 