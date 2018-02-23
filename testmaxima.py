import numpy as np

def testmaxima(maxima, nestedData, rainThreshold):

    maxima = maxima[maxima[:, 0] > rainThreshold]

    return maxima
