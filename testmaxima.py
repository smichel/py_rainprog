import numpy as np

def testmaxima(maxima, nestedData, rainThreshold, distThreshold, res):

    maxima = maxima[maxima[:, 0] > rainThreshold]

    middle = nestedData.shape
    dist = np.sqrt(np.square(maxima[:, 1] - (middle[0] - 1) / 2) + np.square(maxima[:, 2] - (middle[1] - 1) / 2)) * res
    maxima = maxima[dist < distThreshold, :]

    return maxima
