import numpy as np

def testmaxima(maxima, nestedData, rainThreshold):
    t = 0
    for q in range(len(maxima)):
        if t:
            q -= 1

        if maxima[q, 0] < rainThreshold:
            maxima = np.delete(maxima, q, 0)
            t = 1

    return maxima
