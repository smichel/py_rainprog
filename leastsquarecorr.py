import numpy as np

def leastsquarecorr(dataArea, corrArea):
    #%Calculates a leastsquare correlation between 2 matrices c and d

    cLen = len(corrArea)
    c_d = np.zeros([cLen, cLen])

    k = 0; m = 0
    for i in range(cLen):
        for j in range(cLen):
            c_d[k, m] = np.sum(np.sum(np.square(dataArea[i:i + cLen, j:j + cLen] - corrArea)))
            m += 1
        m = 0
        k += 1

    return c_d