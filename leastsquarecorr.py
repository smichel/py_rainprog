import numpy as np

def leastsquarecorr(dataArea, corrArea):
    #%Calculates a leastsquare correlation between 2 matrices c and d

    c_l = len(corrArea)
    d_l = len(dataArea)

    c_d = np.zeros([d_l + c_l, d_l + c_l])
    corr = np.ones([d_l + 2 * c_l, d_l + 2 * c_l]) * np.inf
    corr[c_l:(d_l + c_l), c_l:(d_l + c_l)] = dataArea

    k = 0; m = 0
    for i in range(d_l + c_l):
        for j in range(d_l + c_l):
            c_d[k, m] = np.sum(np.sum(np.square(corr[i:i + c_l, j:j + c_l] - corrArea)))
            m += 1
        m = 0
        k += 1

    return c_d