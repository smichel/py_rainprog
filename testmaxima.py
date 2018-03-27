import numpy as np

def testmaxima(fields, nestedData, rainThreshold, distThreshold, res):

    maxima = np.empty([len(fields), 3])
    status = np.arange(len(fields))
    for i in range(len(fields)):
        maxima[i, 0:3] = fields[i].maxima

    mask = maxima[:, 0] > rainThreshold
    maxima = maxima[mask]
    status = status[mask]

    middle = nestedData.shape
    dist = np.sqrt(np.square(maxima[:, 1] - (middle[0] - 1) / 2) + np.square(maxima[:, 2] - (middle[1] - 1) / 2)) * res
    maxima = maxima[dist < distThreshold, :]
    status = status[dist < distThreshold]

    maximaProx = np.empty([len(maxima)])
    for q in range(len(maxima)):
        maximaProx[q] = np.mean([nestedData[int(maxima[q, 1] - 1), int(maxima[q, 2])],
                                nestedData[int(maxima[q, 1] + 1), int(maxima[q, 2])],
                                nestedData[int(maxima[q, 1]), int(maxima[q, 2] - 1)],
                                nestedData[int(maxima[q, 1]), int(maxima[q, 2] + 1)]])

    maxima = maxima[maximaProx > rainThreshold, :]
    status = status[maximaProx > rainThreshold]

    contained = np.zeros([len(fields)], dtype = bool)
    for i in range(len(fields)):
        if i in status:
            contained[i] = False


    fields = fields[contained]
    return fields
