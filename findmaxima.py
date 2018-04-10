import numpy as np
from init import Square, totalField

def findmaxima(fields, nestedData, cRange, numMaxes, rainThreshold, distThreshold, dist):
    grid = len(nestedData[1])
    dims = nestedData.shape
    nestedData = nestedData.flatten()
    distFlat = dist.flatten()
    sorted = np.empty([grid*grid, 3])

    sortedIdx = np.argsort(nestedData)
    distFlat = distFlat[sortedIdx]
    sorted[:, 0] = nestedData[sortedIdx]
    sorted[:, 1:3] = np.transpose(np.unravel_index(sortedIdx, dims))
    sorted = sorted[distFlat < distThreshold, :]

    if not len(fields):
        fields.append(Square(cRange, np.reshape(sorted[-1, :], (1, 3)), 1, rainThreshold, distThreshold, dist, id))

    dummy = sorted
    for i in range(numMaxes - len(fields)):
        distance = np.zeros([len(dummy), len(fields)])
        for j in range(0, len(fields)):
            distance[:, j] = np.sqrt(np.square(fields[j].maxima[0, 1] - dummy[:, 1]) + np.square(fields[j].maxima[0, 2] - dummy[:, 2]))
        potPoints = np.flatnonzero(np.prod(distance >= cRange*2, axis=1))
        if dummy[potPoints[-1], 0] > rainThreshold:
            fields.append(Square(cRange, np.reshape(dummy[potPoints[-1], :], (1, 3)), 1, rainThreshold, distThreshold, dist, i))
            dummy = dummy[potPoints, :]
        else:
            return fields
    return fields
