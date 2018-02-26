import numpy as np

def findmaxima(maxima, nestedData, cRange, numMaxes, rainThreshold, distThreshold, dist):
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
    if not maxima.size:
        maxima[0, :] = sorted[-1, :]

    dummy = sorted

    for i in range(numMaxes - len(maxima)):
        distance = np.zeros([len(dummy), len(maxima)])
        for j in range(0, len(maxima)):
            distance[:, j] = np.sqrt(np.square(maxima[j, 1] - dummy[:, 1]) + np.square(maxima[j, 2] - dummy[:, 2]))
        potPoints = np.flatnonzero(np.prod(distance >= cRange, axis=1))
        if dummy[potPoints[-1], 0] > rainThreshold:
            maxima = np.row_stack([maxima, np.reshape(dummy[potPoints[-1], :], (1, 3))])
            dummy = dummy[potPoints, :]
        else:
            return maxima
    return maxima
