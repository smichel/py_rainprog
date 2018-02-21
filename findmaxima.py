import numpy as np

def findmaxima(maxima, nestedData, cRange, numMaxes, rainThreshold):
    grid = len(nestedData[1])
    dims = nestedData.shape
    nestedData = nestedData.flatten()

    sorted = np.empty([grid*grid, 3])

    sortedIdx = np.argsort(nestedData)
    sorted[:, 0] = nestedData[sortedIdx]
    sorted[:, 1:3] = np.transpose(np.unravel_index(sortedIdx, dims))
    if not maxima.size:
        maxima[0, :] = sorted[-1, :]

    dummy = sorted

    for i in range(numMaxes - len(maxima) + 1):
        distance = np.zeros([len(dummy), i+1])
        for j in range(0, i+1):
            distance[:, j] = np.sqrt(np.square(maxima[j, 1] - dummy[:, 1]) + np.square(maxima[j, 2] - dummy[:, 2]))
        potPointsIni = distance >= cRange
        potPoints = np.flatnonzero(np.prod(potPointsIni, axis=1))
        if dummy[potPoints[-1], 0] > rainThreshold:
            maxima = np.row_stack([maxima, np.reshape(dummy[potPoints[-1], :], (1, 3))])
            dummy = dummy[potPoints, :]
        else:
            return maxima
    return maxima




