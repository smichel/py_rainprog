import numpy as np
class Square:

    def __init__(self, cRange, maxima, status, rainThreshold, distThreshold, dist, id):
        self.cRange = cRange  # cRange
        self.maxima = maxima  # center point of the square and its rainfall intensity
        self.status = status  # status of the maximum
        self.rainThreshold = rainThreshold  # minimal rainthreshold
        self.distThreshold = distThreshold  # distance threshold to radarboundary
        self.dist = dist  # global distancefield
        self.id = id  # square identifier
        self.shiftX = []
        self.shiftY = []
        self.histX = []  # history of X displacements
        self.histY = []  # history of Y displacements
        self.histMaxima = []  # history of maxima locations

    def add_maximum(self, maximum):
        self.histMaxima.append(maximum)

class totalField:

    def __init__(self, fields, rainThreshold, distThreshold, dist, numMaxes, shiftX, shiftY):
        self.fields = fields
        self.rainThreshold = rainThreshold  # minimal rainthreshold
        self.distThreshold = distThreshold  # distance threshold to radarboundary
        self.dist = dist  # global distancefield
        self.numMaxes = numMaxes
        self.shiftX = shiftX
        self.shiftY = shiftY

    def return_maxima(self, time):
        maxima = np.empty([len(self.fields), 3])
        if time == 0:
            for q, field in enumerate(self.fields):
                maxima[q, 0:3] = field.maxima
        else:
            for q, field in enumerate(self.fields):
                maxima[q, 0:3] = field.histMaxima[time]
        return maxima