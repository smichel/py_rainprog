import numpy as np


class Square:

    def __init__(self, cRange, maxima, status, rainThreshold, distThreshold, dist, id):
        self.cRange = cRange  # cRange
        self.maxima = maxima  # center point of the field and its rainfall intensity
        self.status = status  # status of the maximum
        self.rainThreshold = rainThreshold  # minimal rainthreshold
        self.distThreshold = distThreshold  # distance threshold to radarboundary
        self.dist = dist  # global distancefield
        self.id = id  # square identifier
        self.shiftX = []
        self.shiftY = []
        self.histX = []  # history of X displacements
        self.histY = []  # history of Y displacements
        self.meanX = []  # mean of shiftX over the trainTime
        self.meanY = []  # mean of shiftY over the trainTime
        self.histMeanX = []  # history of meanX
        self.histMeanY = []  # history of meanY
        self.stdX = []  # standard deviation of the x displacement
        self.stdY = []  # standard deviation of the y displacement
        self.histStdX = []
        self.histStdY = []
        self.histMaxima = []  # history of maxima locations
        self.lifeTime = 0  # lifetime of the field in timesteps, begins at 0

    def add_maximum(self, maximum):
        self.histMaxima.append(maximum)

    def add_shift(self, shiftX, shiftY):
        self.histX.append(shiftX)
        self.histY.append(shiftY)


class totalField:

    def __init__(self, fields, rainThreshold, distThreshold, dist, numMaxes, res, cRange, trainTime):
        self.activeFields = fields
        self.inactiveFields = []
        self.rainThreshold = rainThreshold  # minimal rainthreshold
        self.distThreshold = distThreshold  # distance threshold to radarboundary
        self.dist = dist  # global distancefield
        self.numMaxes = numMaxes  # number of maxima
        self.shiftX = []
        self.shiftY = []
        self.res = res  # resolution of the grid in m
        self.cRange = cRange  # correlation range in gridboxes
        self.trainTime = trainTime
        #self.ids = np.arange(self.numMaxes*self.trainTime)
        #self.activeIds = []  # ids in use (active and inactive fields)
        #self.inactiveIds = []  # ids not in use

    def return_maxima(self, time):
        maxima = np.empty([len(self.activeFields), 3])
        if time == 0:
            for q, field in enumerate(self.activeFields):
                maxima[q, 0:3] = field.maxima
        else:
            for q, field in enumerate(self.activeFields):
                maxima[q, 0:3] = field.histMaxima[time]
        return maxima

    def return_fieldMeanX(self):
        fieldMeanX = []
        for field in self.activeFields:
            fieldMeanX.append(field.meanX)

        return np.asarray(fieldMeanX)

    def return_fieldMeanY(self):
        fieldMeanY = []

        for field in self.activeFields:
            fieldMeanY.append(field.meanY)

        return np.asarray(fieldMeanY)

    def return_fieldStdX(self):
        fieldStdX = []

        for field in self.activeFields:
            fieldStdX.append(field.stdX)

        return np.asarray(fieldStdX)

    def return_fieldStdY(self):
        fieldStdY = []

        for field in self.activeFields:
            fieldStdY.append(field.stdY)

        return np.asarray(fieldStdY)

    def testmaxima(self, nestedData):
        maxima = np.empty([len(self.activeFields), 3])
        status = np.arange(len(self.activeFields))

        for i in range(len(self.activeFields)):
            maxima[i, 0:3] = self.activeFields[i].maxima

        mask = maxima[:, 0] > self.rainThreshold
        maxima = maxima[mask]
        status = status[mask]

        middle = nestedData.shape
        dist = np.sqrt(
            np.square(maxima[:, 1] - (middle[0] - 1) / 2) + np.square(maxima[:, 2] - (middle[1] - 1) / 2)) * self.res
        maxima = maxima[dist < self.distThreshold, :]
        status = status[dist < self.distThreshold]

        maximaProx = np.empty([len(maxima)])
        for q in range(len(maxima)):
            maximaProx[q] = np.mean([nestedData[int(maxima[q, 1] - 1), int(maxima[q, 2])],
                                     nestedData[int(maxima[q, 1] + 1), int(maxima[q, 2])],
                                     nestedData[int(maxima[q, 1]), int(maxima[q, 2] - 1)],
                                     nestedData[int(maxima[q, 1]), int(maxima[q, 2] + 1)]])

        maxima = maxima[maximaProx > self.rainThreshold, :]
        status = status[maximaProx > self.rainThreshold]

        for i in reversed(range(len(self.activeFields))):
            if i not in status:
                self.inactiveFields.append(self.activeFields[i])
                self.inactiveFields[-1].lifeTime = -1
                del self.activeFields[i]

    def update_fields(self):

        for field in self.activeFields:
            field.lifeTime = field.lifeTime + 1
            #self.activeIds.append(field.id)
            if field.lifeTime > self.trainTime:
                field.meanX = np.mean(np.asarray(field.histX[-self.trainTime:-1]))
                field.meanY = np.mean(np.asarray(field.histY[-self.trainTime:-1]))
                field.stdX = np.std(np.asarray(field.histX[-self.trainTime:-1]))
                field.stdY = np.std(np.asarray(field.histY[-self.trainTime:-1]))
                field.histMeanX.append(field.meanX)
                field.histMeanY.append(field.meanY)
                field.histStdX.append(field.stdX)
                field.histStdY.append(field.stdY)

        for field in self.inactiveFields:
            field.lifeTime = field.lifeTime - 1
            #if field.lifeTime <= -self.trainTime:
            #    self.inactiveFields.remove(field)
            #self.activeIds.append(field.id)

        #do that