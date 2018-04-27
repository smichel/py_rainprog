import numpy as np
import numpy.ma as ma
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import itertools


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
        self.histStdX = []  # history of stdX
        self.histStdY = []  # history of stdY

        self.norm = []  # length of displacement
        self.angle = []  # angle of displacement
        self.meanNorm = []
        self.meanAngle = []
        self.histNorm = []  # history of norms
        self.histAngle = []  # history of angles
        self.stdNorm = []
        self.stdAngle = []
        self.histMeanNorm = []
        self.histMeanAngle = []
        self.histStdNorm = []
        self.histStdAngle = []
        self.relStdNorm = []
        self.histRelStdNorm = []
        self.histMaxima = []  # history of maxima locations

        self.lifeTime = 0  # lifetime of the field in timesteps, begins at 0

    def add_maximum(self, maximum):
        self.histMaxima.append(maximum)

    def add_shift(self, shiftX, shiftY):
        self.histX.append(shiftX)
        self.histY.append(shiftY)

    def add_norm(self, norm):
        self.histNorm.append(norm)

    def add_angle(self, angle):
        self.histAngle.append(angle)

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
        self.meanX = []
        self.meanY = []
        self.histMeanX = []
        self.histMeanY = []
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
            if field.lifeTime >= self.trainTime:
                fieldMeanX.append(field.meanX)

        return np.asarray(fieldMeanX)

    def return_fieldMeanY(self):
        fieldMeanY = []

        for field in self.activeFields:
            if field.lifeTime >= self.trainTime:
                fieldMeanY.append(field.meanY)

        return np.asarray(fieldMeanY)

    def return_fieldHistX(self):
        fieldHistX = []
        for field in self.activeFields:
            if field.lifeTime >= self.trainTime:
                fieldHistX.append(field.histX[-self.trainTime:-1])

        return np.asarray(fieldHistX)

    def return_fieldHistY(self):
        fieldHistY = []
        for field in self.activeFields:
            if field.lifeTime >= self.trainTime:
                fieldHistY.append(field.histY[-self.trainTime:-1])

        return np.asarray(fieldHistY)

    def return_fieldStdX(self):
        fieldStdX = []

        for field in self.activeFields:
            if field.lifeTime >= self.trainTime:
                fieldStdX.append(field.stdX)

        return np.asarray(fieldStdX)

    def return_fieldStdY(self):
        fieldStdY = []

        for field in self.activeFields:
            if field.lifeTime >= self.trainTime:
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

        status=list(status)

        for i, field in enumerate(self.activeFields):
            if field.lifeTime > 1:
                if (field.maxima[0, 1:3] == field.histMaxima[-1][0, 1:3]).all():
                    try:
                        status.remove(i)
                    except ValueError:
                        pass

        for i in reversed(range(len(self.activeFields))):
            if i not in status:
                self.inactiveFields.append(self.activeFields[i])
                self.inactiveFields[-1].lifeTime = -1
                del self.activeFields[i]

    def update_fields(self):

        self.histMeanX.append(self.meanX)
        self.histMeanY.append(self.meanY)

        for field in self.activeFields:
            field.lifeTime = field.lifeTime + 1
            #self.activeIds.append(field.id)
            if field.lifeTime >= self.trainTime:
                field.meanX = np.nanmean(np.asarray(field.histX[-self.trainTime:-1]))
                field.meanY = np.nanmean(np.asarray(field.histY[-self.trainTime:-1]))
                field.stdX = np.nanstd(np.asarray(field.histX[-self.trainTime:-1]))
                field.stdY = np.nanstd(np.asarray(field.histY[-self.trainTime:-1]))
                field.meanNorm = np.linalg.norm([field.meanX, field.meanY])
                field.meanAngle = get_metangle(field.meanX, field.meanY)
                field.stdNorm = np.nanstd(np.asarray(field.histNorm[-self.trainTime:-1]))
                field.relStdNorm = field.stdNorm / field.meanNorm
                field.stdAngle = get_metangle(field.stdX, field.stdY)
                field.stdAngle = field.stdAngle.filled()
                field.histMeanX.append(field.meanX)
                field.histMeanY.append(field.meanY)
                field.histStdX.append(field.stdX)
                field.histStdY.append(field.stdY)
                field.histMeanNorm.append(field.meanNorm)
                field.histMeanAngle.append(field.meanAngle)
                field.histStdNorm.append(field.stdNorm)
                field.histRelStdNorm.append(field.relStdNorm)
                field.histStdAngle.append(field.stdAngle)


        for field in self.inactiveFields:
            field.lifeTime = field.lifeTime - 1
            if field.lifeTime <= -self.trainTime*4:
                self.inactiveFields.remove(field)
            #self.activeIds.append(field.id)

        #do that

    def testangles(self):
        for field in reversed(self.activeFields):
            if field.lifeTime < self.trainTime:
                self.activeFields.remove(field)

        filter = []
        extraFilter = set()
        for t in range(self.trainTime):
            shiftX = np.empty([len(self.activeFields)])
            shiftY = np.empty([len(self.activeFields)])
            status = np.arange(len(self.activeFields))

            for i in range(len(self.activeFields)):
                shiftX[i] = self.activeFields[i].histX[-t-1]
                shiftY[i] = self.activeFields[i].histY[-t-1]

            lengths = np.sqrt(np.square(shiftX) + np.square(shiftY)) * self.res

            shiftXex = shiftX[lengths <= 800]
            shiftYex = shiftY[lengths <= 800]
            extraFilter = extraFilter.union(set(status[lengths > 800]))
            status = status[lengths <= 800]

            meanXex = np.empty([len(shiftXex)])
            meanYex = np.empty([len(shiftYex)])

            for i in range(len(meanXex)):
                meanXex[i] = np.mean(np.delete(shiftXex, i))
                meanYex[i] = np.mean(np.delete(shiftYex, i))

            # shiftX = shiftXex[(np.sign(meanXex) == np.sign(shiftXex)) | (np.sign(meanYex) == np.sign(shiftYex))]
            # shiftY = shiftYex[(np.sign(meanXex) == np.sign(shiftXex)) | (np.sign(meanYex) == np.sign(shiftYex))]
            # status = status[(np.sign(meanXex) == np.sign(shiftXex)) | (np.sign(meanYex) == np.sign(shiftYex))]
            angleEx = get_metangle(meanXex, meanYex)
            angle = get_metangle(shiftXex, shiftYex)
            a = 180 - np.abs(np.abs(angle - angleEx) - 180)

            status = status[a<45]
            filter.extend(list(status))

        unique, counts = np.unique(np.array(filter), return_counts=True)
        #extraFilter = np.arange(len(self.activeFields)) in extraFilter
        filter = (np.full_like(counts, self.trainTime) - counts) >= 3

        for i in reversed(range(len(self.activeFields))):
            if filter[i]:
                self.inactiveFields.append(self.activeFields[i])
                self.inactiveFields[-1].lifeTime = -1
                del self.activeFields[i]

def get_metangle(x, y):
    '''Get meteorological angle of input vector.

    Args:
        x (numpy.ndarray): X-components of input vectors.
        y (numpy.ndarray): Y-components of input vectors.

    Returns:
        (numpy.ma.core.MaskedArray): Meteorological angles.

    '''
    mask = np.logical_and(x == 0, y == 0)  # Vectors (0, 0) not valid.
    met_ang = ma.masked_array((90 - np.degrees(np.arctan2(x, y)) + 360) % 360, mask=mask,
                                  fill_value=np.nan)
    return met_ang

def interp_weights(xy, uv, d=2):
    #  from https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
    tri = qhull.Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts, fill_value=np.nan):
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    ret[np.any(wts < -1e5, axis=1)] = fill_value
    return ret