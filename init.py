import os
import numpy as np
import numpy.ma as ma
import netCDF4
import time
import scipy.spatial.qhull as qhull
import xarray as xr
from osgeo import osr
from scipy.interpolate import griddata, RegularGridInterpolator, interp1d
from scipy.ndimage import map_coordinates
from skimage.feature import match_template
import scipy.odr
from sklearn.feature_extraction import image
from numba import jit
from wradlib_snips import make_2D_grid, reproject
import cv2
from datetime import datetime
import datetime
import h5py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class radarData:

    def __init__(self, path):
        self.rainThreshold = 0.5

    def set_auxillary_geoData(self, dwd, lawr, HHGposition):
        dwd.HHGdist = np.sqrt(np.square(dwd.XCar_nested - dwd.XCar_nested.min() - HHGposition[0] * dwd.resolution) +
                              np.square(dwd.YCar_nested - dwd.YCar_nested.min() - HHGposition[1] * dwd.resolution))

        #dwd.HHG_cart_points = np.concatenate((np.reshape(dwd.XCar - dwd.XCar.min() - HHGposition[0] * dwd.resolution,
        #                                                 (dwd.d_s * dwd.d_s, 1)),
        #                                      np.reshape(dwd.YCar - dwd.YCar.min() - HHGposition[1] * dwd.resolution,
        #                                                 (dwd.d_s * dwd.d_s, 1))), axis=1)


    def interpolate(self, values, vtx, wts, fill_value=np.nan):
        ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
        ret[np.any(wts < -1e5, axis=1)] = fill_value
        return ret

    def interp_weights(self, xy, uv, d=2):
        #  from https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
        tri = qhull.Delaunay(xy)
        simplex = tri.find_simplex(uv)
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = uv - temp[:, d]
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
    ##TODO change start time explicit for dwd and pattern
    def initial_maxima(self):

        self.progField = Totalfield(
            Totalfield.findmaxima([], self.nested_data[self.startTime, :, :], self.cRange, self.numMaxima,
                                  self.rainThreshold, self.distThreshold, self.dist_nested, self.resolution),
            self.rainThreshold, self.distThreshold, self.dist_nested, self.numMaxima, self.nested_data,
            self.resolution, self.cRange, self.trainTime, self.d_s)
        self.progField.deltaT = 1

    def find_displacement(self,prog):
        for t in range(self.trainTime):
            if len(self.progField.activeFields) < self.numMaxima:
                self.progField.assign_ids()
            self.progField.test_maxima(self.nested_data[self.startTime + t,:,:])
            self.progField.prog_step(self.startTime+t)
            self.progField.update_fields()

        self.progField.test_angles2()
        #self.progField.test_angles()

        self.meanXDisplacement = np.nanmean(self.progField.return_fieldHistX())
        self.meanYDisplacement = np.nanmean(self.progField.return_fieldHistY())
        allFieldsNorm = self.progField.return_fieldHistMeanNorm()
        self.normEqualOneSum = np.sum(self.progField.return_fieldHistNorm()!=0)
        allFieldsAngle = self.progField.return_fieldHistMeanAngle()
        self.allFieldsNorm = allFieldsNorm[~np.isnan(allFieldsAngle)]
        self.allFieldsAngle = allFieldsAngle[~np.isnan(allFieldsAngle)]
        try:
            #self.covNormAngle = np.cov(allFieldsNorm[~np.isnan(np.sin(allFieldsAngle * allFieldsNorm))], np.sin(allFieldsAngle * allFieldsNorm)[~np.isnan(np.sin(allFieldsAngle * allFieldsNorm))])
            self.covNormAngle = np.cov(self.progField.return_fieldHistX().flatten(),self.progField.return_fieldHistY().flatten())
            self.gaussMeans = [self.meanXDisplacement, self.meanYDisplacement]
        except ValueError:
            self.covNormAngle = np.nan
            self.gaussMeans = np.nan

class Square:

    def __init__(self, cRange, maxima, status, rainThreshold, distThreshold, dist):
        self.cRange = cRange  # cRange
        self.maxima = maxima  # center point of the field and its rainfall intensity
        self.status = status  # status of the maximum
        self.rainThreshold = rainThreshold  # minimal rainthreshold
        self.distThreshold = distThreshold  # distance threshold to radarboundary
        self.dist = dist  # global distancefield
        self.id = 0  # square identifier

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

    def get_id(self, inactiveIds):
        self.id = inactiveIds[-1]

class Totalfield:

    def __init__(self, fields, rainThreshold, distThreshold, dist, numMaxes, nested_data, res, cRange, trainTime,d_s):
        self.activeFields = fields
        self.rejectedFields = []
        self.inactiveFields = []
        self.rainThreshold = rainThreshold  # minimal rainthreshold
        self.distThreshold = distThreshold  # distance threshold to radarboundary
        self.dist = dist  # global distancefield
        self.numMaxes = numMaxes  # number of maxima
        self.nested_data = nested_data
        self.shiftX = []
        self.shiftY = []
        self.meanX = []
        self.meanY = []
        self.histMeanX = []
        self.histMeanY = []
        self.res = res  # resolution of the grid in m
        self.cRange = cRange  # correlation range in gridboxes
        self.trainTime = trainTime
        self.ids = np.arange(self.numMaxes*self.trainTime, 0, -1)
        self.activeIds = []  # ids in use (active and inactive fields)
        self.inactiveIds = list(self.ids)  # ids not in use
        self.deltas = []
        self.sum_square_delta = []
        self.sum_square = []
        self.betas = []
        self.d_s = d_s
        self.assign_ids()
    def return_maxima(self):
        maxima = np.empty([len(self.activeFields), 3])
        for q, field in enumerate(self.activeFields):
            maxima[q, 0:3] = field.maxima
        return maxima

    def return_histMaxima(self,time):
        maxima = np.empty([len(self.activeFields), 3])
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

    def return_fieldHistNorm(self):
        fieldHistNorm = []
        for field in self.activeFields:
            if field.lifeTime >= self.trainTime:
                fieldHistNorm.append(field.histNorm[-self.trainTime:-1])

        return np.asarray(fieldHistNorm)

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

    def return_fieldSumSquare(self):
        fieldSumSquare=[]

        for field in self.activeFields:
            if field.lifeTime >= self.trainTime:
                fieldSumSquare.append(field.sum_square)

        return np.asarray(fieldSumSquare)

    def return_fieldTotalLength(self):
        fieldTotalLength=[]

        for field in self.activeFields:
            if field.lifeTime >= self.trainTime:
                fieldTotalLength.append(field.totalLength)

        return np.asarray(fieldTotalLength)

    def return_fieldRelStdNorm(self):
        fieldRelStdNorm = []

        for field in self.activeFields:
            if field.lifeTime >= self.trainTime:
                fieldRelStdNorm.append(field.relStdNorm)

        return np.asarray(fieldRelStdNorm)

    def return_fieldHistMeanNorm(self):
        fieldHistMeanNorm = []

        for field in self.activeFields:
            if field.lifeTime >= self.trainTime:
                fieldHistMeanNorm.extend(field.histMeanNorm)

        return np.asarray(fieldHistMeanNorm)

    def return_fieldHistNorm(self):
        fieldHistNorm = []

        for field in self.activeFields:
            if field.lifeTime >= self.trainTime:
                fieldHistNorm.extend(field.histNorm)

        return np.asarray(fieldHistNorm)

    def return_fieldHistMeanAngle(self):
        fieldHistMeanAngle = []

        for field in self.activeFields:
            if field.lifeTime >= self.trainTime:
                fieldHistMeanAngle.extend(field.histMeanAngle)

        return np.asarray(fieldHistMeanAngle)

    def return_fieldHistAngle(self):
        fieldHistAngle = []

        for field in self.activeFields:
            if field.lifeTime >= self.trainTime:
                fieldHistAngle.extend(field.histAngle)

        return np.asarray(fieldHistAngle)

    def test_maxima(self, nestedData):


        maxima = np.empty([len(self.activeFields), 3])
        status = np.arange(len(self.activeFields))

        for i in range(len(self.activeFields)):
            maxima[i, 0:3] = self.activeFields[i].maxima

        #mask = maxima[:, 0] > self.rainThreshold
        #maxima = maxima[mask]
        #status = status[mask]

        middle = nestedData.shape
        dist = np.sqrt(
            np.square(maxima[:, 1] - (middle[0] - 1) / 2) + np.square(maxima[:, 2] - (middle[1] - 1) / 2)) * self.res
        maxima = maxima[dist < self.distThreshold, :]
        status = status[dist < self.distThreshold]

        #maximaProx = np.empty([len(maxima)])
        #for q in range(len(maxima)):
        #    maximaProx[q] = np.mean([nestedData[int(maxima[q, 1] - 1), int(maxima[q, 2])],
        #                             nestedData[int(maxima[q, 1] + 1), int(maxima[q, 2])],
        #                             nestedData[int(maxima[q, 1]), int(maxima[q, 2] - 1)],
        #                             nestedData[int(maxima[q, 1]), int(maxima[q, 2] + 1)]])

        #maxima = maxima[maximaProx > self.rainThreshold, :]
        #status = status[maximaProx > self.rainThreshold]

        status=list(status)

        for i in reversed(range(len(self.activeFields))):
            if i not in status:
                self.inactiveFields.append(self.activeFields[i])
                self.inactiveFields[-1].lifeTime = -1
                del self.activeFields[i]

    def prog_step(self,t):

        for field in self.activeFields:

            corrArea = self.nested_data[t-self.deltaT+1,
                       (int(field.maxima[0, 1]) - self.cRange):(int(field.maxima[0, 1]) + self.cRange),
                       (int(field.maxima[0, 2]) - self.cRange):(int(field.maxima[0, 2]) + self.cRange)]
            dataArea = self.nested_data[t+1,
                       (int(field.maxima[0, 1]) - self.cRange * 2):(int(field.maxima[0, 1]) + self.cRange * 2),
                       (int(field.maxima[0, 2]) - self.cRange * 2):(int(field.maxima[0, 2]) + self.cRange * 2)]
            # maybe consider using "from skimage.feature import match_template" template matching
            # or using shift,error,diffphase = register_translation(self.nested_data[t+5,:,:],self.nested_data[t,:,:])
            # print("Detected pixel offset (y, x): {}".format(shift))
            # http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_template.html
            c = leastsquarecorr(dataArea, corrArea)
            cIdx = np.unravel_index((np.nanargmin(c)), c.shape)
            field.shiftX = (cIdx[0] - 0.5 * len(c))/self.deltaT
            field.shiftY = (cIdx[1] - 0.5 * len(c))/self.deltaT
            field.norm = np.linalg.norm([field.shiftX, field.shiftY])

            field.angle = get_metangle(field.shiftX, field.shiftY)
            field.angle = field.angle.filled()
            field.add_norm(field.norm)
            field.add_angle(field.angle)
            field.add_maximum(np.copy(field.maxima))
            field.add_shift(field.shiftX, field.shiftY)
            field.maxima[0, 0] = self.nested_data[t, int(field.maxima[0, 1] + field.shiftX),
                                                  int(field.maxima[0, 2] + field.shiftY)]
            field.maxima[0, 1] = field.maxima[0, 1] + field.shiftX
            field.maxima[0, 2] = field.maxima[0, 2] + field.shiftY


            # if field.norm == 1:
            #     corrArea = self.nested_data[t,
            #                (int(field.maxima[0, 1]) - self.cRange):(int(field.maxima[0, 1]) + self.cRange),
            #                (int(field.maxima[0, 2]) - self.cRange):(int(field.maxima[0, 2]) + self.cRange)]
            #     dataArea = self.nested_data[t + 2,
            #                (int(field.maxima[0, 1]) - self.cRange * 2):(int(field.maxima[0, 1]) + self.cRange * 2),
            #                (int(field.maxima[0, 2]) - self.cRange * 2):(int(field.maxima[0, 2]) + self.cRange * 2)]
            #     c = leastsquarecorr(dataArea, corrArea)
            #     cIdx = np.unravel_index((np.nanargmin(c)), c.shape)
            #     field.shiftX = int(cIdx[0] - 0.5 * len(c))*0.5
            #     field.shiftY = int(cIdx[1] - 0.5 * len(c))*0.5
            #     field.norm = np.linalg.norm([field.shiftX, field.shiftY])
            #     field.angle = get_metangle(field.shiftX, field.shiftY)
            #     field.angle = field.angle.filled()
            #     field.add_norm(field.norm)
            #     field.add_angle(field.angle)
            #     field.add_maximum(np.copy(field.maxima))
            #     field.add_shift(field.shiftX, field.shiftY)
            #     field.maxima[0, 0] = self.nested_data[t, int(field.maxima[0, 1] + field.shiftX),
            #                                           int(field.maxima[0, 2] + field.shiftY)]
            #     field.maxima[0, 1] = (field.maxima[0, 1] + field.shiftX)
            #     field.maxima[0, 2] = (field.maxima[0, 2] + field.shiftY)


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
                field.meanAngle = field.meanAngle.filled()
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
            if field.lifeTime <= -self.trainTime:
                self.inactiveIds.append(field.id)
                self.activeIds.remove(field.id)
                self.inactiveFields.remove(field)

        #do that
    def test_angles2(self):

        def f(B, x):
            return B[0] * x + B[1]

        fieldNums = 0
        for field in reversed(self.activeFields):
            if field.lifeTime < 3:
                self.rejectedFields.add(field)
                self.activeFields.remove(field)
            else:
                fieldNums += 1


        linear = scipy.odr.Model(f)
        for i,field in enumerate(self.activeFields):
            mydata = scipy.odr.Data([x[0,1] for x in field.histMaxima],[x[0,2] for x in field.histMaxima])
            myodr = scipy.odr.ODR(mydata, linear, beta0=[0,0])
            myoutput = myodr.run()
            field.beta = myoutput.beta
            field.delta = myoutput.delta
            field.sum_square = myoutput.sum_square
            field.sum_square_delta = myoutput.sum_square_delta
            field.totalLength = np.sqrt(np.square(field.histMaxima[0].squeeze()[1]-field.histMaxima[-1].squeeze()[1])+np.square(field.histMaxima[0].squeeze()[2]-field.histMaxima[-1].squeeze()[2]))

        for i in reversed(range(len(self.activeFields))):

            if self.activeFields[i].sum_square>self.activeFields[i].totalLength/2:

                self.inactiveFields.append(self.activeFields[i])
                self.inactiveFields[-1].lifeTime = -1
                del self.activeFields[i]

    def test_angles(self):
        fieldNums = 0
        for field in reversed(self.activeFields):
            if field.lifeTime < self.trainTime:
                self.activeFields.remove(field)
            else:
                fieldNums += 1

        angleFilter = list(range(fieldNums))
        lengthFilter = list(range(fieldNums))
        maxDist = self.cRange * self.res * 1.2
        for t in range(self.trainTime):
            shiftX = np.empty([len(self.activeFields)])
            shiftY = np.empty([len(self.activeFields)])
            status = np.arange(len(self.activeFields))

            for i in range(len(self.activeFields)):
                shiftX[i] = self.activeFields[i].histX[-t - 1]
                shiftY[i] = self.activeFields[i].histY[-t - 1]

            lengths = np.sqrt(np.square(shiftX) + np.square(shiftY)) * self.res

            zeros = sum(lengths == 0)

            shiftX = shiftX[lengths != 0]
            shiftY = shiftY[lengths != 0]
            zero = status[lengths == 0]
            status = status[lengths != 0]
            lengths = lengths[lengths != 0]

            shiftXex = shiftX[lengths <= maxDist]
            shiftYex = shiftY[lengths <= maxDist]
            lengthFilter.extend(status[lengths <= maxDist])
            lengthFilter.extend(zero)

            status = status[lengths <= maxDist]

            meanXex = np.empty([len(shiftXex)])
            meanYex = np.empty([len(shiftYex)])

            for i in range(len(meanXex)):
                meanXex[i] = np.mean(np.delete(shiftXex, i))
                meanYex[i] = np.mean(np.delete(shiftYex, i))

            angleEx = get_metangle(meanXex, meanYex)
            angle = get_metangle(shiftXex, shiftYex)
            a = 180 - np.abs(np.abs(angle - angleEx) - 180)

            status = status[a < 45]
            angleFilter.extend(list(status))

        lengthUnique, lengthCounts = np.unique(np.array(lengthFilter), return_counts=True)
        angleUnique, angleCounts = np.unique(np.array(angleFilter), return_counts=True)
        aFilter = (np.full_like(angleCounts, self.trainTime) - angleCounts) > int(self.trainTime/2) # angleFilter
        lFilter = (np.full_like(lengthCounts, self.trainTime+1) - lengthCounts) > 1 # lengthFilter

        for i in reversed(range(len(self.activeFields))):
            if aFilter[i] | lFilter[i]:
                self.inactiveFields.append(self.activeFields[i])
                self.inactiveFields[-1].lifeTime = -1
                del self.activeFields[i]

    def assign_ids(self):
        for field in self.activeFields:
            if not field.id:
                field.get_id(self.inactiveIds)
                self.activeIds.append(self.inactiveIds[-1])
                del self.inactiveIds[-1]

    def findmaxima(fields, nestedData, cRange, numMaxes, rainThreshold, distThreshold, dist, resolution):
        grid = len(nestedData[1])
        dims = nestedData.shape
        nestedData = nestedData.flatten()
        distFlat = dist.flatten()
        sorted = np.empty([grid * grid, 3])
        minDistance = 3000/resolution

        sortedIdx = np.argsort(nestedData)
        distFlat = distFlat[sortedIdx]
        sorted[:, 0] = nestedData[sortedIdx]
        sorted[:, 1:3] = np.transpose(np.unravel_index(sortedIdx, dims))
        sorted = sorted[distFlat < distThreshold, :]

        if not len(fields):
            fields.append(Square(cRange, np.reshape(sorted[-1, :], (1, 3)), 1, rainThreshold, distThreshold, dist))

        dummy = sorted
        for i in range(numMaxes - len(fields)):
            distance = np.zeros([len(dummy), len(fields)])
            for j in range(0, len(fields)):
                distance[:, j] = np.sqrt(
                    np.square(fields[j].maxima[0, 1] - dummy[:, 1]) + np.square(fields[j].maxima[0, 2] - dummy[:, 2]))
            potPoints = np.flatnonzero(np.prod(distance >= minDistance, axis=1))
            if not len(potPoints):
                return fields
            if dummy[potPoints[-1], 0] > rainThreshold:
                fields.append(
                    Square(cRange, np.reshape(dummy[potPoints[-1], :], (1, 3)), 1, rainThreshold, distThreshold, dist))
                dummy = dummy[potPoints, :]
            else:
                return fields
        return fields

class DWDData(radarData, Totalfield):

    def __init__(self, filePath,dwdTime):

        '''Read in DWD radar data.

        Reads DWD radar data and saves data to object. Only attributes
        needed for my calculations are read in. If more information
        about the file and the attributes is wished, check out the
        `hdf5 <https://support.hdfgroup.org/HDF5/>`_-file with hdfview
        or hd5dump -H.

        Args:
            filename (str): Name of radar data file.

        '''
        if os.path.splitext(filePath)[1]=='.nc':
            with xr.open_dataset(filePath) as dataset:
                try:
                    data = dataset['reflectivity'].values
                    if np.ma.is_masked(data):
                        data.fill_value = -32.5
                        self.refl = data.filled()
                    else:
                        self.refl = data
                    self.azi = dataset['azimuth']
                    self.r = dataset['range']
                    self.time = dataset.time.values

                except:
                    data = dataset['CLT_Corr_Reflectivity'].values
                    if np.ma.is_masked(data):
                        data.fill_value = -32.5
                        self.refl = data.filled()
                    else:
                        self.refl = data
                    self.azi = dataset['azimuth']
                    self.r = dataset['range']
                    self.time = dataset.time.values

            self.refl[np.isnan(self.refl)] = -32.5
            self.time = (self.time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        else:
            with h5py.File(filePath, 'r') as boo:
                # radar site coordinates and elevation
                lon_site = boo.get('where').attrs['lon']
                lat_site = boo.get('where').attrs['lat']
                alt_site = boo.get('where').attrs['height']
                self.sitecoords = (lon_site, lat_site, alt_site)
                self.elevation = boo.get('dataset1/where').attrs['elangle']

                # Number of azimuth rays, start azimuth, azimuth steps
                az_rays = boo.get('dataset1/where').attrs['nrays']
                az_start = boo.get('dataset1/where').attrs['startaz']
                az_steps = boo.get('dataset1/how').attrs['angle_step']

                # Number of range bins, start range, range steps
                r_bins = boo.get('dataset1/where').attrs['nbins']
                r_start = boo.get('dataset1/where').attrs['rstart']
                r_steps = boo.get('dataset1/where').attrs['rscale']

                # Azimuth and range arrays
                self.r = np.arange(r_start, r_start + r_bins*r_steps, r_steps)
                self.azi = np.arange(az_start, az_start + az_rays*az_steps, az_steps)

                # Corrected reflectivity data
                gain = boo.get('dataset1/data1/what').attrs['gain']
                offset = boo.get('dataset1/data1/what').attrs['offset']
                refl = boo.get('dataset1/data1/data')
                self.refl = refl*gain + offset
                self.time = boo.get('what').attrs['time']
                self.date =boo.get('what').attrs['date']
                year = int("".join(map(chr, self.date))[0:4])
                mon = int("".join(map(chr, self.date))[4:6])
                day = int("".join(map(chr, self.date))[6:8])
                hour = int("".join(map(chr, self.time))[0:2])
                minute = int("".join(map(chr, self.time))[2:4])
                second = int("".join(map(chr, self.time))[4:6])

                self.hour = []
                self.minute = []
                self.second = []

                self.hour.append(hour)
                self.minute.append(minute)
                self.second.append(second)

                self.time = time.mktime((datetime.datetime(year,mon,day,hour,minute,second)+datetime.timedelta(hours=1)).utctimetuple())

            #time = (time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        #self.startTime = np.argmin(np.abs(time.mktime((datetime.datetime(date[0],date[1],date[2],date[3],date[4])-
        #                                               datetime.timedelta(hours=-1,minutes=self.trainTime/2)).utctimetuple())-self.time))
        super().__init__(filePath)

        self.sitecoords = [10.04683, 54.0044]
        self.resolution = 200  # horizontal resolution in m
        self.trainTime = 6  # 6 Timesteps for training to find the displacement vector (equals 25 minutes)
        self.numMaxima = 20  # number of tracked maxima
        self.distThreshold = 50000
        self.getGrid()
        self.offset = self.cRange * 2
        self.nested_data[0, self.offset:self.offset + self.d_s, self.offset:self.offset + self.d_s] = self.gridding()
        self.startTime=0 # number of the file where the startpoint for the prognosis is

    def getGrid(self):
        mett_azi = met2math_angle(self.azi)
        aziCos = np.cos(np.radians(mett_azi))
        aziSin = np.sin(np.radians(mett_azi))
        xPolar = np.outer(self.r, aziCos)
        yPolar = np.outer(self.r, aziSin)
        xPolar = np.reshape(xPolar, (len(self.azi) * len(self.r), 1))
        yPolar = np.reshape(yPolar, (len(self.azi) * len(self.r), 1))
        points = np.concatenate((xPolar, yPolar), axis=1)
        proj_cart = osr.SpatialReference()
        proj_cart.ImportFromEPSG(32632)
        proj_geo = osr.SpatialReference()
        proj_geo.ImportFromEPSG(4326)
        lawr_sitecoords = [9.973997, 53.56833]
        #lawr_cartcoords = reproject(lawr_sitecoords, projection_target=proj_cart)
        boo_cartcoords = reproject(self.sitecoords[0:2], projection_target=proj_cart)
        max_dist = 60000
        self.polar_grid= reproject(points+boo_cartcoords, projection_source=proj_cart, projection_target=proj_geo)
        self.cRange = int(6000 / self.resolution)

        self.xCar = np.arange(-max_dist, max_dist + 1, self.resolution).squeeze()
        self.xCar_nested = np.arange(-max_dist - self.cRange * 2 * self.resolution,
                                     max_dist + self.cRange * 2 * self.resolution + 1, self.resolution).squeeze()

        self.yCar = np.arange(-max_dist, max_dist + 1, self.resolution).squeeze()
        self.yCar_nested = np.arange(-max_dist- self.cRange * 2 * self.resolution,
                                     max_dist+ self.cRange * 2 * self.resolution + 1, self.resolution).squeeze()
        self.d_s = len(self.xCar)
        self.d_s_nested = len(self.xCar_nested)
        [self.XCar, self.YCar] = np.meshgrid(self.xCar, self.yCar)
        [self.XCar_nested, self.YCar_nested] = np.meshgrid(self.xCar_nested, self.yCar_nested)

        self.dist = np.sqrt(np.square(self.XCar) + np.square(self.YCar))
        self.dist_nested = np.sqrt(np.square(self.XCar_nested) + np.square(self.YCar_nested))


        self.Lon, self.Lat = get_Grid(lawr_sitecoords, max_dist, self.resolution)
        self.Lon_nested, self.Lat_nested = get_Grid(lawr_sitecoords, max_dist+ self.cRange * 2 * self.resolution, self.resolution)

        target = np.zeros([self.Lon.shape[0] * self.Lat.shape[1], 2])
        target[:, 0] = self.Lon.flatten()
        target[:, 1] = self.Lat.flatten()




        self.vtx, self.wts = super().interp_weights(self.polar_grid, target)
        self.nested_data = np.zeros([1, self.d_s + 4 * self.cRange, self.d_s + 4 * self.cRange])

        # self.cart_points = np.concatenate((np.reshape(self.Lon_nested, (self.d_s * self.d_s, 1)),
        #                                   np.reshape(self.Lat_nested, (self.d_s * self.d_s, 1))), axis=1)
        # cart_points = points + boo_cartcoords
        # target = np.zeros([self.XCar.shape[0] * self.XCar.shape[1], 2])
        # target[:, 0] = self.XCar.flatten()+lawr_cartcoords[0]
        # target[:, 1] = self.YCar.flatten()+lawr_cartcoords[1]
        # plt.figure()
        # plt.scatter(cart_points[:, 0], cart_points[:, 1])
        # plt.scatter(target[:, 0], target[:, 1], c='r')

    def gridding(self):

        rPolar = z2rainrate(self.refl).T
        rPolar = np.reshape(rPolar, (len(self.azi) * len(self.r), 1)).squeeze()
        return np.reshape(super().interpolate(rPolar.flatten(), self.vtx, self.wts), (self.d_s, self.d_s))



    def addTimestep(self, filePath):
        if os.path.splitext(filePath)[1]=='.nc':
            with xr.open_dataset(filePath) as dataset:
                try:
                    data = dataset['reflectivity'].values
                    if np.ma.is_masked(data):
                        data.fill_value = -32.5
                        self.refl = data.filled()
                    else:
                        self.refl = data
                    self.azi = dataset['azimuth'].values
                    self.r = dataset['range'].values
                    timedummy = dataset.time.values

                except:
                    data = dataset['CLT_Corr_Reflectivity'].values
                    if np.ma.is_masked(data):
                        data.fill_value = -32.5
                        self.refl = data.filled()
                    else:
                        self.refl = data
                    self.azi = dataset['azimuth'].values
                    self.r = dataset['range'].values
                    timedummy = dataset.time.values

                timedummy = (timedummy - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                self.refl[np.isnan(self.refl)] = -32.5
                self.time = np.append(self.time, timedummy)
                nested_data = np.zeros([1, self.d_s + 4 * self.cRange, self.d_s + 4 * self.cRange])
                nested_data[0, self.offset: self.d_s + self.offset,
                self.offset: self.d_s + self.offset] = self.gridding()
                self.nested_data = np.vstack((self.nested_data, nested_data))

        else:
            with h5py.File(filePath, 'r') as boo:
                gain = boo.get('dataset1/data1/what').attrs['gain']
                offset = boo.get('dataset1/data1/what').attrs['offset']
                refl = boo.get('dataset1/data1/data')
                self.refl = refl * gain + offset
                timedummy = boo.get('what').attrs['time']
                date =boo.get('what').attrs['date']

                year = int("".join(map(chr, date))[0:4])
                mon = int("".join(map(chr, date))[4:6])
                day = int("".join(map(chr, date))[6:8])
                hour = int("".join(map(chr, timedummy))[0:2])
                minute = int("".join(map(chr, timedummy))[2:4])
                second = int("".join(map(chr, timedummy))[4:6])

                self.hour.append(int("".join(map(chr, timedummy))[0:2]))
                self.minute.append(int("".join(map(chr, timedummy))[2:4]))
                self.second.append(int("".join(map(chr, timedummy))[4:6]))
                self.time = np.append(self.time, time.mktime((datetime.datetime(year,mon,day,hour,minute,second)+datetime.timedelta(hours=1)).utctimetuple()))

                nested_data = np.zeros([1, self.d_s + 4 * self.cRange, self.d_s + 4 * self.cRange])
                nested_data[0, self.offset: self.d_s + self.offset,
                self.offset: self.d_s + self.offset] = self.gridding()
                self.nested_data = np.vstack((self.nested_data, nested_data))

    def timeInterpolation(self, timeSteps):
        # faster implementation of the timeInterpolation
        z = np.arange(self.R.shape[2])
        interpolating_function = RegularGridInterpolator((z,), self.R.transpose())
        z_ = np.linspace(0, z[-1], timeSteps)
        self.R = interpolating_function(z_).transpose()


    def extrapolation(self, progTimeSteps,varianceFactor=1):

        self.prog_data = np.zeros([progTimeSteps, self.nested_data.shape[1], self.nested_data.shape[2]])
        filtersize = 10 # default filtersize
        rho = (self.covNormAngle_norm[0,1]*varianceFactor)/(np.sqrt(self.covNormAngle_norm[0,0]*varianceFactor)*np.sqrt(self.covNormAngle_norm[1,1])*varianceFactor)
        [y,x] = np.meshgrid(np.arange(-filtersize,filtersize+1),np.arange(-filtersize,filtersize+1))
        self.kernel = twodgauss(x,y,np.sqrt(self.covNormAngle_norm[0,0])*varianceFactor,np.sqrt(self.covNormAngle_norm[1,1])*varianceFactor,rho,-self.gaussMeans_norm[0],-self.gaussMeans_norm[1])
        while np.sum(self.kernel)<(1-0.0001):
            filtersize += 10  # default filtersize
            [y, x] = np.meshgrid(np.arange(-filtersize, filtersize + 1), np.arange(-filtersize, filtersize + 1))

            self.kernel = twodgauss(x, y, np.sqrt(self.covNormAngle[0, 0]) * varianceFactor,
                                    np.sqrt(self.covNormAngle[1, 1]) * varianceFactor, rho, -self.gaussMeans[0],
                                    -self.gaussMeans[1])

        self.kernel = self.kernel/np.sum(self.kernel)

        for t in range(progTimeSteps):
            if t == 0:
                self.prog_data[t, :, :] = self.nested_data[-1,:,:]>self.rainThreshold
                self.prog_data[t, :, :] = cv2.filter2D(self.prog_data[t,:,:], -1, self.kernel)

            #self.prog_data[t, :, :] = booDisplacement(self,self.nested_data[-1,:,:], (self.gaussMeans[0]/10)*t, (self.gaussMeans[1]/10)*t)
            else:
                self.prog_data[t, :, :] = cv2.filter2D(self.prog_data[t-1,:,:], -1, self.kernel)

            self.time = np.append(self.time, self.time[-1] + 30)

class LawrData(radarData, Totalfield):

    def __init__(self, filePath):

        with xr.open_dataset(filePath) as dataset:
            try:
                data = dataset['reflectivity'].values
                if np.ma.is_masked(data):
                    data.fill_value = -32.5
                    self.refl = data.filled()
                else:
                    self.refl = data
                try:
                    self.azi = dataset['azimuth'].values
                    self.r = dataset['range'].values

                except:
                    self.azi = dataset['Azimuth'].values
                    self.r = dataset['Distance'].values

                self.r = dataset['range'].values
                self.time = dataset.time.values

            except:
                data = dataset['CLT_Corr_Reflectivity'].values
                if np.ma.is_masked(data):
                    data.fill_value = -32.5
                    self.refl = data.filled()
                else:
                    self.refl = data
                try:
                    self.azi = dataset['azimuth'].values
                    self.r = dataset['range'].values

                except:
                    self.azi = dataset['Azimuth'].values
                    self.r = dataset['Distance'].values
                    self.time = dataset.Time.values

            self.refl[np.isnan(self.refl)] = -32.5
            self.time = (self.time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            super().__init__(filePath)
            self.trainTime = 10  # 10 Timesteps for training to find the displacement vector (equals 5 minutes)
            self.numMaxima = 20  # number of tracked maxima
            self.resolution = 100
            self.timeSteps = len(self.time)
            self.distThreshold = 19000
            aziCorr = -5
            self.azi = np.mod(self.azi + aziCorr, 360)
            self.cRange = int(
                1000 / self.resolution)  # 800m equals an windspeed of aprox. 100km/h and is set as the upper boundary for a possible cloud movement
            self.offset = self.cRange * 2

            self.lon = 9.973997  # location of the hamburg radar
            self.lat = 53.56833
            self.sitecoords = [self.lon, self.lat]
            self.getGrid()
            self.zsl = 100  # altitude of the hamburg radar
            self.nested_data = self.gridding()

    def getGrid(self):

        mett_azi = met2math_angle(self.azi)
        aziCos = np.cos(np.radians(mett_azi))
        aziSin = np.sin(np.radians(mett_azi))
        xPolar = np.outer(self.r, aziCos)
        xPolar = np.reshape(xPolar, (333 * 360, 1))
        yPolar = np.outer(self.r, aziSin)
        yPolar = np.reshape(yPolar, (333 * 360, 1))
        self.nesting_factor = 2
        self.points = np.concatenate((xPolar, yPolar), axis=1)

        self.xCar = np.arange(-20000, 20000 + 1, self.resolution).squeeze()
        self.yCar = np.arange(-20000, 20000 + 1, self.resolution).squeeze()

        [self.XCar, self.YCar] = np.meshgrid(self.xCar, self.yCar)

        self.Lon,self.Lat = get_Grid(self.sitecoords,20000,self.resolution)
        self.Lon_nested, self.Lat_nested = get_Grid(self.sitecoords, 20000 + self.cRange * self.nesting_factor * self.resolution, self.resolution)

        self.dist = np.sqrt(np.square(self.xCar) + np.square(self.YCar))

        xCar_nested = np.arange(-20000 - self.cRange * self.nesting_factor * self.resolution,
                                20000 + self.cRange * self.nesting_factor * self.resolution + 1, self.resolution).squeeze()
        yCar_nested = np.copy(xCar_nested)

        [XCar_nested, YCar_nested] = np.meshgrid(xCar_nested, yCar_nested)

        self.dist_nested = np.sqrt(np.square(xCar_nested) + np.square(YCar_nested))

        self.target_nested = np.zeros([XCar_nested.shape[0] * XCar_nested.shape[1], 2])
        self.target_nested[:, 0] = XCar_nested.flatten()
        self.target_nested[:, 1] = YCar_nested.flatten()

        self.target = np.zeros([self.XCar.shape[0] * self.XCar.shape[1], 2])
        self.target[:, 0] = self.XCar.flatten()
        self.target[:, 1] = self.YCar.flatten()

        self.d_s = len(self.XCar)
        self.d_s_nested = len(xCar_nested)

        self.R = np.empty([self.timeSteps, self.d_s, self.d_s])
        self.vtx, self.wts = self.interp_weights(self.points, self.target)

    def gridding(self):

        rPolar = z2rainrate(self.refl)
        R = np.empty([self.timeSteps, self.d_s, self.d_s])
        nested_data = np.zeros([self.timeSteps, self.d_s + self.nesting_factor * 2 * self.cRange, self.d_s + self.nesting_factor * 2 * self.cRange])

        for t in range(nested_data.shape[0]):
            rPolarT = rPolar[t, :, :].T
            rPolarT = np.reshape(rPolarT, (333 * 360, 1)).squeeze()
            R[t, :, :] = np.reshape(self.interpolate(rPolarT.flatten(), self.vtx, self.wts), (self.d_s, self.d_s))
            R[t, (self.dist >= np.max(self.r))] = 0
            nested_data[t, self.nesting_factor  * self.cRange: self.nesting_factor  * self.cRange + self.d_s,
            self.nesting_factor  * self.cRange: self.nesting_factor  * self.cRange + self.d_s] = R[t, :, :]

        nested_data = np.nan_to_num(nested_data)
        return nested_data

    def addTimestep(self,filePath):
        with xr.open_dataset(filePath) as dataset:
            try:
                data = dataset['reflectivity'].values
                if np.ma.is_masked(data):
                    data.fill_value = -32.5
                    self.refl = data.filled()
                else:
                    self.refl = data
                try:
                    self.azi = dataset['azimuth'].values
                    self.r = dataset['range'].values
                    time = dataset.time.values
                except:
                    self.azi = dataset['Azimuth'].values
                    self.r = dataset['Distance'].values
                    time = dataset.Time.values
                self.r = dataset['range'].values


            except:
                data = dataset['CLT_Corr_Reflectivity'].values
                if np.ma.is_masked(data):
                    data.fill_value = -32.5
                    self.refl = data.filled()
                else:
                    self.refl = data
                try:
                    self.azi = dataset['azimuth'].values
                    self.r = dataset['range'].values
                    time = dataset.time.values
                except:
                    self.azi = dataset['Azimuth'].values
                    self.r = dataset['Distance'].values
                    time = dataset.Time.values

            time = (time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            self.refl[np.isnan(self.refl)] = -32.5
            self.time = np.append(self.time, time)
            nested_data = np.zeros([self.timeSteps, self.d_s + self.nesting_factor * 2 * self.cRange, self.d_s + self.nesting_factor * 2 * self.cRange])
            nested_data[:,:,:] = self.gridding()
            self.nested_data = np.vstack((self.nested_data,nested_data))
    def setStart(self,date):
        import time
        import datetime
        self.startTime = np.argmin(np.abs(time.mktime((datetime.datetime(date[0],date[1],date[2],date[3],date[4])-
                                                       datetime.timedelta(hours=-1,minutes=self.trainTime/2)).utctimetuple())-self.time))

    def extrapolation(self, dwd, progTimeSteps, prog,probabilityFlag,varianceFactor=1):
        filtersize = 10 # default filtersize

        rho = (self.covNormAngle[0, 1] * varianceFactor) / (
                    np.sqrt(self.covNormAngle[0, 0]) * varianceFactor * np.sqrt(
                self.covNormAngle[1, 1]) * varianceFactor)
        [y,x] = np.meshgrid(np.arange(-filtersize,filtersize+1),np.arange(-filtersize,filtersize+1))

        self.kernel = twodgauss(x,y,np.sqrt(self.covNormAngle[0,0])*varianceFactor,np.sqrt(self.covNormAngle[1,1])*varianceFactor,rho,-self.gaussMeans[0],-self.gaussMeans[1])
        while np.sum(self.kernel)<(1-0.1):
            filtersize += 10  # default filtersize
            [y, x] = np.meshgrid(np.arange(-filtersize, filtersize + 1), np.arange(-filtersize, filtersize + 1))

            self.kernel = twodgauss(x, y, np.sqrt(self.covNormAngle[0, 0]) * varianceFactor,
                                    np.sqrt(self.covNormAngle[1, 1]) * varianceFactor, rho, -self.gaussMeans[0],
                                    -self.gaussMeans[1])

        self.kernel = self.kernel/np.sum(self.kernel)

        # self.kernel2 = twodgauss(x,y,np.sqrt(self.covNormAngle[0,0])*1.1,np.sqrt(self.covNormAngle[1,1])*1.1,rho,-self.gaussMeans[0],-self.gaussMeans[1])
        # self.kernel2 = self.kernel2/np.sum(self.kernel2)

        self.prog_data = np.copy(self.nested_data[0,:,:])
        self.probabilities = np.zeros([progTimeSteps, self.prog_data.shape[0],self.prog_data.shape[1]])
        #self.probabilities2 = np.zeros([progTimeSteps, self.d_s + 4 * self.cRange, self.d_s + 4 * self.cRange])
        rainThreshold=0.5
        self.progStartIdx = self.startTime + self.trainTime
        self.prog_data[:, :] = self.nested_data[self.progStartIdx, :, :]
        self.prog_start = 3 * ['']

        self.dwdProgStartIdx = np.argmin(np.abs(dwd.time - self.time[self.progStartIdx]))

        self.prog_data[:, :] = nesting(self.prog_data[:, :], self.dist_nested, self.target_nested,
                                          dwd.prog_data[self.dwdProgStartIdx, :, :], dwd, self.r[-1],
                                          self.rainThreshold, self,
                                          self.Lat_nested, self.Lon_nested)

        if probabilityFlag:
            self.probabilities[0,:,:] = self.nested_data[self.progStartIdx,:,:]>self.rainThreshold

            self.probabilities[0,:,:] = nesting(self.probabilities[0, :, :], self.dist_nested, self.target_nested,
                                          dwd.prog_data[self.dwdProgStartIdx, :, :], dwd, self.r[-1],
                                          self.rainThreshold, self,
                                          self.Lat_nested, self.Lon_nested)

            self.probabilities[0, :, :] = cv2.filter2D(self.probabilities[0, :, :], -1, self.kernel)

            # self.probabilities2[0,:,:] = self.nested_data[self.progStartIdx,:,:]>self.rainThreshold
            #
            # self.probabilities2[0,:,:] = nesting(self.probabilities2[0, :, :], self.dist_nested, self.target_nested,
            #                               dwd.prog_data[self.dwdProgStartIdx, :, :]>rainThreshold, dwd, self.r[-1],
            #                               self.rainThreshold, self,
            #                               self.Lat_nested, self.Lon_nested)
            #
            # self.probabilities2[0, :, :] = cv2.filter2D(self.probabilities[0, :, :], -1, self.kernel2)
        #self.prog_data[0,:,:] = cv2.filter2D(self.prog_data[0,:,:],-1,self.kernel)
        for t in range(1,progTimeSteps):
            #self.prog_data[t, :, :] = self.prog_data[t - 1, :, :]

            #self.prog_data[t,:,:] = nesting(self.prog_data[t, :, :], self.dist_nested, self.target_nested,
            #                              dwd.prog_data[self.dwdProgStartIdx + t, :, :], dwd, self.r[-1], self.rainThreshold, self,
            #                              self.Lat_nested, self.Lon_nested)

            #self.prog_data[t, :, :] = cv2.filter2D(self.prog_data[t, :, :], -1, self.kernel)

            if probabilityFlag:
                self.probabilities[t, :, :] = self.probabilities[t - 1, :, :]

                self.probabilities[t, :, :] = nesting(self.probabilities[t, :, :], self.dist_nested, self.target_nested,
                                              dwd.prog_data[self.dwdProgStartIdx + t, :, :], dwd, self.r[-1],
                                              self.rainThreshold, self,
                                              self.Lat_nested, self.Lon_nested)
                self.probabilities[t, :, :] = cv2.filter2D(self.probabilities[t, :, :], -1, self.kernel)

                # self.probabilities2[t, :, :] = self.probabilities2[t - 1, :, :]
                #
                # self.probabilities2[t, :, :] = nesting(self.probabilities2[t, :, :], self.dist_nested, self.target_nested,
                #                               dwd.prog_data[self.dwdProgStartIdx + t, :, :]>rainThreshold, dwd, self.r[-1],
                #                               self.rainThreshold, self,
                #                               self.Lat_nested, self.Lon_nested)
                # self.probabilities2[t, :, :] = cv2.filter2D(self.probabilities2[t, :, :], -1, self.kernel2)
def z2rainrate(z):# Conversion between reflectivity and rainrate, a and b are empirical parameters of the function
    a = np.full_like(z, 77, dtype=np.double)
    b = np.full_like(z, 1.9, dtype=np.double)
    cond1 = z <= 44.0
    a[cond1] = 200
    b[cond1] = 1.6
    cond2 = z <= 36.5
    a[cond2] = 320
    b[cond2] = 1.4
    return ((10 ** (z / 10)) / a) ** (1. / b)

def get_Grid(sitecoords, maxrange, horiz_res):
    #  spatialRef = osr.SpatialReference()
    #  spatialRef.ImportFromEPSG(4326) # lon lat
    #  32632

    proj_cart = osr.SpatialReference()
    proj_cart.ImportFromEPSG(32632)
    proj_geo = osr.SpatialReference()
    proj_geo.ImportFromEPSG(4326)

    trgxyz, trgshape = make_2D_grid(sitecoords, proj_cart, maxrange, 0, horiz_res, 9999)
    trgxy = trgxyz[:, 0:2]
    trgx = trgxyz[:, 0].reshape(trgshape)[0, :, :]
    trgy = trgxyz[:, 1].reshape(trgshape)[0, :, :]
    lon,lat = reproject(trgx,trgy, projection_source=proj_cart, projection_target=proj_geo)
    return lon,lat

def findRadarSite(lawr, dwd):
    lat = np.abs(dwd.Lat_nested - lawr.lat)
    lon = np.abs(dwd.Lon_nested - lawr.lon)
    latIdx = np.unravel_index(np.argmin(lat),lat.shape)[0]
    lonIdx = np.unravel_index(np.argmin(lon),lon.shape)[1]
    return lonIdx, latIdx

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

def met2math_angle(met_angle):
    '''Convert meteorological to mathematical angles.

    Args:
        met_angle (numpy.ndarray): Meteorological angles.

    Returns:
        (numpy.ndarray): Mathematical angles.

    '''
    if np.logical_or(np.any(met_angle > 360),  np.any(met_angle < 0)):
        raise ValueError('Meteorological angles must be between 0 and 360')
    math_angle = (90 - met_angle + 360) % 360
    return math_angle

def polar2xy(r, az):
    '''Transform polar to cartesian x,y-values.

    Transform azimuth angle to mathematical angles and uses trigonometric
    functions to obtain cartesian x,y - values out of angle and radius.

    Note:
        This function does not calculate Cartesian coordinates, but simple x,y
        values. For coordinates, a Cartesian grid object is needed.
exi
    Args:
        r (numpy.ndarray): Array of ranges in m.
        az (numpy.ndarray): Corresponding array of azimuths angles.

    Returns:
        (numpy.ndarray): Array of cartesian x, y-values in m.

    '''
    if np.logical_or(np.any(az < 0), np.any(az > 360)):
        raise ValueError('Azimuth angles must be between 0 and 360')
    if np.any(r < 0):
        raise ValueError('Range values must be greater than 0')
    math_az = met2math_angle(az)  # Transform meteo. to mathematical angles.
    y = np.sin(np.radians(math_az))*r
    x = np.cos(np.radians(math_az))*r
    return np.stack((x, y), axis=-1)


def importance_sampling(nested_data, nested_dist, rMax, xy, yx, xSample, ySample, d_s, cRange):
    nested_data_ = ma.array(nested_data, mask=nested_dist >= rMax, dtype=float,fill_value=np.nan)
    xy = ma.array(xy, mask=nested_dist >= rMax, dtype=int,fill_value=-999999)
    yx = ma.array(yx, mask=nested_dist >= rMax, dtype=int,fill_value=-999999)
    prog_data_ = get_values(xSample, ySample, xy.flatten()[~xy.mask.flatten()], yx.flatten()[~yx.mask.flatten()].flatten(), nested_data)
    nested_data.squeeze()[~nested_data_.mask.squeeze()] = prog_data_
    prog_data = np.reshape(nested_data,[d_s+4*cRange,d_s+4*cRange])
    return prog_data

def create_sample(gaussMeans, covNormAngle, samples):
    '''Draw random samples from a multivariate normal distribution which is specified by the input gaussMeans and covariances.

    Note:
        https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.multivariate_normal.html

    Args:
        gaussMeans(list): Mean of the distribution
        covNormAngle (numpy.ndarray): 2d array containing the covariances
        samples (int): Number of samples to be drawn

    Returns:
        vals (numpy.ndarray): interpolated data 2d array at new x,y coordinates
    '''
    return np.random.multivariate_normal(gaussMeans, covNormAngle, samples).T

def get_values(xSample, ySample, x, y, nested_data):
    '''Returns an interpolated array of the x and y coordinated from the input data array (nested_data).

    Note:
        The new and x and y coordinates need to be positive and within range of the dimensions of nested_data.
        There is no extrapolation implemented.

    Args:
        xSample(numpy.ndarray): data array in 2d
        ySample (numpy.ndarray): 2d array containing new x coordinates
        x (numpy.ndarray): 2d array containing new y coordinates
        y (numpy.ndarray):

    Returns:
        vals (numpy.ndarray): interpolated data 2d array at new x,y coordinates
    '''
    x = np.full((len(xSample), len(x)), x).T
    y = np.full((len(ySample), len(y)), y).T
    x_= x - xSample
    y_= y - ySample
    vals = np.nanmean(interp2d(nested_data, x_, y_),axis=1)
    return vals

@jit(nopython=True)
def interp2d(nested_data, x, y):
    '''Returns an interpolated(bilinear interpolation) array of the x and y coordinated from the input data array (nested_data).

    Note:
        The new and x and y coordinates need to be positive and within range of the dimensions of nested_data.
        There is no extrapolation implemented.

    Args:
        nested_data (numpy.ndarray): data array in 2d
        x (numpy.ndarray): 2d array containing new x coordinates
        y (numpy.ndarray): 2d array containing new y coordinates

    Returns:
        vals (numpy.ndarray): interpolated data 2d array at new x,y coordinates
    '''
    vals = np.empty_like(x)
    for idx, xx in np.ndenumerate(x):
        xi = int(xx)
        xmod = xx % 1
        yy = y[idx]
        yi = int(yy)
        ymod = yy % 1

        vals[idx] = nested_data[xi, yi] * ((1 - xmod) * (1 - ymod)) + \
             nested_data[xi, yi+1] * (1- xmod) * (ymod) + \
             nested_data[xi+1, yi+1] * ((xmod) * ymod) + \
             nested_data[xi+1, yi] * ((xmod) * (1 - ymod))
    return vals


def getFiles(filelist, time):
    files = []
    for i, file in enumerate(filelist):
        if (np.abs(int(file[41:43])- time) <= 0):
            files.append(file)
        #if ((np.abs(int(file[41:43]) - (time + 1)) <= 0) & (int(file[43:45]) == 0)):
        #    files.append(file)

    return files

def dwdFileSelector(directoryPath, time, trainTime = 6):
    booFileList = sorted(os.listdir(directoryPath))
    year = np.asarray([int(x[33:37]) for x in booFileList])
    mon = np.asarray([int(x[37:39]) for x in booFileList])
    day = np.asarray([int(x[39:41]) for x in booFileList])
    hour = np.asarray([int(x[41:43]) for x in booFileList])
    min = np.asarray([int(x[43:45]) for x in booFileList])
    idx = np.where((time[0]==year)&(time[1]==mon)&(time[2]==day)&(time[3]==hour)&((time[4]/2)-(time[4]/2)%5==min))[0][0]
    selectedFiles = booFileList[idx-trainTime:idx+1]
    return selectedFiles

def lawrFileSelector(directoryPath,time,trainTime=10,progTime=60):
    lawrFileList = sorted(os.listdir(directoryPath))
    year = np.asarray([int(x[7:11]) for x in lawrFileList])
    mon = np.asarray([int(x[11:13]) for x in lawrFileList])
    day = np.asarray([int(x[13:15]) for x in lawrFileList])
    hour = np.asarray([int(x[15:17]) for x in lawrFileList])
    idx = np.where((time[2] == day) & (time[3] == hour))[0][0]
    selectedFiles = lawrFileList[idx+int(np.floor((time[4]-trainTime)/60)):idx+int((time[4]+progTime)/60)+1]
    return selectedFiles

def nesting(prog_data, nested_dist, nested_points, boo_prog_data, dwd, rMax, rainthreshold, lawr, HHGlat, HHGlon):
    boo_pixels = ((dwd.dist_nested>= rMax) & (dwd.dist_nested<= lawr.dist_nested.max()))
    hhg_pixels = ((lawr.dist_nested >= rMax) & (lawr.dist_nested<= lawr.dist_nested.max()))

    lat1 = dwd.Lat_nested - lawr.Lat_nested.min()
    lat2 = dwd.Lat_nested - lawr.Lat_nested.max()
    latstart = np.unravel_index(np.abs(lat1).argmin(),lat1.shape)
    latend= np.unravel_index(np.abs(lat2).argmin(),lat2.shape)

    HHGLatInBOO = (lawr.Lat_nested[:, :] - dwd.Lat_nested[latstart]) / (
            dwd.Lat_nested[latend] - dwd.Lat_nested[latstart]) * (latend[0] - latstart[0]) + latstart[0]

    lon1 = dwd.Lon_nested - lawr.Lon_nested.min()
    lon2 = dwd.Lon_nested - lawr.Lon_nested.max()
    lonstart = np.unravel_index(np.abs(lon1).argmin(), lon1.shape)
    lonend = np.unravel_index(np.abs(lon2).argmin(), lon2.shape)
    HHGLonInBOO = (lawr.Lon_nested[:, :] - dwd.Lon_nested[lonstart[0], lonstart[1]]) / (
        dwd.Lon_nested[lonend[0], lonend[1]] - dwd.Lon_nested[lonstart[0], lonstart[1]]) * (lonend[1] - lonstart[1]) + lonstart[1]

    if np.sum(boo_prog_data[boo_pixels]>lawr.rainThreshold):
        prog_data[hhg_pixels] = interp2d(boo_prog_data, HHGLatInBOO[hhg_pixels], HHGLonInBOO[
            hhg_pixels])        #prog_data[hhg_pixels] = griddata(boo.HHG_cart_points[boo_pixels.flatten()], boo_prog_data[boo_pixels].flatten(), nested_points[hhg_pixels.flatten()], method='cubic')
    return prog_data

def booDisplacement(boo, boo_prog_data, displacementX, displacementY):

    x = np.arange(boo_prog_data.shape[0]) - displacementX
    y = np.arange(boo_prog_data.shape[1]) - displacementY
    [Y, X] = np.meshgrid(y,x)
    boo_prog_data = interp2d(boo_prog_data, X, Y)
    return boo_prog_data

def leastsquarecorr(dataArea, corrArea):
    """
    Calculates the least squared difference between two input arrays.
    coraArea array moves over the dataArea array.

    Parameters
    ----------
        dataArea (numpy.ndarray): 2d array, the larger array (m x m)
        corrArea (numpy.ndarray): 2d array, the smaller array (n x n)

    Returns
    -------
    (numpy.ndarray):
        least squared differences of both arrays size should be (n x n)
    """
    c_d = np.sum((image.extract_patches_2d(dataArea, corrArea.shape) - corrArea) ** 2, axis=(1, 2)).reshape(np.array(dataArea.shape) - corrArea.shape + 1)
    return c_d

def twodgauss(x, y, sigma1, sigma2, rho, mu1, mu2):
    """
    Constructs and returns a bivariate normal distribution.

    This function creates a bivariate normal distribution with the specifics from the input.
    x and y create the grid for the distribution. They have to be created with np.meshgrid beforehand.
    For example with [y,x] = np.meshgrid(np.arange(-5,5),np.arange(-5,5). The 0 has to be in the middle of the created
    x,y arrays.

    Parameters
    ----------
        x : numpy.ndarray
            2d array containing x grid for the bivariate normal distribution
        y : numpy.ndarray
            2d array containing y grid for the bivariate normal distribution
        sigma1 : float
            standard deviation of variable 1
        sigma2 : float
            standard deviation of variable 2
        rho : float
            correlation between variable 1 and variable 2
        mu1 : float
            mean of variable 1
        mu2 : float
            mean of variable 2

    Returns
    -------
    numpy.ndarray
        bivariate normal distribution for the input parameters with the size of the input grids
    """
    return 1 / (2 * np.pi * sigma1 * sigma2 * np.sqrt(1 - rho ** 2)) * np.exp(-1 / (2 * (1 - rho ** 2)) * (
           (x - mu1) ** 2 / sigma1 ** 2 - (2 * rho * (x - mu1) * (y - mu2) / (sigma1 * sigma2)) + (
           y - mu2) ** 2 / sigma2 ** 2))

class Results:
    def __init__(self, hit,miss,f_alert,corr_zero,roc_hr,roc_far,year,mon,day,hour,minute,dwdGaussMeans,dwdCov,lawrGaussMeans,lawrCov,progStart,progEnd):
        self.hit = hit
        self.miss = miss
        self.f_alert = f_alert
        self.corr_zero = corr_zero
        self.roc_hr = roc_hr
        self.roc_far = roc_far
        self.year = year
        self.mon = mon
        self.day = day
        self.hour = hour
        self.minute = minute
        self.dwdGaussMeans = dwdGaussMeans
        self.dwdCov = dwdCov
        self.lawrGaussMeans = lawrGaussMeans
        self.lawrCov = lawrCov
        self.progStart = progStart
        self.progEnd = progEnd

def verification(lawr,dwd, year, mon, day, hour, minute,progTime):
    #function [BIAS,CSI,FAR,ORSS,PC,POD,hit,miss,f_alert,corr_zero]=verification(prog_data,real_data)
# %for documentation see master thesis: Niederschlags-Nowcasting fuer ein
# %hochaufgeloestes X-Band Regenradar from Timur Eckmann
    rain_thresholds=np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30])
    rain_threshold = 0.5
    prob_thresholds = np.arange(0,1.01,0.1)

    prog_data = lawr.probabilities
    real_data = lawr.nested_data[lawr.progStartIdx:lawr.progStartIdx+progTime,:,:]>rain_threshold
    prog_data[:, lawr.dist_nested >= np.max(lawr.r)] = 0
    real_data[:, lawr.dist_nested >= np.max(lawr.r)] = 0
    num_prob= len(prob_thresholds)
    time = prog_data.shape[0]

    hit = np.zeros([time, num_prob])
    miss = np.zeros([time, num_prob])
    f_alert = np.zeros([time, num_prob])
    corr_zero = np.zeros([time, num_prob])
    event = np.zeros([time, num_prob])
    nonevent= np.zeros([time, num_prob])
    total = np.zeros([time, num_prob])



    roc_far = np.zeros([time,len(prob_thresholds)])
    roc_hr = np.zeros([time,len(prob_thresholds)])

    for i, p in enumerate(prob_thresholds):
        p_dat = lawr.probabilities>p
        r_dat = real_data>rain_threshold
        for t in range(time):
            hit[t,i] = np.sum(r_dat[t,:,:] & p_dat[t,:,:])
            miss[t,i] = np.sum(r_dat[t,:,:] & ~p_dat[t,:,:])
            f_alert[t,i] = np.sum(~r_dat[t,:,:] & p_dat[t,:,:])
            corr_zero[t,i] = np.sum(~r_dat[t,:,:] & ~p_dat[t,:,:])

            event[t,i] = hit[t,i] + miss[t,i]
            nonevent[t,i] = f_alert[t,i] + corr_zero[t,i]
            total[t,i] = hit[t,i]+miss[t,i]+f_alert[t,i]+corr_zero[t,i]

            roc_hr[t,i] = hit[t,i]/event[t,i]
            roc_far[t,i] = f_alert[t,i]/nonevent[t,i]

    y_score_bin_mean = np.zeros([time,num_prob])
    empirical_prob_pos = np.copy(y_score_bin_mean)
    sample_num = np.copy(y_score_bin_mean)
    for t in range(time):
        y_score_bin_mean[t,:],empirical_prob_pos[t,:],sample_num[t,:],bin_centers = reliability_curve(real_data[t,:,:],prog_data[t,:,:])

    inverse_number_of_forecasts = 1/np.sum(lawr.dist_nested<lawr.r[-1])
    bs = inverse_number_of_forecasts*np.sum(np.sum((prog_data[:,:,:]-real_data[:,:,:])**2,axis=2),axis=1)

    roc_hr[:, 0] = 1
    roc_hr[:, -1] = 0
    roc_far[:, 0] = 1
    roc_far[:, -1] = 0
    result = Results(hit, miss, f_alert, corr_zero, roc_hr,roc_far,year, mon, day, hour, minute,
                     dwd.gaussMeans, dwd.covNormAngle, lawr.gaussMeans, lawr.covNormAngle, lawr.probabilities[0, :, :],
                     lawr.probabilities[-1, :, :])
    result.hit = hit
    result.miss = miss
    result.f_alert = f_alert
    result.corr_zero = corr_zero
    result.event = event
    result.nonevent = nonevent
    result.roc_hr = roc_hr
    result.roc_far = roc_far
    result.bs=bs
    result.y_score_bin_mean =y_score_bin_mean
    result.empirical_prob_pos = empirical_prob_pos
    result.sample_num = sample_num
    result.bin_centers= bin_centers
    result.realEnd = real_data[-1,:,:]
    result.realStart = real_data[0,:,:]
    return result

def reliability_curve(y_true, y_score, bins=11, normalize=False):
    """Compute reliability curve

    Reliability curves allow checking if the predicted probabilities of a
    binary classifier are well calibrated. This function returns two arrays
    which encode a mapping from predicted probability to empirical probability.
    For this, the predicted probabilities are partitioned into equally sized
    bins and the mean predicted probability and the mean empirical probabilties
    in the bins are computed. For perfectly calibrated predictions, both
    quantities whould be approximately equal (for sufficiently many test
    samples).

    Note: this implementation is restricted to binary classification.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels (0 or 1).

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class or confidence values. If normalize is False, y_score must be in
        the interval [0, 1]

    bins : int, optional, default=10
        The number of bins into which the y_scores are partitioned.
        Note: n_samples should be considerably larger than bins such that
              there is sufficient data in each bin to get a reliable estimate
              of the reliability

    normalize : bool, optional, default=False
        Whether y_score needs to be normalized into the bin [0, 1]. If True,
        the smallest value in y_score is mapped onto 0 and the largest one
        onto 1.


    Returns
    -------
    y_score_bin_mean : array, shape = [bins]
        The mean predicted y_score in the respective bins.

    empirical_prob_pos : array, shape = [bins]
        The empirical probability (frequency) of the positive class (+1) in the
        respective bins.


    References
    ----------
    .. [1] `Predicting Good Probabilities with Supervised Learning
            <http://machinelearning.wustl.edu/mlpapers/paper_files/icml2005_Niculescu-MizilC05.pdf>`_

    """
    if normalize:  # Normalize scores into bin [0, 1]
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())

    bin_width = 1.0 / bins
    bin_centers = np.linspace(0, 1.0, bins)

    y_score_bin_mean = np.empty(bins)
    empirical_prob_pos = np.empty(bins)
    sample_num = np.empty(bins)
    for i, threshold in enumerate(bin_centers):
        # determine all samples where y_score falls into the i-th bin
        bin_idx = np.logical_and(threshold - bin_width / 2 < y_score,
                                 y_score <= threshold + bin_width / 2)
        # Store mean y_score and mean empirical probability of positive class
        y_score_bin_mean[i] = y_score[bin_idx].mean()
        empirical_prob_pos[i] = y_true[bin_idx].mean()
        sample_num[i] = np.sum(bin_idx)
    return y_score_bin_mean, empirical_prob_pos, sample_num, bin_centers