import numpy as np
import numpy.ma as ma
import scipy.spatial.qhull as qhull
from scipy.interpolate import griddata, RegularGridInterpolator, interp1d
from scipy.ndimage import map_coordinates

from datetime import datetime
import h5py
import matplotlib.pyplot as plt
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

class totalField:

    def __init__(self, fields, rainThreshold, distThreshold, dist, numMaxes, nested_data, R, res, cRange, trainTime):
        self.activeFields = fields
        self.inactiveFields = []
        self.rainThreshold = rainThreshold  # minimal rainthreshold
        self.distThreshold = distThreshold  # distance threshold to radarboundary
        self.dist = dist  # global distancefield
        self.numMaxes = numMaxes  # number of maxima
        self.nested_data = nested_data
        self.R = R
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

        self.assign_ids()
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

    def return_fieldHistMeanAngle(self):
        fieldHistMeanAngle = []

        for field in self.activeFields:
            if field.lifeTime >= self.trainTime:
                fieldHistMeanAngle.extend(field.histMeanAngle)

        return np.asarray(fieldHistMeanAngle)

    def test_maxima(self, nestedData):
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

        for i in reversed(range(len(self.activeFields))):
            if i not in status:
                self.inactiveFields.append(self.activeFields[i])
                self.inactiveFields[-1].lifeTime = -1
                del self.activeFields[i]

    def prog_step(self,t):

        for field in self.activeFields:
            corrArea = self.nested_data[t, (int(field.maxima[0, 1]) - self.cRange):(int(field.maxima[0, 1]) + self.cRange),
                       (int(field.maxima[0, 2]) - self.cRange):(int(field.maxima[0, 2]) + self.cRange)]
            dataArea = self.nested_data[t + 1, (int(field.maxima[0, 1]) - self.cRange * 2):(int(field.maxima[0, 1]) + self.cRange * 2),
                       (int(field.maxima[0, 2]) - self.cRange * 2):(int(field.maxima[0, 2]) + self.cRange * 2)]
            # maybe consider using "from skimage.feature import match_template" template matching
            # http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_template.html
            c = leastsquarecorr(dataArea, corrArea)
            cIdx = np.unravel_index((np.nanargmin(c)), c.shape)
            field.shiftX = int(cIdx[0] - 0.5 * len(c))
            field.shiftY = int(cIdx[1] - 0.5 * len(c))
            field.norm = np.linalg.norm([field.shiftX, field.shiftY])
            field.angle = get_metangle(field.shiftX, field.shiftY)
            field.angle = field.angle.filled()
            field.add_norm(field.norm)
            field.add_angle(field.angle)
            field.add_maximum(np.copy(field.maxima))
            field.add_shift(field.shiftX, field.shiftY)
            field.maxima[0, 0] = self.nested_data[t, int(field.maxima[0, 1] + cIdx[0] - 0.5 * len(c)),
                                             int(field.maxima[0, 2] + cIdx[1] - 0.5 * len(c))]
            field.maxima[0, 1] = int(field.maxima[0, 1] + cIdx[0] - 0.5 * len(c))
            field.maxima[0, 2] = int(field.maxima[0, 2] + cIdx[1] - 0.5 * len(c))

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

    def test_angles(self):
        fieldNums = 0
        for field in reversed(self.activeFields):
            if field.lifeTime < self.trainTime:
                self.activeFields.remove(field)
            else:
                fieldNums = fieldNums + 1

        angleFilter = list(range(fieldNums))
        lengthFilter = list(range(fieldNums))

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
            status = status[lengths != 0]
            lengths = lengths[lengths != 0]

            shiftXex = shiftX[lengths <= 1000]
            shiftYex = shiftY[lengths <= 1000]
            lengthFilter.extend(status[lengths <= 1000])

            status = status[lengths <= 1000]

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
        aFilter = (np.full_like(angleCounts, self.trainTime) - angleCounts) > 4 # angleFilter
        lFilter = (np.full_like(lengthCounts, self.trainTime) - lengthCounts) > 1 # lengthFilter

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

def importance_sampling(nested_data, nested_dist, rMax, xy, yx, xSample, ySample, d_s, cRange):
    nested_data_ = ma.array(nested_data, mask=nested_dist >= rMax, dtype=float,fill_value=np.nan)
    xy = ma.array(xy, mask=nested_dist >= rMax, dtype=int,fill_value=-999999)
    yx = ma.array(yx, mask=nested_dist >= rMax, dtype=int,fill_value=-999999)
    prog_data_ = get_values(xSample, ySample, xy.flatten()[~xy.mask.flatten()], yx.flatten()[~yx.mask.flatten()].flatten(), nested_data)
    nested_data.squeeze()[~nested_data_.mask.squeeze()] = prog_data_
    prog_data = np.reshape(nested_data,[d_s+4*cRange,d_s+4*cRange])
    return prog_data

def create_sample(gaussMeans, covNormAngle, samples):
    return np.random.multivariate_normal(gaussMeans, covNormAngle, samples).T

def get_values(xSample, ySample, x, y, nested_data):  # nested_data should be 2d
    x = np.full((len(xSample), len(x)), x).T
    y = np.full((len(ySample), len(y)), y).T
    x_= x - xSample
    y_= y - ySample
    vals = np.nanmean(interp2d(nested_data, x_, y_),axis=1)
    return vals

def interp2d(nested_data, x, y):  # nested_data in 2d
    vals = nested_data[np.int_(x), np.int_(y)] * ((1 - np.mod(x, 1)) * (1 - np.mod(y, 1))) + \
        nested_data[np.int_(x), np.int_(y)+1] * (1- np.mod(x, 1)) * (np.mod(y, 1)) + \
        nested_data[np.int_(x)+1, np.int_(y)+1] * ((np.mod(x, 1)) * np.mod(y, 1)) + \
        nested_data[np.int_(x)+1, np.int_(y)] * ((np.mod(x, 1)) * (1 - np.mod(y, 1)))
    return vals

def findRadarSite(HHGlat, HHGlon, BOO):
    lat = np.abs(BOO.lat - HHGlat)
    lon = np.abs(BOO.lon - HHGlon)
    latIdx = np.where(lat == lat.min())
    lonIdx = np.where(lon == lon.min())
    return latIdx[1][0], lonIdx[0][0]

def getFiles(filelist, time):
    files = []
    for i, file in enumerate(filelist):
        if (np.abs(int(file[41:43])- time) <= 0):
            files.append(file)
    return files

def nesting(prog_data, nested_dist, nested_points, boo_prog_data, boo, rMax, rainthreshold, HHGlat, HHGlon):
    boo_pixels = ((boo.HHGdist >= rMax) & (boo.HHGdist <= nested_dist.max()))
    hhg_pixels = ((nested_dist >= rMax) & (nested_dist <= nested_dist.max()))
    lat1 = boo.lat - HHGlat.min()
    lat2 = boo.lat - HHGlat.max()
    latstart = lat1[lat1 < 0].argmax()
    latend= lat2[lat2 < 0].argmax()

    HHGLatInBOO = (HHGlat[:, :] - boo.lat[0, latstart]) / (
            boo.lat[0, latend] - boo.lat[0, latstart]) * (latend - latstart) + latstart

    lon1 = boo.lon - HHGlon.min()
    lon2 = boo.lon - HHGlon.max()
    lonstart = np.unravel_index(lon1[lon1 < 0].argmax(), boo.lat.shape)
    lonend = np.unravel_index(lon2[lon2 < 0].argmax(), boo.lat.shape)

    HHGLonInBOO = np.flipud(np.rot90((HHGlon[:, :] - boo.lon[lonstart]) / (
            boo.lon[lonend] - boo.lon[lonstart]) * (lonend[0] - lonstart[0]) + lonstart[0],1,(0,1)))


    if np.sum(boo_prog_data[boo_pixels]>rainthreshold):
        prog_data[hhg_pixels] = interp2d(boo_prog_data, HHGLonInBOO[hhg_pixels], HHGLatInBOO[hhg_pixels]) # new method, using the 2d interpolation method, is 10x faster than gridding
        #prog_data[hhg_pixels] = griddata(boo.HHG_cart_points[boo_pixels.flatten()], boo_prog_data[boo_pixels].flatten(), nested_points[hhg_pixels.flatten()], method='cubic')
    return  prog_data

def booDisplacement(boo, boo_prog_data, displacementX, displacementY):
    paddingNaNs = int(1500/boo.resolution)

    x = np.arange(paddingNaNs, boo.d_s + paddingNaNs) - displacementX/boo.resolution
    y = np.arange(paddingNaNs, boo.d_s + paddingNaNs) - displacementY/boo.resolution

    [Y, X] = np.meshgrid(y,x)
    # padding boo data with nans to prevent errors, this should equal a distance of 1500m with a resolution of 500m. This
    # is far over the possible maximum movespeed of clouds (1500m in 30s equals 180 km/h)

    boo_data = np.empty([boo.d_s + paddingNaNs*2, boo.d_s + paddingNaNs*2])
    boo_data.fill(np.nan)
    boo_data[paddingNaNs : boo.d_s+ paddingNaNs, paddingNaNs : boo.d_s + paddingNaNs] = boo_prog_data
    boo_prog_data = interp2d(boo_data, X, Y)
    return boo_prog_data

def leastsquarecorr(dataArea, corrArea):
        # %Calculates a leastsquare correlation between 2 matrices c and d

    cLen = len(corrArea)
    c_d = np.zeros([cLen, cLen])

    k = 0
    m = 0
    for i in range(cLen):
        for j in range(cLen):
            c_d[k, m] = np.sum(np.sum(np.square(dataArea[i:i + cLen, j:j + cLen] - corrArea)))
            m += 1
        m = 0
        k += 1

    return c_d

class DWDData:

    def read_dwd_file(self, filepath):

        '''Read in DWD radar data.

        Reads DWD radar data and saves data to object. Only attributes
        needed for my calculations are read in. If more information
        about the file and the attributes is wished, check out the
        `hdf5 <https://support.hdfgroup.org/HDF5/>`_-file with hdfview
        or hd5dump -H.

        Args:
            filename (str): Name of radar data file.

        '''
        with h5py.File(filepath, 'r') as boo:
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
            self.time = []

    def getGrid(self, booresolution):

        latDeg = 110540  # one degree equals 110540 m
        lonDeg = 113200  # one degree * cos(lat*pi/180) equals 113200 m
        aziCos = np.cos(np.radians(self.azi+90))
        aziSin = np.sin(np.radians(self.azi+90))
        xPolar = np.outer(self.r, aziCos)
        xPolar = np.reshape(xPolar, (len(self.azi) * len(self.r), 1))
        yPolar = np.outer(self.r, aziSin)
        yPolar = np.reshape(yPolar, (len(self.azi) * len(self.r), 1))
        points = np.concatenate((xPolar, yPolar), axis=1)

        self.resolution = booresolution
        self.xCar = np.arange(-60000, 60000 + 1, self.resolution).squeeze()
        self.yCar = np.arange(-110000, 10000 + 1, self.resolution).squeeze()
        self.d_s = len(self.xCar)

        [self.XCar, self.YCar] = np.meshgrid(self.xCar, self.yCar)
        self.dist = np.sqrt(np.square(self.XCar) + np.square(self.YCar))
        self.lat = self.sitecoords[0] + self.XCar / latDeg
        self.lon = self.sitecoords[1] + self.YCar / (lonDeg * (np.cos(self.lat * np.pi / 180)))
        target = np.zeros([self.XCar.shape[0] * self.XCar.shape[1], 2])
        target[:, 0] = self.XCar.flatten()
        target[:, 1] = self.YCar.flatten()

        self.cart_points = np.concatenate((np.reshape(self.XCar, (self.d_s * self.d_s, 1)),
                                          np.reshape(self.YCar, (self.d_s * self.d_s, 1))), axis=1)


        self.vtx, self.wts = interp_weights(points, target)

    def gridding(self, vtx, wts, d_s):

        rPolar = z2rainrate(self.refl).T
        rPolar = np.reshape(rPolar, (len(self.azi) * len(self.r), 1)).squeeze()

        self.R = np.reshape(interpolate(rPolar.flatten(), vtx, wts), (d_s, d_s))

    def addTimestep(self, R):
        self.R = np.dstack((self.R, R))

    def timeInterpolation(self,timeSteps):
        #https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.interpolate.RegularGridInterpolator.html
        x = np.arange(self.d_s)
        y = np.arange(self.d_s)
        z = np.arange(self.R.shape[2])
        interpolating_function = RegularGridInterpolator((x,y,z),self.R)
        z_ = np.linspace(0,z[-1],timeSteps)
        pts = np.array(np.meshgrid(x, y, z_)).reshape(3,self.d_s*self.d_s*timeSteps).transpose()
        self.R = interpolating_function(pts).reshape(self.d_s,self.d_s,timeSteps)