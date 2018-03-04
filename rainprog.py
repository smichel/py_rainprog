import numpy as np
import netCDF4
import time
from datetime import datetime
# from matplotlib.mlab import griddata
import matplotlib
matplotlib.use('TkAgg')
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from findmaxima import findmaxima
from leastsquarecorr import leastsquarecorr
from testmaxima import testmaxima
from testangles import testangles

def z2rainrate(z):# Conversion between reflectivity and rainrate, a and b are parameters of the function
    a = np.full_like(z, 77, dtype=np.double)
    b = np.full_like(z, 1.9, dtype=np.double)
    cond1 = z <= 44.0
    a[cond1] = 200
    b[cond1] = 1.6
    cond2 = z <= 36.5
    a[cond2] = 320
    b[cond2] = 1.4
    return ((10 ** (z / 10)) / a) ** (1. / b)

fp = 'E:/Rainprog/data/m4t_BKM_wrx00_l2_dbz_v00_20130511160000.nc'
#fp = '/Users/u300675/m4t_BKM_wrx00_l2_dbz_v00_20130511160000.nc'
res = 200
timeSteps = 30
smallVal = 2
rainThreshold = 0.1
distThreshold = 17000
prog = 10
trainTime = 8
numMaxes = 10
progTime = 15

nc = netCDF4.Dataset(fp)
data = nc.variables['dbz_ac1'][:][:][:]
z = data
azi = nc.variables['azi'][:]
dist = nc.variables['range'][:]

aziCos = np.cos(np.radians(azi))
aziSin = np.sin(np.radians(azi))
xPolar = np.outer(dist, aziCos)
xPolar = np.reshape(xPolar, (333*360, 1))
yPolar = np.outer(dist, aziSin)
yPolar = np.reshape(yPolar, (333*360, 1))
points = np.concatenate((xPolar, yPolar), axis = 1)
meanX = np.zeros([timeSteps - 1])
meanY = np.zeros([timeSteps - 1])

xCar = np.arange(-20000, 20000+1, res).squeeze()
yCar = np.arange(-20000, 20000+1, res).squeeze()

[XCar, YCar] = np.meshgrid(xCar, yCar)

dist = np.sqrt(np.square(xCar)+np.square(YCar))

cRange = int((len(XCar) - 1) / 16)
d_s = len(XCar)

R = np.empty([timeSteps,d_s,d_s])

rPolar = z2rainrate(z)

nestedData = np.zeros([timeSteps, d_s + 4 * cRange, d_s + 4 * cRange])
startTime = datetime.now()
for t in range(timeSteps):
    rPolarT = rPolar[t, :, :].T
    rPolarT = np.reshape(rPolarT, (333*360, 1)).squeeze()
    R[t, :, :] = griddata(points, rPolarT, (XCar, YCar), method='nearest')
    R[t, (dist > 20000)] = 0
    nestedData[t, 2 * cRange: 2 * cRange + d_s, 2 * cRange: 2 * cRange + d_s] = R[t, :, :]
#R = griddata(xPolar, yPolar, rPolar, xCar, yCar, interp = 'linear')

time_elapsed = datetime.now() - startTime
print(time_elapsed)
nestedData = np.nan_to_num(nestedData)
R = np.nan_to_num(R)


maxima = np.empty([1, 3])
maxima[0, 0] = np.nanmax(R[0, :, :])
maxima[0, 1:3] = np.unravel_index(np.nanargmax(R[0, :, :]), R[0, :, :].shape)
status = np.ones([1])

startTime = datetime.now()
maxima, status = findmaxima(maxima, R[0, :, :], cRange, numMaxes, rainThreshold, distThreshold, dist, status)
maxima[:, 1:3] = maxima[:, 1:3] + cRange * 2

time_elapsed = datetime.now() - startTime
contours = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]

print(time_elapsed)
newMaxima = np.copy(maxima)


for t in range(timeSteps-1):
    print(t)
    maxima, status = testmaxima(maxima, nestedData[t, :, :], rainThreshold, distThreshold, res, status)
    if len(maxima) < numMaxes:
        maxima[:, 1:3] = maxima[:, 1:3] - cRange * 2
        maxima, status = findmaxima(maxima, R[t, :, :], cRange, numMaxes, rainThreshold, distThreshold, dist, status)
        maxima[:, 1:3] = maxima[:, 1:3] + cRange * 2
        print('looked for new maxima')

    newMaxima = np.empty([len(maxima), 3])
    for q in range(len(maxima)):

        corrArea = nestedData[t, (int(maxima[q, 1]) - cRange):(int(maxima[q, 1]) + cRange), (int(maxima[q, 2]) - cRange):(int(maxima[q, 2]) + cRange)]
        dataArea = nestedData[t+1, (int(maxima[q, 1]) - cRange * 2):(int(maxima[q, 1]) + cRange * 2), (int(maxima[q, 2]) - cRange * 2):(int(maxima[q, 2]) + cRange * 2)]
        c = leastsquarecorr(dataArea, corrArea)
        cIdx = np.unravel_index((np.nanargmin(c)), c.shape)

        newMaxima[q, 0] = nestedData[t, int(maxima[q, 1] + cIdx[0] - 0.5 * len(c)), int(maxima[q, 2] + cIdx[1] - 0.5 * len(c))]
        newMaxima[q, 1] = (int(maxima[q, 1]) + cIdx[0] - 0.5 * len(c))
        newMaxima[q, 2] = (int(maxima[q, 2]) + cIdx[1] - 0.5 * len(c))

    if t == 0:
        plt.figure(figsize=(8,8))
        im = plt.imshow(nestedData[t, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
        plt.show(block=False)
        o, = plt.plot(*np.transpose(newMaxima[:, 2:0:-1]), 'ko')
        n, = plt.plot(*np.transpose(maxima[:, 2:0:-1]), 'wo')
        s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter())
        s.set_ticks(contours)
        #s.set_ticklabels(contourLabels)
    else:
        im.set_data(nestedData[t, :, :])
        o.set_data(*np.transpose(newMaxima[:, 2:0:-1]))
        n.set_data(*np.transpose(maxima[:, 2:0:-1]))

    shiftX = newMaxima[:, 1] - maxima[:, 1]
    shiftY = newMaxima[:, 2] - maxima[:, 2]
    #angles = np.arctan2(shiftY, shiftX) * 180 / np.pi
    shiftX, shiftY = testangles(shiftX, shiftY, status)
    meanX[t] = np.mean(shiftX)
    meanY[t] = np.mean(shiftY)

    plt.pause(0.01)
    maxima = np.copy(newMaxima)
    status = np.zeros(len(maxima))

displacementX = np.nanmean(meanX[prog - trainTime:prog]) * res
displacementY = np.nanmean(meanY[prog - trainTime:prog]) * res

progData = np.zeros([progTime, d_s, d_s])
points = np.concatenate((np.reshape(XCar, (d_s * d_s, 1)), np.reshape(YCar, (d_s * d_s, 1))), axis = 1)

for t in range(progTime):
    progData[t, :, :] = griddata(points, nestedData[prog, 2 * cRange: 2 * cRange + d_s, 2 * cRange: 2 * cRange + d_s].flatten(), (XCar - displacementX * t, YCar - displacementY * t), method='nearest')
    if t == 0:
        plt.figure(figsize=(8,8))
        imP = plt.imshow(progData[t, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
        imR = plt.contour(nestedData[prog + t, 2 * cRange: 2 * cRange + d_s, 2 * cRange: 2 * cRange + d_s], contours, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
        plt.show(block=False)
        sa = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter())
        sa.set_ticks(contours)

    else:
        imP.set_data(progData[t, :, :])
        for tp in imR.collections:
            tp.remove()
        imR = plt.contour(nestedData[prog + t, 2 * cRange: 2 * cRange + d_s, 2 * cRange: 2 * cRange + d_s], contours, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
        plt.pause(0.1)
