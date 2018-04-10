import numpy as np
import netCDF4
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from createblob import createblob
from findmaxima import findmaxima
from leastsquarecorr import leastsquarecorr
from testangles import testangles
from init import Square, totalField


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


# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/2.*sigma**2)




#fp = 'C:/Rainprog/m4t_BKM_wrx00_l2_dbz_v00_20130511160000.nc'
fp = '/home/zmaw/u300675/pattern_data/m4t_BKM_wrx00_l2_dbz_v00_20130511160000.nc'
res = 200
smallVal = 2
rainThreshold = 0.1
distThreshold = 17000
prog = 10
global trainTime
trainTime = 5
numMaxes = 20
progTime = 2
useRealData = 1
timeSteps = prog + progTime

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
meanX = np.zeros([prog])
meanY = np.zeros([prog])

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
if useRealData:
    for t in range(timeSteps):
        rPolarT = rPolar[t, :, :].T
        rPolarT = np.reshape(rPolarT, (333*360, 1)).squeeze()
        R[t, :, :] = griddata(points, rPolarT, (XCar, YCar), method='nearest')
        R[t, (dist > 20000)] = 0
        nestedData[t, 2 * cRange: 2 * cRange + d_s, 2 * cRange: 2 * cRange + d_s] = R[t, :, :]
else:
    R = createblob(d_s, res, timeSteps)
    R[:, (dist > 20000)] = 0
    nestedData[:, 2 * cRange: 2 * cRange + d_s, 2 * cRange: 2 * cRange + d_s] = R

#R = griddata(xPolar, yPolar, rPolar, xCar, yCar, interp = 'linear')

time_elapsed = datetime.now() - startTime
print(time_elapsed)
nestedData = np.nan_to_num(nestedData)
R = np.nan_to_num(R)


startTime = datetime.now()
allFields = totalField(findmaxima([], R[0, :, :], cRange, numMaxes, rainThreshold, distThreshold, dist), rainThreshold, distThreshold, dist, numMaxes, res, cRange)
for field in allFields.activeFields:
    if field.status:
        field.maxima[:, 1:3] = field.maxima[:, 1:3] + cRange * 2


time_elapsed = datetime.now() - startTime
contours = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]

print(time_elapsed)

for t in range(prog):
    #print(t)
    #maxima, status = testmaxima(maxima, nestedData[t, :, :], rainThreshold, distThreshold, res, status)
    if len(allFields.activeFields) < numMaxes:
        for field in allFields.activeFields:
            field.maxima[0, 1:3] = field.maxima[0, 1:3] - cRange * 2
        allFields.activeFields = findmaxima(allFields.activeFields, R[t, :, :], cRange, numMaxes, rainThreshold, distThreshold, dist)
        for field in allFields.activeFields:
            field.maxima[0, 1:3] = field.maxima[0, 1:3] + cRange * 2

        print('looked for new maxima')

    #fields = testmaxima(fields, nestedData[t, :, :], rainThreshold, distThreshold, res, cRange)
    allFields.testmaxima(nestedData[t, :, :])
    for field in allFields.activeFields:

        corrArea = nestedData[t, (int(field.maxima[0, 1]) - cRange):(int(field.maxima[0, 1]) + cRange),
                   (int(field.maxima[0, 2]) - cRange):(int(field.maxima[0, 2]) + cRange)]
        dataArea = nestedData[t+1, (int(field.maxima[0, 1]) - cRange * 2):(int(field.maxima[0, 1]) + cRange * 2),
                   (int(field.maxima[0, 2]) - cRange * 2):(int(field.maxima[0, 2]) + cRange * 2)]
        c = leastsquarecorr(dataArea, corrArea)
        cIdx = np.unravel_index((np.nanargmin(c)), c.shape)

        field.shiftX = int(cIdx[0] - 0.5 * len(c))
        field.shiftY = int(cIdx[1] - 0.5 * len(c))
        field.add_maximum(np.copy(field.maxima))
        field.maxima[0, 0] = nestedData[t, int(field.maxima[0, 1] + cIdx[0] - 0.5 * len(c)),
                                        int(field.maxima[0, 2] + cIdx[1] - 0.5 * len(c))]
        field.maxima[0, 1] = int(field.maxima[0, 1] + cIdx[0] - 0.5 * len(c))
        field.maxima[0, 2] = int(field.maxima[0, 2] + cIdx[1] - 0.5 * len(c))

    #angles = np.arctan2(shiftY, shiftX) * 180 / np.pi
    shiftX, shiftY, status = testangles(allFields.activeFields, res)

    meanX[t] = np.mean(shiftX)
    meanY[t] = np.mean(shiftY)

    #allFields = totalField(fields, rainThreshold, distThreshold, dist, numMaxes)

    if t == 0:
        plt.figure(figsize=(8,8))
        im = plt.imshow(nestedData[t, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
        plt.show(block=False)
        o, = plt.plot(*np.transpose(allFields.return_maxima(0)[:, 2:0:-1]), 'ko')
        n, = plt.plot(*np.transpose(allFields.return_maxima(-1)[:, 2:0:-1]), 'wo')
        s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter())
        s.set_ticks(contours)
        #s.set_ticklabels(contourLabels)
    else:
        im.set_data(nestedData[t, :, :])
        o.set_data(*np.transpose(allFields.return_maxima(0)[:, 2:0:-1]))
        n.set_data(*np.transpose(allFields.return_maxima(-1)[:, 2:0:-1]))
    plt.pause(0.01)
    allFields.update_fields()

displacementX = np.nanmean(meanX[prog - trainTime:prog]) * res
displacementY = np.nanmean(meanY[prog - trainTime:prog]) * res

progData = np.zeros([progTime, d_s, d_s])
points = np.concatenate((np.reshape(XCar, (d_s * d_s, 1)), np.reshape(YCar, (d_s * d_s, 1))), axis = 1)

for t in range(progTime):
    progData[t, :, :] = griddata(points, nestedData[prog, 2 * cRange: 2 * cRange + d_s, 2 * cRange: 2 * cRange + d_s].flatten(),
                                 (XCar - displacementY * t, YCar - displacementX * t), method='nearest')
    if t == 0:
        plt.figure(figsize=(8,8))
        imP = plt.imshow(progData[t, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
        imR = plt.contour(nestedData[prog + t, 2 * cRange: 2 * cRange + d_s, 2 * cRange: 2 * cRange + d_s],
                          contours, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
        plt.show(block=False)
        sa = plt.colorbar(imP, format=matplotlib.ticker.ScalarFormatter())
        sa.set_ticks(contours)

    else:
        imP.set_data(progData[t, :, :])
        for tp in imR.collections:
            tp.remove()
        imR = plt.contour(nestedData[prog + t, 2 * cRange: 2 * cRange + d_s, 2 * cRange: 2 * cRange + d_s], contours,
                          norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
        plt.pause(0.1)

#fig, ax = plt.subplots()
#ax.bar(bin_edges[:-1]-0.1, histX, color='b', width = 0.2)
#ax.bar(bin_edges[:-1]+0.1, histY, color='r', width = 0.2)
#plt.show(block=False)
