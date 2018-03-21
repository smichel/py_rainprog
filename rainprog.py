import numpy as np
import netCDF4
import time
from datetime import datetime
# from matplotlib.mlab import griddata
import matplotlib
matplotlib.use('TkAgg')
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from createblob import createblob
from findmaxima import findmaxima
from leastsquarecorr import leastsquarecorr
from testmaxima import testmaxima
from testangles import testangles


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

#fp = 'E:/Rainprog/data/m4t_BKM_wrx00_l2_dbz_v00_20130511160000.nc'
fp = '/home/zmaw/u300675/pattern_data/m4t_HWT_wrx00_l2_dbz_v00_20130613030000.nc'
res = 200
smallVal = 2
rainThreshold = 0.1
distThreshold = 17000
prog = 10
trainTime = 8
numMaxes = 10
progTime = 20
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
#plt.imshow(nestedData[prog, :, :])
#plt.show()

maxima = np.empty([1, 3])
maxima[0, 0] = np.nanmax(R[0, :, :])
maxima[0, 1:3] = np.unravel_index(np.nanargmax(R[0, :, :]), R[0, :, :].shape)
status = np.ones([1])

startTime = datetime.now()
maxima, status = findmaxima(maxima, R[0, :, :], cRange, numMaxes, rainThreshold, distThreshold, dist, status)
maxima[:, 1:3] = maxima[:, 1:3] + cRange * 2

time_elapsed = datetime.now() - startTime
contours = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]

shiftXlist = []
shiftYlist = []
print(time_elapsed)
newMaxima = np.copy(maxima)


for t in range(prog):
    #print(t)
    #maxima, status = testmaxima(maxima, nestedData[t, :, :], rainThreshold, distThreshold, res, status)
    if len(maxima) < numMaxes:
        maxima[:, 1:3] = maxima[:, 1:3] - cRange * 2
        maxima, status = findmaxima(maxima, R[t, :, :], cRange, numMaxes, rainThreshold, distThreshold, dist, status)
        maxima[:, 1:3] = maxima[:, 1:3] + cRange * 2
        print('looked for new maxima')

    maxima, status = testmaxima(maxima, nestedData[t, :, :], rainThreshold, distThreshold, res, status)

    newMaxima = np.empty([len(maxima), 3])
    for q in range(len(maxima)):

        corrArea = nestedData[t, (int(maxima[q, 1]) - cRange):(int(maxima[q, 1]) + cRange),
                   (int(maxima[q, 2]) - cRange):(int(maxima[q, 2]) + cRange)]
        dataArea = nestedData[t+1, (int(maxima[q, 1]) - cRange * 2):(int(maxima[q, 1]) + cRange * 2),
                   (int(maxima[q, 2]) - cRange * 2):(int(maxima[q, 2]) + cRange * 2)]
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
    plt.pause(0.01)
    shiftX = newMaxima[:, 1] - maxima[:, 1]
    shiftY = newMaxima[:, 2] - maxima[:, 2]
    #angles = np.arctan2(shiftY, shiftX) * 180 / np.pi
    shiftX, shiftY = testangles(shiftX, shiftY, status, res)

    shiftXlist.append(shiftX)
    shiftYlist.append(shiftY)

    meanX[t] = np.mean(shiftX)
    meanY[t] = np.mean(shiftY)

    maxima = np.copy(newMaxima)
    status = np.zeros(len(maxima))

displacementX = np.nanmean(meanX[prog - trainTime:prog]) * res
displacementY = np.nanmean(meanY[prog - trainTime:prog]) * res
bins = np.linspace(-800/res,800/res, num=9)
shiftX = np.hstack(shiftXlist[prog - trainTime:prog])

histX, bin_edges = np.histogram(np.hstack(shiftXlist[prog - trainTime:prog]), bins)
histY, _ = np.histogram(np.hstack(shiftYlist[prog - trainTime:prog]), bins)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
new_bin_centres = np.linspace(bin_centres[0], bin_centres[-1], 200)
# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
p0 = [1., 0., 1.]

coeffX, var_matrix = curve_fit(gauss, bin_centres, histX, p0=p0, maxfev=2000)
coeffY, var_matrix = curve_fit(gauss, bin_centres, histY, p0=p0, maxfev=2000)
# Get the fitted curve
histX_fit = gauss(new_bin_centres, *coeffX)
histY_fit = gauss(new_bin_centres, *coeffY)

fig = plt.subplots()
plt.plot(bin_centres, histX, label='Data')
plt.plot(new_bin_centres, histX_fit, label='Fitted data')

# Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
print('Fitted mean = ', coeffX[1])
print('Fitted standard deviation = ', coeffX[2])

plt.show(block=False)

fig = plt.subplots()
plt.plot(bin_centres, histY, label='Data')
plt.plot(new_bin_centres, histY_fit, label='Fitted data')

# Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
print('Fitted mean = ', coeffY[1])
print('Fitted standard deviation = ', coeffY[2])

plt.show(block=False)


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
