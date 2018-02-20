import numpy as np
import netCDF4
import math
from datetime import datetime
# from matplotlib.mlab import griddata
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from findmaxima import findmaxima

fp = 'E:/Rainprog/data/m4t_BKM_wrx00_l2_dbz_v00_20130511160000.nc'
res = 100
timeSteps = 2
smallVal = 2
rainThreshold = 0.1
time = 1
prog = 10
uk = 5
numMaxes = 8






nc = netCDF4.Dataset(fp)
data = nc.variables['dbz_ac1'][:][:][:]
z = data
azi = nc.variables['azi'][:]
dist = nc.variables['range'][:]

aziCos = np.cos(np.radians(azi))
aziSin = np.sin(np.radians(azi))
startTime = datetime.now()
xPolar = np.outer(dist, aziCos)# * math.pi / 180
xPolar = np.reshape(xPolar, (333*360, 1))
yPolar = np.outer(dist, aziSin)# * math.pi / 180
yPolar = np.reshape(yPolar, (333*360, 1))
points = np.concatenate((xPolar, yPolar),axis=1)
print(max(xPolar))
print(max(yPolar))
time_elapsed = datetime.now() - startTime
print(time_elapsed)

xCar = np.arange(-20000, 20000+1, res).squeeze()
yCar = np.arange(-20000, 20000+1, res).squeeze()

[XCar, YCar] = np.meshgrid(xCar, yCar)

cRange = math.floor((len(XCar) - 1) / 12)
d_s = len(XCar)



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

R = np.empty([timeSteps,d_s,d_s])

rPolar = z2rainrate(z)

nestedData = np.zeros([timeSteps, d_s + 4 * cRange, d_s + 4 * cRange])
startTime = datetime.now()
for t in range(timeSteps):
    rPolarT = rPolar[t, :, :].T
    rPolarT = np.reshape(rPolarT, (333*360, 1)).squeeze()
    R[t, :, :] = griddata(points, rPolarT, (XCar, YCar), method='linear')
    nestedData[t, 2 * cRange: 2 * cRange + d_s, 2 * cRange: 2 * cRange + d_s] = R[t, :, :]
#R = griddata(xPolar, yPolar, rPolar, xCar, yCar, interp = 'linear')

time_elapsed = datetime.now() - startTime
print(time_elapsed)
nestedData = np.nan_to_num(nestedData)
R = np.nan_to_num(R)
plt.imshow(nestedData[0, :, :])
plt.show()
maxima = np.empty([1, 3])
maxima[0, 0] = np.nanmax(R[0, :, :])
maximaIdx = np.unravel_index(np.nanargmax(R[0, :, :]), R[0, :, :].shape)
maxima[0, 1] = maximaIdx[0]
maxima[0, 2] = maximaIdx[1]

startTime = datetime.now()
maxima = findmaxima(maxima, nestedData[0, :, :], cRange, numMaxes, rainThreshold)
time_elapsed = datetime.now() - startTime

print(time_elapsed)







for t in range(timeSteps):
    plt.imshow(nestedData[t, :, :])
    plt.show()

