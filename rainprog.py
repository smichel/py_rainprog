import numpy as np
import netCDF4
import os
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from createblob import createblob
from findmaxima import findmaxima
from leastsquarecorr import leastsquarecorr
from init import Square, totalField, get_metangle, interp_weights, interpolate, importance_sampling, DWDData, z2rainrate, findRadarSite, getFiles


# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/2.*sigma**2)


#fp = 'E:/Rainprog/data/m4t_BKM_wrx00_l2_dbz_v00_20130511160000.nc'
#fp = '/home/zmaw/u300675/pattern_data/m4t_BKM_wrx00_l2_dbz_v00_20130511160000.nc'
rTime = 14
fp = '/scratch/local1/HHG/2016/m4t_HHG_wrx00_l2_dbz_v00_20160607'+ str(rTime+1) + '0000.nc'
directoryPath = '/scratch/local1/BOO/2016/06/07/'
#fp = '/home/zmaw/u300675/pattern_data/m4t_BKM_wrx00_l2_dbz_v00_20130426120000.nc' difficult field to predict

fp_boo = '/scratch/local1/BOO/2016/06/07/ras07-pcpng01_sweeph5allm_any_00-2016060714003300-boo-10132-hd5'
booFileList = sorted(os.listdir(directoryPath))
selectedFiles = getFiles(booFileList, rTime)

res = 100
booResolution = 500
smallVal = 2
rainThreshold = 0.1
distThreshold = 19500
prog = 60
trainTime = 8
numMaxes = 20
progTime = 50
useRealData = 1
prognosis = 1
statistics = 1
livePlot = 1
samples = 12
timeSteps = prog + progTime
contours = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]

nc = netCDF4.Dataset(fp)
data = nc.variables['dbz_ac1'][:][:][:]
z = data
azi = nc.variables['azi'][:]
r = nc.variables['range'][:]
lat = 9.973997  # location of the hamburg radar
lon = 53.56833
zsl = 100  # altitude of the hamburg radar
latDeg = 110540  # one degree equals 110540 m
lonDeg = 113200  # one degree * cos(lat*pi/180) equals 113200 m
aziCos = np.cos(np.radians(azi))
aziSin = np.sin(np.radians(azi))
xPolar = np.outer(r, aziCos)
xPolar = np.reshape(xPolar, (333*360, 1))
yPolar = np.outer(r, aziSin)
yPolar = np.reshape(yPolar, (333*360, 1))
points = np.concatenate((xPolar, yPolar), axis = 1)

xCar = np.arange(-20000, 20000+1, res).squeeze()
yCar = np.arange(-20000, 20000+1, res).squeeze()

[XCar, YCar] = np.meshgrid(xCar, yCar)
Lat = lat + XCar / latDeg
Lon = lon + YCar / (lonDeg * (np.cos(Lat * np.pi / 180)))
dist = np.sqrt(np.square(xCar)+np.square(YCar))

target = np.zeros([XCar.shape[0]*XCar.shape[1],2])
target[:, 0] = XCar.flatten()
target[:, 1] = YCar.flatten()

cRange = int(800/res) # 800m equals an windspeed of aprox. 100km/h and is set as the upper boundary for a possible cloud movement
d_s = len(XCar)

R = np.empty([timeSteps,d_s,d_s])

rPolar = z2rainrate(z)

nestedData = np.zeros([timeSteps, d_s + 4 * cRange, d_s + 4 * cRange])
vtx, wts = interp_weights(points, target)
if useRealData:
    for t in range(timeSteps):
        rPolarT = rPolar[t, :, :].T
        rPolarT = np.reshape(rPolarT, (333*360, 1)).squeeze()
        R[t, :, :] = np.reshape(interpolate(rPolarT.flatten(), vtx, wts), (d_s, d_s))
        R[t, (dist >= np.max(r))] = 0
        nestedData[t, 2 * cRange: 2 * cRange + d_s, 2 * cRange: 2 * cRange + d_s] = R[t, :, :]
else:
    R = createblob(d_s, res, timeSteps)
    R[:, (dist > 20000)] = 0
    nestedData[:, 2 * cRange: 2 * cRange + d_s, 2 * cRange: 2 * cRange + d_s] = R

startTime = datetime.now()
boo=DWDData()
boo.read_dwd_file(directoryPath + selectedFiles[0])
selectedFiles.pop(0)
boo.getGrid(booResolution)
boo.gridding(boo.vtx, boo.wts, boo.d_s)


for i, file in enumerate(selectedFiles):
    buf = DWDData()
    buf.read_dwd_file(directoryPath + selectedFiles[i])
    buf.gridding(boo.vtx, boo.wts, boo.d_s)
    boo.addTimestep(buf.R)


boo.R = np.swapaxes(boo.R, 0, 2)
HHGposition = findRadarSite(lat, lon, boo)
boo.R[:, (boo.dist > boo.r.max())] = 0
time_elapsed = datetime.now() - startTime
print(time_elapsed)
fig,ax = plt.subplots(figsize=(8,8))

for i in range(boo.R.shape[0]):
    if (i == 0):
        im = plt.imshow(boo.R[i, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
        plt.gca().invert_yaxis()
        s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter())
        s.set_ticks(contours)
        radarCircle = mpatches.Circle((HHGposition[0], HHGposition[1]), 20000 / 500, color='w', linewidth=1, fill=0)
        ax.add_patch(radarCircle)
        plt.show(block=False)
    plt.pause(0.1)
    im.set_data(boo.R[i, :, :])

nestedData = np.nan_to_num(nestedData)
R = np.nan_to_num(R)


startTime = datetime.now()
allFields = totalField(findmaxima([], R[0, :, :], cRange, numMaxes, rainThreshold, distThreshold, dist), rainThreshold, distThreshold, dist, numMaxes, res, cRange, trainTime)
for field in allFields.activeFields:
    if field.status:
        field.maxima[:, 1:3] = field.maxima[:, 1:3] + cRange * 2




for t in range(prog):
    #print(t)
    #maxima, status = testmaxima(maxima, nestedData[t, :, :], rainThreshold, distThreshold, res, status)
    if len(allFields.activeFields) < numMaxes:
        for field in allFields.activeFields:
            field.maxima[0, 1:3] = field.maxima[0, 1:3] - cRange * 2
        allFields.activeFields = findmaxima(allFields.activeFields, R[t, :, :], cRange, numMaxes, rainThreshold, distThreshold, dist)
        allFields.assign_ids()
        for field in allFields.activeFields:
            field.maxima[0, 1:3] = field.maxima[0, 1:3] + cRange * 2

        print('looked for new maxima')

    #fields = testmaxima(fields, nestedData[t, :, :], rainThreshold, distThreshold, res, cRange)
    allFields.test_maxima(nestedData[t, :, :])
    for field in allFields.activeFields:

        corrArea = nestedData[t, (int(field.maxima[0, 1]) - cRange):(int(field.maxima[0, 1]) + cRange),
                   (int(field.maxima[0, 2]) - cRange):(int(field.maxima[0, 2]) + cRange)]
        dataArea = nestedData[t+1, (int(field.maxima[0, 1]) - cRange * 2):(int(field.maxima[0, 1]) + cRange * 2),
                   (int(field.maxima[0, 2]) - cRange * 2):(int(field.maxima[0, 2]) + cRange * 2)]
        c = leastsquarecorr(dataArea, corrArea)
        cIdx = np.unravel_index((np.nanargmin(c)), c.shape)
        # maybe consider using "from skimage.feature import match_template" template matching
        # http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_template.html
        field.shiftX = int(cIdx[0] - 0.5 * len(c))
        field.shiftY = int(cIdx[1] - 0.5 * len(c))
        field.norm = np.linalg.norm([field.shiftX, field.shiftY])
        field.angle = get_metangle(field.shiftX, field.shiftY)
        field.angle = field.angle.filled()
        field.add_norm(field.norm)
        field.add_angle(field.angle)
        field.add_maximum(np.copy(field.maxima))
        field.add_shift(field.shiftX, field.shiftY)
        field.maxima[0, 0] = nestedData[t, int(field.maxima[0, 1] + cIdx[0] - 0.5 * len(c)),
                                        int(field.maxima[0, 2] + cIdx[1] - 0.5 * len(c))]
        field.maxima[0, 1] = int(field.maxima[0, 1] + cIdx[0] - 0.5 * len(c))
        field.maxima[0, 2] = int(field.maxima[0, 2] + cIdx[1] - 0.5 * len(c))

    if livePlot:
        if t == 0:
            plt.figure(figsize=(8, 8))
            im = plt.imshow(nestedData[t, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
            plt.gca().invert_yaxis()
            plt.gca().invert_xaxis()
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




col = np.concatenate([np.zeros([1,np.max(allFields.activeIds)]), np.random.rand(2,np.max(allFields.activeIds))])

if statistics:
    plt.figure(figsize=(8, 8))
    for i, field in enumerate(allFields.activeFields):
        for t in field.histMaxima:
            plt.plot(*np.transpose(t[0][2:0:-1]), color=col[:,field.id-1], marker='o')
    for i, field in enumerate(allFields.inactiveFields):
        for t in field.histMaxima:
            plt.plot(*np.transpose(t[0][2:0:-1]), color=(1, 0, 0), marker='x')
    plt.show(block=False)

allFields.test_angles()

if statistics:
    plt.figure(figsize=(8, 8))
    for i, field in enumerate(allFields.activeFields):
        for t in field.histMaxima:
            plt.plot(*np.transpose(t[0][2:0:-1]), color=col[:,field.id-1], marker='o')
        for i, field in enumerate(allFields.inactiveFields):
            for t in field.histMaxima:
                plt.plot(*np.transpose(t[0][2:0:-1]), color=(1, 0, 0), marker='x')
    plt.show(block=False)


    plt.figure(figsize=(8, 8))
    for q, field in enumerate(allFields.activeFields):
        if field.lifeTime >= trainTime:
            for i, t in enumerate(field.histNorm):
                plt.plot(i, t, color='black', marker='o')
                #plt.plot(i, t - field.histStdNorm[i], color='gray', marker='o')
                #plt.plot(i, t + field.histStdNorm[i], color='gray', marker='o')

            plt.title('HistNorm')

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    for q, field in enumerate(allFields.activeFields):
        if field.lifeTime >= trainTime:
            for i, t in enumerate(field.histAngle):
                plt.polar(t * np.pi / 180, allFields.activeFields[q].histNorm[i], color='k', marker='o',
                          alpha=0.1)

            plt.title('HistAngle')

    plt.show(block=False)

    plt.figure(figsize=(8, 8))
    for q, field in enumerate(allFields.activeFields):
        if field.lifeTime >= trainTime:
            for i, t in enumerate(field.histX):
                plt.plot(i, t, alpha = 0.1, color='black', marker='o')
                #plt.plot(i, t - field.histStdNorm[i], color='gray', marker='o')
                #plt.plot(i, t + field.histStdNorm[i], color='gray', marker='o')

            plt.title('HistX')

    plt.figure(figsize=(8, 8))
    for q, field in enumerate(allFields.activeFields):
        if field.lifeTime >= trainTime:
            for i, t in enumerate(field.histY):
                plt.plot(i, t, alpha = 0.1, color='navy', marker='o')
                #plt.plot(i, t - field.histStdAngle[i], color='blue', marker='o')
                #plt.plot(i, t + field.histStdAngle[i], color='blue', marker='o')

            plt.title('HistY')

    plt.show(block=False)

allFieldsMeanX = np.nanmean(allFields.return_fieldMeanX())
allFieldsMeanY = np.nanmean(allFields.return_fieldMeanY())
allFieldsStdX = np.nanstd(allFields.return_fieldMeanX())
allFieldsStdY = np.nanstd(allFields.return_fieldMeanY())

allFieldsNorm = allFields.return_fieldHistMeanNorm()
allFieldsAngle = allFields.return_fieldHistMeanAngle()

allFieldsMeanNorm = np.linalg.norm([allFieldsMeanX, allFieldsMeanY])
allFieldsMeanAngle = get_metangle(allFieldsMeanX, allFieldsMeanY)

allFieldsStdNorm = np.linalg.norm([allFieldsStdX, allFieldsStdY])
allFieldsStdAngle = get_metangle(allFieldsStdX, allFieldsStdY)

displacementX = np.nanmean(allFields.return_fieldHistX())*res
displacementY = np.nanmean(allFields.return_fieldHistY())*res

covNormAngle = np.cov(allFieldsNorm, np.sin(allFieldsAngle*allFieldsNorm))
gaussMeans = [allFieldsMeanX, allFieldsMeanY]
#gaussMeans = [allFieldsMeanNorm, np.sin(allFieldsMeanAngle*allFieldsMeanNorm)]
#use allFieldsMeanNorm & np.sin(allFieldsMeanAngle*allFieldsMeanNorm for means
#use covNormAngle for covariance matrix
#x,y = np.random.multivariate_normal([allFieldsMeanNorm, np.sin(allFieldsMeanAngle*allFieldsMeanNorm)], np.cov(allFieldsNorm, np.sin(allFieldsAngle*allFieldsNorm)), 32).T

#print(allFieldsMeanX)
#print(allFieldsMeanY)
#print(allFieldsStdX)
#print(allFieldsStdY)

prog_data = np.zeros([progTime, d_s + 4 * cRange, d_s + 4 * cRange])
x = np.arange(2 * cRange, 2 * cRange + d_s)
y = np.arange(2 * cRange, 2 * cRange + d_s)
yx, xy = np.meshgrid(x, y)

if prognosis:
    for t in range(progTime):
        #progData[t, :, :] = griddata(points, nestedData[prog, 2 * cRange: 2 * cRange + d_s, 2 * cRange: 2 * cRange + d_s].flatten(),
                                     #(XCar - displacementY * t, YCar - displacementX * t), method='nearest')
        if t == 0:
            prog_data[t, 2 * cRange: 2 * cRange + d_s, 2 * cRange: 2 * cRange + d_s] = importance_sampling(nestedData[prog, :, :], gaussMeans, covNormAngle, xy, yx, samples, d_s, cRange)
        else:
            prog_data[t, 2 * cRange: 2 * cRange + d_s, 2 * cRange: 2 * cRange + d_s] = importance_sampling(prog_data[t-1, :, :], gaussMeans, covNormAngle, xy, yx, samples, d_s, cRange)

        if livePlot:
            if t == 0:
                plt.figure(figsize=(8, 8))
                imP = plt.imshow(prog_data[t, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
                imR = plt.contour(nestedData[prog + t, :, :],
                                  contours, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
                plt.gca().invert_yaxis()
                plt.show(block=False)
                sa = plt.colorbar(imP, format=matplotlib.ticker.ScalarFormatter())
                sa.set_ticks(contours)
                sa.set_clim(0.0, 20.0)
            else:
                imP.set_data(prog_data[t, :, :])
                for tp in imR.collections:
                    tp.remove()
                imR = plt.contour(nestedData[prog + t, :, :], contours,
                                  norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
                plt.pause(0.1)
            #plt.savefig('/scratch/local1/plots/test_prognosis_timestep_'+str(t)+'.png')
time_elapsed = datetime.now() - startTime

print(time_elapsed)
#fig, ax = plt.subplots()
#ax.bar(bin_edges[:-1]-0.1, histX, color='b', width = 0.2)
#ax.bar(bin_edges[:-1]+0.1, histY, color='r', width = 0.2)
#plt.show(block=False)
