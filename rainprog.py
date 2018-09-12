import numpy as np
import netCDF4
import os
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import griddata, RegularGridInterpolator
from createblob import createblob
from findmaxima import findmaxima
from init import Square, totalField, get_metangle, interp_weights, interpolate, create_sample, importance_sampling, \
    DWDData, z2rainrate, findRadarSite, getFiles, nesting, booDisplacement, verification

#plt.rcParams['image.cmap'] = 'gist_ncar'
cmap = plt.get_cmap('viridis')
cmap.colors[0] = [0.75, 0.75, 0.75]

# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/2.*sigma**2)


#fp = '/home/zmaw/u300675/pattern_data/m4t_BKM_wrx00_l2_dbz_v00_20130511160000.nc'
startTime = datetime.now()
year = str(2016)
mon=str(6)
day=str(7)
rTime = 17-2
fp = 'G:/Rainprog/m4t_HHG_wrx00_l2_dbz_v00_20160607150000.nc'
directoryPath = 'G:/Rainprog/boo/'
#fp = '/home/zmaw/u300675/pattern_data/m4t_BKM_wrx00_l2_dbz_v00_20130426120000.nc' difficult field to predict
strTime = str(rTime)
if len(strTime) == 1:
    strTime = '0' + strTime
if len(mon) == 1:
    mon = '0' + mon
if len(day) == 1:
    day = '0' + day



#fp = 'G:/Rainprog/m4t_HHG_wrx00_l2_dbz_v00_20160607150000.nc'
#directoryPath = 'G:/Rainprog/boo/'

#directoryPath = '/scratch/local1/radardata/simon/dwd_boo/sweeph5allm/2016/'+mon+'/'+day
#fp = '/scratch/local1/radardata/simon/lawr/hhg/level1/'+year+'/'+ mon +'/HHGlawr2016'+mon+day+ strTime + '_111_L1.nc'

booFileList = sorted(os.listdir(directoryPath))
selectedFiles = getFiles(booFileList, rTime)

res = 100
booResolution = 200
resScale = booResolution / res
smallVal = 2
rainThreshold = 0.1
distThreshold = 19000
prog = 30
trainTime = 8
numMaxes = 20
progTime = 60
useRealData = 1
prognosis = 1
statistics = 0
livePlot = 0
samples = 16
blobDisplacementX = -3
blobDisplacementY = -1
timeSteps = prog + progTime
contours = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]

nc = netCDF4.Dataset(fp)
try:
    data = nc.variables['dbz_ac1'][:][:][:]
    z = data
    azi = nc.variables['azi'][:]
    r = nc.variables['range'][:]
    time = nc.variables['time'][:]
except:
    #data = nc.variables['Att_Corr_Xband_Reflectivity'][:][:][:]
    data = nc.variables['CLT_Corr_Reflectivity'][:][:][:]
    #data_Xband = nc.variables['Att_Corr_Xband_Reflectivity'][:][:][:]
    #data_Cband = nc.variables['Att_Corr_Cband_Reflectivity'][:][:][:]
    if np.ma.is_masked(data):
        data.fill_value = -32.5
        z = data.filled()
    else:
        z = data

    #if np.ma.is_masked(data_Cband):
    #    data_Cband.fill_value = -32.5
    #    z_Cband = data_Cband.filled()
    #else:
    #    z_Cband = data_Cband

    azi = nc.variables['Azimuth'][:]
    r = nc.variables['Distance'][:]
    time = nc.variables['Time'][:]

aziCorr = -5
azi = np.mod(azi + aziCorr,360)
cRange = int(800/res) # 800m equals an windspeed of aprox. 100km/h and is set as the upper boundary for a possible cloud movement
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

xCar_nested = np.arange(-20000 - cRange * 2 * res, 20000 + cRange * 2 * res + 1, res).squeeze()
yCar_nested = xCar_nested

[XCar_nested, YCar_nested] = np.meshgrid(xCar_nested, yCar_nested)

Lat_nested = lat + XCar_nested / latDeg
Lon_nested = lon + XCar_nested / (lonDeg * (np.cos(Lat_nested * np.pi / 180)))
nested_dist = np.sqrt(np.square(xCar_nested)+np.square(YCar_nested))

target_nested = np.zeros([XCar_nested.shape[0]*XCar_nested.shape[1],2])
target_nested[:, 0] = XCar_nested.flatten()
target_nested[:, 1] = YCar_nested.flatten()


target = np.zeros([XCar.shape[0]*XCar.shape[1],2])
target[:, 0] = XCar.flatten()
target[:, 1] = YCar.flatten()


d_s = len(XCar)

R = np.empty([timeSteps,d_s,d_s])
rPolar = z2rainrate(z)

nested_data = np.zeros([timeSteps, d_s + 4 * cRange, d_s + 4 * cRange])
vtx, wts = interp_weights(points, target)
if useRealData:
    for t in range(timeSteps):
        rPolarT = rPolar[t, :, :].T
        rPolarT = np.reshape(rPolarT, (333*360, 1)).squeeze()
        R[t, :, :] = np.reshape(interpolate(rPolarT.flatten(), vtx, wts), (d_s, d_s))
        R[t, (dist >= np.max(r))] = 0
        nested_data[t, 2 * cRange: 2 * cRange + d_s, 2 * cRange: 2 * cRange + d_s] = R[t, :, :]
    nested_data = np.rot90(nested_data, 1, (1, 2))
    R = np.rot90(R, 1, (1, 2))
else:
    R = createblob(d_s, res, timeSteps, u = blobDisplacementX, v = blobDisplacementY)
    R[:, (dist >= np.max(r))] = 0
    nested_data[:, 2 * cRange: 2 * cRange + d_s, 2 * cRange: 2 * cRange + d_s] = R
    #nested_data = np.rot90(nested_data, 1, (1, 2))
    #R = np.rot90(R, 1, (1, 2))
startTime2=datetime.now()

boo=DWDData()
boo.read_dwd_file(directoryPath + '/' + selectedFiles[0])
selectedFiles.pop(0)
boo.getGrid(booResolution)
boo.gridding(boo.vtx, boo.wts, boo.d_s)


for i, file in enumerate(selectedFiles):
    buf = DWDData()
    buf.read_dwd_file(directoryPath + '/' + selectedFiles[i])
    buf.gridding(boo.vtx, boo.wts, boo.d_s)
    boo.addTimestep(buf.R)
    boo.time = int(selectedFiles[i][43:45])

print(datetime.now()-startTime2)

startTime2=datetime.now()
boo.timeInterpolation(121)
boo.R = boo.R[:,:,:120]
print(datetime.now()-startTime2)
boo.R = np.flip(np.rot90(np.swapaxes(boo.R, 0, 2),1,(1,2)),2)
HHGposition = findRadarSite(lat, lon, boo)


if not useRealData:
    boo.R = createblob(boo.d_s, booResolution, timeSteps, u = blobDisplacementX/resScale, v = blobDisplacementY/resScale, x0 = HHGposition[0], x1= HHGposition[0], y0=HHGposition[1]+200, amp = 25, sigma = 4)


boo.HHGdist = np.sqrt(np.square(boo.XCar - boo.XCar.min()- HHGposition[0] * booResolution) +
                      np.square(boo.YCar - boo.YCar.min()- HHGposition[1] * booResolution))

boo.HHG_cart_points = np.concatenate((np.reshape(boo.XCar - boo.XCar.min()- HHGposition[0] * booResolution,
                                       (boo.d_s * boo.d_s,1)),
                                     np.reshape(boo.YCar - boo.YCar.min()- HHGposition[1] * booResolution,
                                      (boo.d_s * boo.d_s,1))), axis=1)

boo.R[:, (boo.dist > boo.r.max())] = 0
fig,ax = plt.subplots(figsize=(8,8))
if livePlot:
    for i in range(boo.R.shape[0]):
        if (i == 0):
            im = plt.imshow(boo.R[i, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=cmap)
            s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter())
            s.set_clim(0, np.max(nested_data))
            s.set_ticks(contours)
            s.draw_all()
            radarCircle = mpatches.Circle((HHGposition[0], HHGposition[1]), 20000 / booResolution, color='w', linewidth=1, fill=0)
            ax.add_patch(radarCircle)
            plt.show(block=False)
        plt.pause(0.01)
        im.set_data(boo.R[i, :, :])

nested_data = np.nan_to_num(nested_data)
R = np.nan_to_num(R)

allFields = totalField(findmaxima([], R[0, :, :], cRange, numMaxes, rainThreshold, distThreshold, dist), rainThreshold, distThreshold, dist, numMaxes, nested_data, R, res, cRange, trainTime)
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

        #print('looked for new maxima')

    #fields = testmaxima(fields, nestedData[t, :, :], rainThreshold, distThreshold, res, cRange)
    allFields.test_maxima(nested_data[t, :, :])
    allFields.prog_step(t)

    if livePlot:
        if t == 0:
            plt.figure(figsize=(8, 8))
            im = plt.imshow(nested_data[t, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=cmap)
            plt.show(block=False)
            o, = plt.plot(*np.transpose(allFields.return_maxima(0)[:, 2:0:-1]), 'ko')
            n, = plt.plot(*np.transpose(allFields.return_maxima(-1)[:, 2:0:-1]), 'wo')
            s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter())
            s.set_clim(0, np.max(nested_data))
            s.set_ticks(contours)
            s.draw_all()
            #s.set_ticklabels(contourLabels)
        else:
            im.set_data(nested_data[t, :, :])
            o.set_data(*np.transpose(allFields.return_maxima(0)[:, 2:0:-1]))
            n.set_data(*np.transpose(allFields.return_maxima(-1)[:, 2:0:-1]))
        plt.pause(0.01)
        #plt.savefig('/scratch/local1/plots/analysis_timestep_' + str(t) + '.png')

    allFields.update_fields()




col = np.concatenate([np.zeros([1,np.max(allFields.activeIds)]), np.random.rand(2,np.max(allFields.activeIds))])

if statistics:
    plt.figure(figsize=(8, 8))
    ax = plt.axes()
    for i, field in enumerate(allFields.activeFields):
        for t in field.histMaxima:
            plt.plot(*np.transpose(t[0][2:0:-1]), color=col[:,field.id-1], marker='o')
    for i, field in enumerate(allFields.inactiveFields):
        for t in field.histMaxima:
            plt.plot(*np.transpose(t[0][2:0:-1]), color=(1, 0, 0), marker='x')
    ax.set_ylim(0, d_s+4*cRange)
    ax.set_xlim(0, d_s+4*cRange)
    plt.gca().invert_yaxis()
    plt.show(block=False)

allFields.test_angles()

if statistics:
    plt.figure(figsize=(8, 8))
    ax = plt.axes()
    for i, field in enumerate(allFields.activeFields):
        for t in field.histMaxima:
            plt.plot(*np.transpose(t[0][2:0:-1]), color=col[:,field.id-1], marker='o')
    for i, field in enumerate(allFields.inactiveFields):
        for t in field.histMaxima:
            plt.plot(*np.transpose(t[0][2:0:-1]), color=(1, 0, 0), marker='x')
    ax.set_ylim(0, d_s)
    ax.set_xlim(0, d_s)
    plt.gca().invert_yaxis()
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

allFieldsMeanX = np.nanmean(allFields.return_fieldHistX())
allFieldsMeanY = np.nanmean(allFields.return_fieldHistY())
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

allFieldsNorm = allFieldsNorm[~np.isnan(allFieldsAngle)]
allFieldsAngle = allFieldsAngle[~np.isnan(allFieldsAngle)]

covNormAngle = np.cov(allFieldsNorm, np.sin(allFieldsAngle*allFieldsNorm))
gaussMeans = [allFieldsMeanX, allFieldsMeanY]
#use allFieldsMeanNorm & np.sin(allFieldsMeanAngle*allFieldsMeanNorm for means
#use covNormAngle for covariance matrix
#x,y = np.random.multivariate_normal([allFieldsMeanNorm, np.sin(allFieldsMeanAngle*allFieldsMeanNorm)], np.cov(allFieldsNorm, np.sin(allFieldsAngle*allFieldsNorm)), 32).T

boo.nested_data = np.zeros([1, boo.d_s, boo.d_s])
#boo.nested_data[0, 2 * cRange:boo.d_s + 2 * cRange, 2 * cRange:boo.d_s + 2 * cRange] =boo.R[prog,:,:]
boo.nested_data[0, :, :] =boo.R[prog+5,:,:]
#if not useRealData:
resScale=1

startTime = datetime.now()

if prognosis:
    for t in range(progTime):
        if t == 0:
            boo.prog_data = np.zeros([progTime, boo.d_s, boo.d_s])

            #boo.prog_data[t, :, :] = griddata(boo.cart_points,
            #                                  boo.nested_data[0, :, :].flatten(),
            #                                  (boo.XCar - displacementY * resScale,
            #                                   boo.YCar - displacementX * resScale), method='linear')

            boo.prog_data[t, :, :] = booDisplacement(boo, boo.nested_data[0,:,:], displacementX * resScale, displacementY * resScale)

            prog_data = np.zeros([progTime, d_s + 4 * cRange, d_s + 4 * cRange])
            yx, xy = np.meshgrid(np.arange(0, 4 * cRange + d_s), np.arange(0, 4 * cRange + d_s))
            xSample, ySample = create_sample(gaussMeans, covNormAngle, samples)

            prog_data[t, :, :] = nesting(nested_data[prog, :, :], nested_dist, target_nested,
                                         boo.prog_data[t, :, :], boo, r[-1], rainThreshold, Lat_nested, Lon_nested)

            prog_data[t, :, :] = \
                importance_sampling(prog_data[t, :,:], nested_dist, r[-1], xy, yx, xSample, ySample, d_s, cRange)
                #importance_sampling(nested_data[prog, (nested_dist < np.max(r))], xy, yx, xSample, ySample, d_s, cRange)
        else:
            #boo.prog_data[t, :, :] = griddata(boo.cart_points, boo.prog_data[t - 1, :, :].flatten(),
            #                                  (boo.XCar - displacementY * resScale,
            #                                   boo.YCar - displacementX * resScale), method='linear')
            boo.prog_data[t, :, :] = booDisplacement(boo, boo.prog_data[t-1,:,:], displacementX * resScale, displacementY * resScale)

            prog_data[t, :, :] = prog_data[t-1, :, :]

            prog_data[t, :, :] = nesting(prog_data[t, :, :], nested_dist, target_nested, boo.prog_data[t, :, :], boo,
                                         r[-1], rainThreshold, Lat_nested, Lon_nested)

            prog_data[t, :, :] = \
                importance_sampling(prog_data[t, :,:], nested_dist, r[-1], xy, yx, xSample, ySample, d_s, cRange)



        if livePlot:
            if t == 0:
                hhgFig,ax1 = plt.subplots(1)
                imP = ax1.imshow(prog_data[t, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=cmap)
                imR = ax1.contour(nested_data[prog + t, :, :],
                                  contours, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
                radarCircle2 = mpatches.Circle(
                    (int(prog_data[t, :, :].shape[0] / 2), int(prog_data[t, :, :].shape[1] / 2)),
                    20000 / res, color='w', linewidth=1, fill=0)
                ax1.add_patch(radarCircle2)
                plt.show(block=False)
                s1 = plt.colorbar(imP, format=matplotlib.ticker.ScalarFormatter())
                s1.set_clim(0, np.nanmax(prog_data))
                s1.set_ticks(contours)
                s1.draw_all()
            else:
                imP.set_data(prog_data[t, :, :])
                for tp in imR.collections:
                    tp.remove()
                imR = ax1.contour(nested_data[prog + t, :, :], contours,
                                  norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
            plt.pause(0.1)
            #plt.savefig('/scratch/local1/plots/prognosis_timestep_' + str(t) + '.png')

time_elapsed = datetime.now()- startTime
print('prognosis'+str(time_elapsed))
hit,miss,f_alert,corr_zero,BIAS,PC,POD,FAR,CSI,ORSS =verification(prog_data, nested_data[prog:,:,:])
if livePlot:
    for t in range(progTime):
        if t == 0:
            booFig,ax2 = plt.subplots(1)
            booIm = ax2.imshow(boo.prog_data[t, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=cmap)
            booImR = ax2.contour(boo.R[prog+t, :, :],
                                 contours, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
            plt.show(block=False)
            s2 = plt.colorbar(booIm, format=matplotlib.ticker.ScalarFormatter())
            s2.set_clim(0, np.nanmax(boo.prog_data))
            s2.set_ticks(contours)
            s2.draw_all()
        else:
            booIm.set_data(boo.prog_data[t, :, :])
            for tp in booImR.collections:
                tp.remove()
            booImR = ax2.contour(boo.R[prog+t, :, :],
                                 contours, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
            plt.pause(0.1)
        #plt.savefig('/scratch/local1/plots/test_prognosis_timestep_'+str(t)+'.png')





#fig, ax = plt.subplots()
#ax.bar(bin_edges[:-1]-0.1, histX, color='b', width = 0.2)
#ax.bar(bin_edges[:-1]+0.1, histY, color='r', width = 0.2)
#plt.show(block=False)
