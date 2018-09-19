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
from init import Square, Totalfield, LawrData, DWDData, radarData, get_metangle, create_sample, importance_sampling, \
    z2rainrate, findRadarSite, getFiles, nesting, booDisplacement, verification,fileSelector

#plt.rcParams['image.cmap'] = 'gist_ncar'
cmap = plt.get_cmap('viridis')
cmap.colors[0] = [0.75, 0.75, 0.75]

# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/2.*sigma**2)

startTime = datetime.now()

#fp = '/home/zmaw/u300675/pattern_data/m4t_BKM_wrx00_l2_dbz_v00_20130511160000.nc'
year = 2016
mon= 6
day= 2
hour = 9-2
#fp = 'G:/Rainprog/m4t_HHG_wrx00_l2_dbz_v00_20160607150000.nc'
#directoryPath = 'G:/Rainprog/boo/'
#fp = '/home/zmaw/u300675/pattern_data/m4t_BKM_wrx00_l2_dbz_v00_20130426120000.nc' difficult field to predict

if len(str(hour)) == 1:
    strHour = '0' + str(hour)
if len(str(mon)) == 1:
    strMon = '0' + str(mon)
if len(str(day)) == 1:
    strDay = '0' + str(day)



#fp = 'G:/Rainprog/m4t_HHG_wrx00_l2_dbz_v00_20160607150000.nc'
#directoryPath = 'G:/Rainprog/boo/'

directoryPath = '/scratch/local1/radardata/simon/dwd_boo/sweeph5allm/2016/'+strMon+'/'+strDay
fp = '/scratch/local1/radardata/simon/lawr/hhg/level1/'+str(year)+'/'+ strMon +'/HHGlawr2016'+strMon+strDay+ strHour + '_111_L1.nc'


res = 100
booResolution = 200
resScale = booResolution / res
smallVal = 2
rainThreshold = 0.1
distThreshold = 19000
prog = 45
trainTime = 8
numMaxes = 20
progTime = 60
useRealData = 1
prognosis = 1
statistics = 0
livePlot = 1
samples = 16
blobDisplacementX = -3
blobDisplacementY = -1
timeSteps = prog + progTime
contours = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
booFileList = sorted(os.listdir(directoryPath))
dwdTime = list((mon,day,hour,prog))

selectedFiles = fileSelector(directoryPath, dwdTime, 5)
selectedFiles = getFiles(booFileList, hour)

startTime=datetime.now()

dwd = DWDData(directoryPath + '/' + selectedFiles[0])

for i, file in enumerate(selectedFiles[1:]):
    dwd.addTimestep(directoryPath + '/' + file)
    #print(file)
print(datetime.now()-startTime)
startTime=datetime.now()
dwd.initial_maxima(5)
print(datetime.now()-startTime)
startTime=datetime.now()
dwd.find_displacement()
print(datetime.now()-startTime)
startTime=datetime.now()
dwd.extrapolation(progTime)
print(datetime.now()-startTime)
startTime=datetime.now()
lawr = LawrData(fp)
print(datetime.now()-startTime)


# for t in range(test2.trainTime):
#     if t == 0:
#         plt.figure(figsize=(8, 8))
#         im = plt.imshow(test2.nested_data[t, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=cmap)
#         plt.show(block=False)
#         o, = plt.plot(*np.transpose(test2.progField.return_maxima(t)[:, 2:0:-1]), 'ko')
#         n, = plt.plot(*np.transpose(test2.progField.return_maxima(t-1)[:, 2:0:-1]), 'wo')
#         s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter())
#         s.set_clim(0, np.max(test2.nested_data))
#         s.set_ticks(contours)
#         s.draw_all()
#         # s.set_ticklabels(contourLabels)
#     else:
#         im.set_data(test2.nested_data[t, :, :])
#         o.set_data(*np.transpose(test2.progField.return_maxima(t)[:, 2:0:-1]))
#         n.set_data(*np.transpose(test2.progField.return_maxima(t-1)[:, 2:0:-1]))
#     plt.pause(1)
#
# col = np.concatenate([np.zeros([1,np.max(test2.progField.activeIds)]), np.random.rand(2,np.max(test2.progField.activeIds))])
# plt.figure(figsize=(8, 8))
# ax = plt.axes()
# for i, field in enumerate(test2.progField.activeFields):
#     for t in field.histMaxima:
#         plt.plot(*np.transpose(t[0][2:0:-1]), color=col[:, field.id - 1], marker='o')
# for i, field in enumerate(test2.progField.inactiveFields):
#     for t in field.histMaxima:
#         plt.plot(*np.transpose(t[0][2:0:-1]), color=(1, 0, 0), marker='x')
# ax.set_ylim(0, test2.d_s + 4 * cRange)
# ax.set_xlim(0, test2.d_s + 4 * cRange)
# plt.gca().invert_yaxis()
# plt.show(block=False)
#
# print(datetime.now()-startTime2)

HHGposition = findRadarSite(lat, lon, boo)




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

allFields = Totalfield(Totalfield.findmaxima([], nested_data[0, :, :], cRange, numMaxes, rainThreshold, distThreshold, dist), rainThreshold, distThreshold, dist, numMaxes, nested_data, res, cRange, trainTime)


for t in range(prog):
    #print(t)
    #maxima, status = testmaxima(maxima, nestedData[t, :, :], rainThreshold, distThreshold, res, status)
    if len(allFields.activeFields) < numMaxes:
        #allFields.activeFields = findmaxima(allFields.activeFields, nested_data[t, :, :], cRange, numMaxes, rainThreshold, distThreshold, dist)
        allFields.assign_ids()
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
boo.nested_data[0, :, :] =boo.R[prog,:,:]
#if not useRealData:
resScale=1


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

time_elapsed = datetime.now()- startTime
print('Total time: '+str(time_elapsed))



#fig, ax = plt.subplots()
#ax.bar(bin_edges[:-1]-0.1, histX, color='b', width = 0.2)
#ax.bar(bin_edges[:-1]+0.1, histY, color='r', width = 0.2)
#plt.show(block=False)
