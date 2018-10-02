import numpy as np
import netCDF4
import os
from datetime import datetime
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import griddata, RegularGridInterpolator
from createblob import createblob
from init import Square, Totalfield, LawrData, DWDData, radarData, get_metangle, create_sample, importance_sampling, \
    z2rainrate, findRadarSite, getFiles, nesting, booDisplacement, verification,fileSelector,get_Grid

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
else:
    strHour = str(hour)

if len(str(mon)) == 1:
    strMon = '0' + str(mon)
else:
    strMon = str(mon)
if len(str(day)) == 1:
    strDay = '0' + str(day)
else:
    strDay = str(day)



#fp = 'G:/Rainprog/m4t_HHG_wrx00_l2_dbz_v00_20160607150000.nc'
#directoryPath = 'G:/Rainprog/boo/'

directoryPath = '/scratch/local1/radardata/simon/dwd_boo/sweeph5allm/2016/'+strMon+'/'+strDay
fp = '/scratch/local1/radardata/simon/lawr/hhg/level1/'+str(year)+'/'+ strMon +'/HHGlawr2016'+strMon+strDay+ strHour + '_111_L1.nc'
#directoryPath = 'E:/radardata/02/'
#fp = 'E:/radardata/'+'HHGlawr2016'+strMon+strDay+ strHour + '_111_L1.nc'

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
livePlot = 1
samples = 16
blobDisplacementX = -3
blobDisplacementY = -1
timeSteps = prog + progTime
contours = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
booFileList = sorted(os.listdir(directoryPath))
dwdTime = list((mon,day,hour,prog))

selectedFiles = fileSelector(directoryPath, dwdTime, 5)

startTime=datetime.now()

dwd = DWDData(directoryPath + '/' + selectedFiles[0])

for i, file in enumerate(selectedFiles[1:]):
    dwd.addTimestep(directoryPath + '/' + file)
    #print(file)
print(datetime.now()-startTime)

startTime=datetime.now()
dwd.initial_maxima(1)
print(datetime.now()-startTime)

startTime=datetime.now()
dwd.find_displacement(0)
print(datetime.now()-startTime)

dwd.extrapolation(progTime+15)


startTime=datetime.now()
lawr = LawrData(fp)
print(datetime.now()-startTime)

dwd.HHGPos = findRadarSite(lawr,dwd)
dwd.set_auxillary_geoData(dwd,lawr,dwd.HHGPos)

startTime=datetime.now()
print(datetime.now()-startTime)

startTime=datetime.now()
lawr.initial_maxima(prog)
print(datetime.now()-startTime)

startTime=datetime.now()
lawr.find_displacement(prog)
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


dwd.nested_data[:, (dwd.dist_nested > dwd.r.max())] = 0
fig,ax = plt.subplots(figsize=(8,8))
if livePlot:
    for i in range(dwd.nested_data.shape[0]):
        if (i == 0):
            im = plt.imshow(dwd.nested_data[i, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=cmap)
            s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter())
            s.set_clim(0, np.max(dwd.nested_data))
            s.set_ticks(contours)
            s.draw_all()
            radarCircle = mpatches.Circle((dwd.HHGPos[0], dwd.HHGPos[1]), 20000 / booResolution, color='w', linewidth=1, fill=0)
            ax.add_patch(radarCircle)
            plt.show(block=False)
        plt.pause(0.01)
        im.set_data(dwd.nested_data[i, :, :])

for t in range(prog-lawr.trainTime+1,prog):
    if livePlot:
        if t == prog-lawr.trainTime+1:
            plt.figure(figsize=(8, 8))
            im = plt.imshow(lawr.nested_data[t, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=cmap)
            plt.show(block=False)
            o, = plt.plot(*np.transpose(lawr.progField.return_maxima(t-prog+lawr.trainTime)[:, 2:0:-1]), 'ko')
            n, = plt.plot(*np.transpose(lawr.progField.return_maxima(t-prog+lawr.trainTime-1)[:, 2:0:-1]), 'wo')
            s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter())
            s.set_clim(0, np.max(lawr.nested_data))
            s.set_ticks(contours)
            s.draw_all()
            #s.set_ticklabels(contourLabels)
        else:
            im.set_data(lawr.nested_data[t, :, :])
            o.set_data(*np.transpose(lawr.progField.return_maxima(t-prog+lawr.trainTime)[:, 2:0:-1]))
            n.set_data(*np.transpose(lawr.progField.return_maxima(t-prog+lawr.trainTime-1)[:, 2:0:-1]))
        plt.pause(0.01)
        #plt.savefig('/scratch/local1/plots/analysis_timestep_' + str(t) + '.png')






if statistics:
    col = np.concatenate([np.zeros([1, np.max(lawr.progField.activeIds)]), np.random.rand(2, np.max(lawr.progField.activeIds))])
    plt.figure(figsize=(8, 8))
    ax = plt.axes()
    for i, field in enumerate(lawr.progField.activeFields):
        for t in field.histMaxima:
            plt.plot(*np.transpose(t[0][2:0:-1]), color=col[:,field.id-1], marker='o')
    for i, field in enumerate(lawr.progField.inactiveFields):
        for t in field.histMaxima:
            plt.plot(*np.transpose(t[0][2:0:-1]), color=(1, 0, 0), marker='x')
    ax.set_ylim(0, lawr.d_s+4*lawr.cRange)
    ax.set_xlim(0, lawr.d_s+4*lawr.cRange)
    plt.gca().invert_yaxis()
    plt.show(block=False)

lawr.progField.test_angles()

if statistics:
    plt.figure(figsize=(8, 8))
    ax = plt.axes()
    for i, field in enumerate(lawr.progField.activeFields):
        for t in field.histMaxima:
            plt.plot(*np.transpose(t[0][2:0:-1]), color=col[:,field.id-1], marker='o')
    for i, field in enumerate(lawr.progField.inactiveFields):
        for t in field.histMaxima:
            plt.plot(*np.transpose(t[0][2:0:-1]), color=(1, 0, 0), marker='x')
    ax.set_ylim(0, lawr.d_s+4*lawr.cRange)
    ax.set_xlim(0, lawr.d_s+4*lawr.cRange)
    plt.gca().invert_yaxis()
    plt.show(block=False)


    plt.figure(figsize=(8, 8))
    for q, field in enumerate(lawr.progField.activeFields):
        if field.lifeTime >= trainTime:
            for i, t in enumerate(field.histNorm):
                plt.plot(i, t, color='black', marker='o')
                #plt.plot(i, t - field.histStdNorm[i], color='gray', marker='o')
                #plt.plot(i, t + field.histStdNorm[i], color='gray', marker='o')

            plt.title('HistNorm')

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    for q, field in enumerate(lawr.progField.activeFields):
        if field.lifeTime >= trainTime:
            for i, t in enumerate(field.histAngle):
                plt.polar(t * np.pi / 180, lawr.progField.activeFields[q].histNorm[i], color='k', marker='o',
                          alpha=0.1)

            plt.title('HistAngle')

    plt.show(block=False)

    plt.figure(figsize=(8, 8))
    for q, field in enumerate(lawr.progField.activeFields):
        if field.lifeTime >= trainTime:
            for i, t in enumerate(field.histX):
                plt.plot(i, t, alpha = 0.1, color='black', marker='o')
                #plt.plot(i, t - field.histStdNorm[i], color='gray', marker='o')
                #plt.plot(i, t + field.histStdNorm[i], color='gray', marker='o')

            plt.title('HistX')

    plt.figure(figsize=(8, 8))
    for q, field in enumerate(lawr.progField.activeFields):
        if field.lifeTime >= trainTime:
            for i, t in enumerate(field.histY):
                plt.plot(i, t, alpha = 0.1, color='navy', marker='o')
                #plt.plot(i, t - field.histStdAngle[i], color='blue', marker='o')
                #plt.plot(i, t + field.histStdAngle[i], color='blue', marker='o')

            plt.title('HistY')

    plt.show(block=False)

#use allFieldsMeanNorm & np.sin(allFieldsMeanAngle*allFieldsMeanNorm for means
#use covNormAngle for covariance matrix
#x,y = np.random.multivariate_normal([allFieldsMeanNorm, np.sin(allFieldsMeanAngle*allFieldsMeanNorm)], np.cov(allFieldsNorm, np.sin(allFieldsAngle*allFieldsNorm)), 32).T

#if not useRealData:
resScale=1

startTime=datetime.now()
lawr.extrapolation(dwd,progTime,prog)
print(datetime.now()-startTime)

if prognosis:
    for t in range(progTime):
        if livePlot:
            if t == 0:
                hhgFig,ax1 = plt.subplots(1)
                imP = ax1.imshow(lawr.prog_data[t, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=cmap)
                imR = ax1.contour(lawr.nested_data[prog + t, :, :],
                                  contours, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
                radarCircle2 = mpatches.Circle(
                    (int(lawr.prog_data[t, :, :].shape[0] / 2), int(lawr.prog_data[t, :, :].shape[1] / 2)),
                    20000 / res, color='w', linewidth=1, fill=0)
                ax1.add_patch(radarCircle2)
                plt.show(block=False)
                s1 = plt.colorbar(imP, format=matplotlib.ticker.ScalarFormatter())
                s1.set_clim(0, np.nanmax(lawr.prog_data))
                s1.set_ticks(contours)
                s1.draw_all()
            else:
                imP.set_data(lawr.prog_data[t, :, :])
                for tp in imR.collections:
                    tp.remove()
                imR = ax1.contour(lawr.nested_data[prog + t, :, :], contours,
                                  norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
            plt.pause(0.1)
            #plt.savefig('/scratch/local1/plots/prognosis_timestep_' + str(t) + '.png')


hit,miss,f_alert,corr_zero,BIAS,PC,POD,FAR,CSI,ORSS =verification(lawr.prog_data, lawr.nested_data[prog:,:,:])
# if livePlot:
#     for t in range(progTime):
#         if t == 0:
#             booFig,ax2 = plt.subplots(1)
#             booIm = ax2.imshow(dwd.prog_data[t, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=cmap)
#             booImR = ax2.contour(dwd.R[prog + t, :, :],
#                                  contours, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
#             plt.show(block=False)
#             s2 = plt.colorbar(booIm, format=matplotlib.ticker.ScalarFormatter())
#             s2.set_clim(0, np.nanmax(dwd.prog_data))
#             s2.set_ticks(contours)
#             s2.draw_all()
#         else:
#             booIm.set_data(dwd.prog_data[t, :, :])
#             for tp in booImR.collections:
#                 tp.remove()
#             booImR = ax2.contour(dwd.R[prog + t, :, :],
#                                  contours, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
#             plt.pause(0.1)
        #plt.savefig('/scratch/local1/plots/test_prognosis_timestep_'+str(t)+'.png')

time_elapsed = datetime.now()- startTime
print('Total time: '+str(time_elapsed))



#fig, ax = plt.subplots()
#ax.bar(bin_edges[:-1]-0.1, histX, color='b', width = 0.2)
#ax.bar(bin_edges[:-1]+0.1, histY, color='r', width = 0.2)
#plt.show(block=False)
