import numpy as np
import netCDF4
import os
from datetime import datetime
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import griddata, RegularGridInterpolator
from multiprocessing import Process
from createblob import createblob
from init import Square, Totalfield, LawrData, DWDData, radarData, get_metangle, create_sample, importance_sampling, \
    z2rainrate, findRadarSite, getFiles, nesting, booDisplacement, verification,fileSelector,get_Grid, Results
import multiprocessing as mp

#plt.rcParams['image.cmap'] = 'gist_ncar'
cmap = plt.get_cmap('viridis')
cmap.colors[0] = [0.75, 0.75, 0.75]

# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/2.*sigma**2)



#fp = '/home/zmaw/u300675/pattern_data/m4t_BKM_wrx00_l2_dbz_v00_20130511160000.nc'
#investigate 13.6 18:20
#fp = 'G:/Rainprog/m4t_HHG_wrx00_l2_dbz_v00_20160607150000.nc'
#directoryPath = 'G:/Rainprog/boo/'
#fp = '/home/zmaw/u300675/pattern_data/m4t_BKM_wrx00_l2_dbz_v00_20130426120000.nc' difficult field to predict


def prognosis(date, t):
    try:

        year =date[0]
        mon =date[1]
        day=date[2]
        hour=date[3]
        minute=date[4]
        progTime=date[5]
        startTime = datetime.now()
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

        directoryPath = '/scratch/local1/radardata/simon/dwd_boo/sweeph5allm/'+str(year)+'/'+strMon+'/'+strDay
        fp = '/scratch/local1/radardata/simon/lawr/hhg/level1/'+str(year)+'/'+ strMon +'/HHGlawr2016'+strMon+strDay+ strHour + '_111_L1.nc'
        #directoryPath = 'E:/radardata/02/'
        #fp = 'E:/radardata/'+'HHGlawr2016'+strMon+strDay+ strHour + '_111_L1.nc'

        res = 100
        smallVal = 2
        rainThreshold = 0.5
        distThreshold = 19000
        prog = int(minute*2)
        trainTime = 8
        numMaxes = 20
        useRealData = 1
        prognosis = 1
        statistics = 0
        livePlot = 1
        probabilityFlag = 1
        samples = 16
        blobDisplacementX = -3
        blobDisplacementY = -1
        timeSteps = prog + progTime
        contours = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
        booFileList = sorted(os.listdir(directoryPath))
        dwdTime = list((mon,day,hour,prog))

        selectedFiles = fileSelector(directoryPath, dwdTime, 5)



        dwd = DWDData(directoryPath + '/' + selectedFiles[0])

        for i, file in enumerate(selectedFiles[1:]):
            dwd.addTimestep(directoryPath + '/' + file)
            #print(file)
        dwd.initial_maxima(1)
        dwd.find_displacement(0)



        lawr = LawrData(fp)




        lawr.initial_maxima(prog)
        lawr.find_displacement(prog)

        if np.any(np.isnan(lawr.covNormAngle)) or lawr.normEqualOneSum>len(lawr.progField.activeIds):
            lawr.covNormAngle = dwd.covNormAngle
            lawr.gaussMeans = [x/10*(dwd.resolution/lawr.resolution) for x in dwd.gaussMeans]

        if np.any(np.isnan(dwd.covNormAngle)):
            dwd.covNormAngle = lawr.covNormAngle
            dwd.gaussMeans = [x*10/(dwd.resolution/lawr.resolution) for x in lawr.gaussMeans]


        dwd.extrapolation(progTime+12)
        dwd.HHGPos = findRadarSite(lawr,dwd)
        dwd.set_auxillary_geoData(dwd,lawr,dwd.HHGPos)
        if np.sum(dwd.prog_data[:, ((dwd.dist_nested >= lawr.r[-1]) & (dwd.dist_nested <= lawr.dist_nested.max()))]>rainThreshold)<100:
            nan_dummy = np.zeros([progTime, len(rain_thresholds)]) * np.nan
            result = Results(nan_dummy, nan_dummy, nan_dummy, nan_dummy, nan_dummy, nan_dummy, nan_dummy, nan_dummy,
                             nan_dummy, nan_dummy, year, mon, day, hour, minute)
            return result


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

        if livePlot:
            fig, ax = plt.subplots(figsize=(8, 8))
            for i in range(dwd.nested_data.shape[0]):
                if (i == 0):
                    im = plt.imshow(dwd.nested_data[i, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=cmap)
                    s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter())
                    s.set_clim(0, np.max(dwd.nested_data))
                    s.set_ticks(contours)
                    s.draw_all()
                    radarCircle = mpatches.Circle((dwd.HHGPos[0], dwd.HHGPos[1]), 20000 / dwd.resolution, color='w', linewidth=1, fill=0)
                    ax.add_patch(radarCircle)
                    plt.show(block=False)
                plt.pause(0.1)
                im.set_data(dwd.nested_data[i, :, :])

        if livePlot:
            for t in range(prog-lawr.trainTime+1,prog):
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
                plt.pause(0.1)
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

            col = np.concatenate([np.zeros([1, np.max(dwd.progField.activeIds)]), np.random.rand(2, np.max(dwd.progField.activeIds))])
            plt.figure(figsize=(8, 8))
            ax = plt.axes()
            for i, field in enumerate(dwd.progField.activeFields):
                for t in field.histMaxima:
                    plt.plot(*np.transpose(t[0][2:0:-1]), color=col[:,field.id-1], marker='o')
            for i, field in enumerate(dwd.progField.inactiveFields):
                for t in field.histMaxima:
                    plt.plot(*np.transpose(t[0][2:0:-1]), color=(1, 0, 0), marker='x')
            ax.set_ylim(0, dwd.d_s+4*dwd.cRange)
            ax.set_xlim(0, dwd.d_s+4*dwd.cRange)
            plt.gca().invert_yaxis()
            plt.show(block=False)


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

        lawr.extrapolation(dwd,progTime,prog,probabilityFlag)

        if 0:
            if livePlot:
                for t in range(progTime):
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
                    if len(str(dwd.second[dwd.trainTime+t]))==1:
                        seconds = '0'+str(dwd.second[dwd.trainTime+t])
                    else:
                        seconds = str(dwd.second[dwd.trainTime+t])

                    #plt.savefig('/scratch/local1/plots/prognosis_'+str(year)+strMon+strDay+str(dwd.hour[dwd.trainTime+t])+str(dwd.minute[dwd.trainTime+t])+seconds+'.png')

        #plt.figure()
        #for t in range(len(lawr.probabilities)):
        #    plt.imshow(lawr.probabilities[t, :, :])
        #    if t == 0:
        #        plt.colorbar()

        #    plt.savefig(
        #        '/scratch/local1/plots/probabilities_' + str(year) + strMon + strDay + str(dwd.hour[dwd.trainTime + t]) + str(
        #            dwd.minute[dwd.trainTime + t]) + seconds + '.png')


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


        prog_data = lawr.prog_data
        prog_data[:, lawr.dist_nested >= np.max(lawr.r)] = 0
        hit,miss,f_alert,corr_zero,BIAS,PC,POD,FAR,CSI,ORSS =verification(prog_data, lawr.nested_data[prog:,:,:])
        thresholds=[0.1,0.2,0.5,1,2,5,10,20,30]

        if statistics:
            timeStr = str(year) +'_'+ strMon +'_'+ strDay +'_'+ strHour +'_'+ str(minute)
            path = '/scratch/local1/plots/prognosis_'+timeStr
            if not os.path.exists(path):
                os.mkdir(path)
                print("Directory ", path, " Created ")
            else:
                print("Directory ", path, " already exists")
            plt.figure()
            a=plt.plot(PC[:,0:5])
            plt.legend(a,thresholds[0:5])
            plt.title('PC')
            plt.savefig('/scratch/local1/plots/prognosis_' + timeStr + '/prognosis_' + timeStr + '_PC.png')
            plt.figure()
            a=plt.plot(POD[:,0:5])
            plt.legend(a,thresholds[0:5])
            plt.title('POD')
            plt.savefig('/scratch/local1/plots/prognosis_' + timeStr + '/prognosis_' + timeStr + '_POD.png')
            plt.figure()
            a=plt.plot(FAR[:,0:5])
            plt.legend(a,thresholds[0:5])
            plt.title('FAR')
            plt.savefig('/scratch/local1/plots/prognosis_' + timeStr + '/prognosis_' + timeStr + '_FAR.png')
            plt.figure()
            a=plt.plot(CSI[:,0:5])
            plt.legend(a,thresholds[0:5])
            plt.title('CSI')
            plt.savefig('/scratch/local1/plots/prognosis_' + timeStr + '/prognosis_' + timeStr + '_CSI.png')
            plt.figure()
            a=plt.plot(ORSS[:,0:5])
            plt.legend(a,thresholds[0:5])
            plt.title('ORSS')
            plt.savefig('/scratch/local1/plots/prognosis_' + timeStr + '/prognosis_' + timeStr + '_ORSS.png')

            #fig, ax = plt.subplots()
            #ax.bar(bin_edges[:-1]-0.1, histX, color='b', width = 0.2)
            #ax.bar(bin_edges[:-1]+0.1, histY, color='r', width = 0.2)
            #plt.show(block=False)
            for t in range(progTime):
                if t == 0:
                    pls, axs = plt.subplots(1,2,figsize=(15,9))
                    ax1=axs[1]
                    prob = axs[0].imshow(lawr.probabilities[t,:,:])
                    cb = pls.colorbar(prob, fraction = 0.046, pad = 0.04, ax=axs[0])
                    imP = ax1.contour(lawr.prog_data[t, :, :],
                                      contours, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=cmap)
                    imR = ax1.imshow(lawr.nested_data[prog + t, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=cmap)
                    radarCircle2 = mpatches.Circle(
                        (int(lawr.prog_data[t, :, :].shape[0] / 2), int(lawr.prog_data[t, :, :].shape[1] / 2)),
                        20000 / res, color='w', linewidth=1, fill=0)
                    ax1.add_patch(radarCircle2)
                    ax1.set_xlim([0, lawr.d_s + lawr.cRange * 4])
                    ax1.set_ylim([lawr.d_s + lawr.cRange * 4, 0])
                    s1 = plt.colorbar(imR, fraction = 0.046, pad = 0.04, format=matplotlib.ticker.ScalarFormatter())
                    s1.set_clim(0, np.nanmax(lawr.prog_data))
                    s1.set_ticks(contours)
                    s1.draw_all()
                    plt.tight_layout()
                else:
                    imR.set_data(lawr.nested_data[prog + t, :, :])
                    prob.set_data(lawr.probabilities[t,:,:])
                    for tp in imP.collections:
                        tp.remove()
                    imP = ax1.contour(lawr.prog_data[t, :, :], contours,
                                      norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=cmap)
                if len(str(dwd.second[dwd.trainTime + t])) == 1:
                    seconds = '0' + str(dwd.second[dwd.trainTime + t])
                else:
                    seconds = str(dwd.second[dwd.trainTime + t])
                plt.savefig(
                    '/scratch/local1/plots/prognosis_' + timeStr + '/probability_' + str(year) + strMon + strDay + str(
                        dwd.hour[dwd.trainTime + t]) + str(
                        dwd.minute[dwd.trainTime + t]) + seconds + '.png')


        print(datetime.now()-startTime)


        result = Results(hit, miss, f_alert, corr_zero, BIAS, PC, POD, FAR, CSI, ORSS,year, mon, day, hour, minute)
        return result
    except:
        nan_dummy = np.zeros([progTime, len(rain_thresholds)]) * np.nan
        result = Results(nan_dummy,nan_dummy,nan_dummy,nan_dummy,nan_dummy,nan_dummy,nan_dummy,nan_dummy,nan_dummy,nan_dummy, year, mon, day, hour, minute)
        return result
year = 2016



rain_thresholds = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30])
num_r = len(rain_thresholds)



progTime = 60


runs = 40

months= [6]
startHour = 1
days =[2,13,14,18,23,24,25]
endHour = 24
minutes=np.array([0,10,20])
runs = len(minutes)
hours = np.arange(startHour,endHour)

dimensions = [len(months), len(days), len(hours),len(minutes),progTime,num_r]



hit = np.zeros(dimensions)
miss = np.zeros(dimensions)
f_alert = np.zeros(dimensions)
corr_zero = np.zeros(dimensions)
total = np.zeros(dimensions)
BIAS = np.zeros(dimensions)
PC = np.zeros(dimensions)
POD = np.zeros(dimensions)
FAR = np.zeros(dimensions)
CSI = np.zeros(dimensions)
ORSS = np.zeros(dimensions)

dates = []
for mon in months:
    for day in days:
        for hour in hours:
            for minute in minutes:
                dates.append([year, mon, day, hour, minute, progTime])
t = np.arange(len(dates))
#investigate 13.6 18:20

result = prognosis([2016,6,13,20,0,80],0)
# startTime = datetime.now()
# results2 = []
# for date in dates:
#     try:
#         results2.append(prognosis(date[0],date[1],date[2],date[3],date[4], date[5]))
#     except:
#         results2.append(Results(np.nan([60, 9]), np.nan([60, 9]), np.nan([60, 9]), np.nan([60, 9]), np.nan([60, 9]),
#                                np.nan([60, 9]), np.nan([60, 9]), np.nan([60, 9]), np.nan([60, 9]), np.nan([60, 9]),
#                                year, mon,
#                                day, hour, minute))
# print(datetime.now() - startTime)

# startTime = datetime.now()
# pool = mp.Pool(4)
# results = pool.starmap(prognosis, zip(dates,t))
#
# #except:
# #    nan_dummy = np.zeros([progTime,len(rain_thresholds)])*np.nan
# #    results.append(Results(nan_dummy,nan_dummy,nan_dummy,nan_dummy,nan_dummy,nan_dummy,nan_dummy,nan_dummy,nan_dummy,nan_dummy, np.nan, np.nan, np.nan, np.nan, np.nan))
#
# pool.close()
# pool.join()

import matplotlib.patches as mpatches
contours = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]

fig, ax = plt.subplots(1)
cmap = plt.get_cmap('viridis')
cmap.colors[0] = [0.75, 0.75, 0.75]
dataArea1 =self.nested_data[t+1,
                       (int(field.maxima[0, 1]) - self.cRange * 2):(int(field.maxima[0, 1]) + self.cRange * 2),
                       (int(field.maxima[0, 2]) - self.cRange * 2):(int(field.maxima[0, 2]) + self.cRange * 2)]
dat = ax.imshow(dataArea1, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
plt.plot(48,48,marker='X',color='ghostwhite',markersize=8)
plt.plot(48+int(cIdx[1] - 0.5 * len(c)),48+int(cIdx[0] - 0.5 * len(c)),marker='X',color='black',markersize=8)
s = plt.colorbar(dat, format=matplotlib.ticker.ScalarFormatter())
s.set_label('Precipitation in mm/h')
s.set_clim(0, 20)
s.set_ticks(contours)
s.draw_all()
ax.set_xticklabels(ax.get_xticks()*0.25)
ax.set_yticklabels(ax.get_yticks()*0.25)
plt.show()

fig, ax = plt.subplots(1)
cmap = plt.get_cmap('viridis')
cmap.colors[0] = [0.75, 0.75, 0.75]
dataArea1 =self.nested_data[t,
                       (int(field.maxima[0, 1]) - self.cRange * 2):(int(field.maxima[0, 1]) + self.cRange * 2),
                       (int(field.maxima[0, 2]) - self.cRange * 2):(int(field.maxima[0, 2]) + self.cRange * 2)]
dat = ax.imshow(dataArea1, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1))
corrAreaRect = mpatches.Rectangle((24,24),48,48,color='r',linewidth=1,fill=0)
ax.add_patch(corrAreaRect)
plt.plot(48,48,marker='X',color='ghostwhite',markersize=8)
s = plt.colorbar(dat, format=matplotlib.ticker.ScalarFormatter())
s.set_label('Precipitation in mm/h')
s.set_clim(0, 20)
s.set_ticks(contours)
s.draw_all()
ax.set_xticklabels(ax.get_xticks()*0.25)
ax.set_yticklabels(ax.get_yticks()*0.25)
plt.show()

fig,ax = plt.subplots(1)
corr = ax.imshow(c, cmap=plt.get_cmap('inferno_r'))
plt.colorbar(corr)
plt.plot(24,24,marker='X',color='ghostwhite',markersize=8)
plt.plot(cIdx[1],cIdx[0],marker='X',color='black',markersize=8)
ax.set_xticklabels(ax.get_xticks()*0.25)
ax.set_yticklabels(ax.get_yticks()*0.25)
plt.show()

print(datetime.now() - startTime)