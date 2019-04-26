import numpy as np
import netCDF4
import os,sys
from datetime import datetime
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import griddata, RegularGridInterpolator
from multiprocessing import Process
from createblob import createblob
from init import Square, Totalfield, LawrData, DWDData, radarData, get_metangle, create_sample, importance_sampling, \
    z2rainrate, findRadarSite, getFiles, nesting, booDisplacement, verification,dwdFileSelector,lawrFileSelector,get_Grid, Results
import multiprocessing as mp

#plt.rcParams['image.cmap'] = 'gist_ncar'
import matplotlib.patches as mpatches

import matplotlib.colors as colors
contours = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'cmaptest',cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('BuPu')
newcmap=truncate_colormap(cmap,0.2,1)
newcmap.set_under('1')
#plt.rcParams["figure.figsize"] = (9,6)
params = {"pgf.texsystem": "pdflatex"}
plt.rcParams.update(params)

# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/2.*sigma**2)



#fp = '/home/zmaw/u300675/pattern_data/m4t_BKM_wrx00_l2_dbz_v00_20130511160000.nc'
#investigate 13.6 18:20
#fp = 'G:/Rainprog/m4t_HHG_wrx00_l2_dbz_v00_20160607150000.nc'
#directoryPath = 'G:/Rainprog/boo/'
#fp = '/home/zmaw/u300675/pattern_data/m4t_BKM_wrx00_l2_dbz_v00_20130426120000.nc' difficult field to predict


def prognosis(date,t):
    try:

        year =date[0]
        mon =date[1]
        day=date[2]
        hour=date[3]
        minute=date[4]
        progTime=120
        startTime = datetime.now()
        dwdDirectoryPath = '/scratch/local1/radardata/simon/dwd_boo/sweeph5allm/'+str(year)+'/'+str(mon).zfill(2)+'/'+str(day).zfill(2)
        lawrDirectoryPath = '/scratch/local1/radardata/simon/lawr/hhg/level1/'+str(year)+'/'+ str(mon).zfill(2) +'/'
        #fp = '/scratch/local1/radardata/simon/lawr/hhg/level1/'+str(year)+'/'+ str(mon).zfill(2) +'/HHGlawr2016'+str(mon).zfill(2)+str(day).zfill(2)+ str(hour).zfill(2) + '_111_L1.nc'
        #dwdDirectoryPath = '/scratch/local1/radardata/simon/nonconvective/dwd/level1'
        #lawrDirectoryPath = '/scratch/local1/radardata/simon/nonconvective/lawr/HHG/level1_cband/'
        res = 100
        prog = int(minute*2)
        trainTime = 6
        statistics = 0
        livePlot = 0
        probabilityFlag = 1
        rain_threshold = 0.5
        contours = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
        booFileList = sorted(os.listdir(dwdDirectoryPath))
        dwdTime = list((year, mon,day,hour,prog))

        dwdSelectedFiles = dwdFileSelector(dwdDirectoryPath, dwdTime, trainTime)



        dwd = DWDData(dwdDirectoryPath + '/' + dwdSelectedFiles[0],dwdTime)

        for i,file in enumerate(dwdSelectedFiles[1:]):
            dwd.addTimestep(dwdDirectoryPath + '/' + file)
        dwd.initial_maxima()
        dwd.find_displacement(0)

        lawrSelectedFiles = lawrFileSelector(lawrDirectoryPath, date)
        lawr = LawrData(lawrDirectoryPath+'/'+lawrSelectedFiles[0])

        for i,file in enumerate(lawrSelectedFiles[1:]):
            lawr.addTimestep(lawrDirectoryPath+'/'+file)

        lawr.setStart(date)
        lawr.initial_maxima()

        resFactor = (dwd.resolution/lawr.resolution) # resolution factor  between dwd and lawr
        if np.any(np.abs([x/10*resFactor for x in dwd.gaussMeans])<1):
            lawr.progField.deltaT=int(np.ceil(1/np.max(np.abs([x/10*resFactor for x in dwd.gaussMeans]))))+1

            if lawr.progField.deltaT > 5:
                lawr.progField.deltaT = 5

        lawr.find_displacement(prog)

        if np.any(np.isnan(lawr.covNormAngle)) or lawr.progField.deltaT>8 or len(lawr.progField.activeFields):
            lawr.covNormAngle = np.cov((dwd.progField.return_fieldHistX().flatten() / 10) * resFactor, dwd.progField.return_fieldHistY().flatten() / 10 * resFactor)
            lawr.gaussMeans = [x/10*resFactor for x in dwd.gaussMeans]



        if np.any(np.isnan(dwd.covNormAngle)):
            dwd.covNormAngle = lawr.covNormAngle
            dwd.gaussMeans = [x*10/resFactor for x in lawr.gaussMeans]
            dwd.covNormAngle_norm = np.cov(lawr.progField.return_fieldHistX().flatten() / resFactor,
                                           lawr.progField.return_fieldHistY().flatten() / resFactor)
            dwd.gaussMeans_norm = [x/resFactor for x in lawr.gaussMeans]

        else:
            dwd.covNormAngle_norm = np.cov(dwd.progField.return_fieldHistX().flatten() / 10,
                                           dwd.progField.return_fieldHistY().flatten() / 10)
            dwd.gaussMeans_norm = [x/10 for x in dwd.gaussMeans]

        dwdVariancefactor = 1
        dwd.extrapolation(progTime+12,dwdVariancefactor)
        dwd.HHGPos = findRadarSite(lawr,dwd)
        dwd.set_auxillary_geoData(dwd,lawr,dwd.HHGPos)
        if np.sum(dwd.prog_data[:, ((dwd.dist_nested >= lawr.r[-1]) & (dwd.dist_nested <= lawr.dist_nested.max()))]>lawr.rainThreshold)<100:
            nan_dummy = np.zeros([progTime, len(rain_thresholds)]) * np.nan
            result = Results(nan_dummy, nan_dummy, nan_dummy, nan_dummy, nan_dummy, nan_dummy, nan_dummy, nan_dummy,
                             nan_dummy, nan_dummy, year, mon, day, hour, minute,np.nan,np.nan)
            print(str(mon)+'_'+str(day)+'_'+str(hour)+ '_'+str(minute)+' has no rain.')
            print(datetime.now() - startTime)
            return
            #return result


        # for t in range(test2.trainTime):
        #     if t == 0:
        #         plt.figure(figsize=(8, 8))
        #         im = plt.imshow(test2.nested_data[t, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=cmap)
        #         plt.show(block=False)
        #         o, = plt.plot(*np.transpose(test2.progField.return_histMaxima(t)[:, 2:0:-1]), 'ko')
        #         n, = plt.plot(*np.transpose(test2.progField.return_histMaxima(t-1)[:, 2:0:-1]), 'wo')
        #         s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter())
        #         s.set_clim(0, np.max(test2.nested_data))
        #         s.set_ticks(contours)
        #         s.draw_all()
        #         # s.set_ticklabels(contourLabels)
        #     else:
        #         im.set_data(test2.nested_data[t, :, :])
        #         o.set_data(*np.transpose(test2.progField.return_histMaxima(t)[:, 2:0:-1]))
        #         n.set_data(*np.transpose(test2.progField.return_histMaxima(t-1)[:, 2:0:-1]))
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
                    o, = plt.plot(*np.transpose(lawr.progField.return_histMaxima(t-prog+lawr.trainTime)[:, 2:0:-1]), 'ko')
                    n, = plt.plot(*np.transpose(lawr.progField.return_histMaxima(t-prog+lawr.trainTime-1)[:, 2:0:-1]), 'wo')
                    s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter())
                    s.set_clim(0, np.max(lawr.nested_data))
                    s.set_ticks(contours)
                    s.draw_all()
                    #s.set_ticklabels(contourLabels)
                else:
                    im.set_data(lawr.nested_data[t, :, :])
                    o.set_data(*np.transpose(lawr.progField.return_histMaxima(t-prog+lawr.trainTime)[:, 2:0:-1]))
                    n.set_data(*np.transpose(lawr.progField.return_histMaxima(t-prog+lawr.trainTime-1)[:, 2:0:-1]))
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
        variancefactor = 8
        lawr.extrapolation(dwd,progTime,prog,probabilityFlag,variancefactor)

        if 1:
            if livePlot:
                for t in range(progTime):
                    if t == 0:
                        hhgFig,ax1 = plt.subplots(1)
                        imP = ax1.imshow(lawr.prog_data[t, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=cmap)
                        imR = ax1.contour(lawr.nested_data[lawr.progStartIdx + t, :, :],
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
                        imR = ax1.contour(lawr.nested_data[lawr.progStartIdx + t, :, :], contours,
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
        real_data = lawr.nested_data[lawr.progStartIdx:lawr.progStartIdx + progTime, :, :] > rain_threshold

        f = netCDF4.Dataset('/scratch/local1/radardata/prognosis12_variance8/'+str(mon) + '_' + str(day) +'_'+str(hour)+ '_' + str(minute) + '_prognosis.nc', 'w', format='NETCDF4')  # 'w' stands for write
        tempgrp = f.createGroup('Prognosis Data')
        tempgrp.createDimension('time', len(lawr.probabilities))
        tempgrp.createDimension('x', lawr.probabilities.shape[1])
        tempgrp.createDimension('y', lawr.probabilities.shape[2])

        TIME = tempgrp.createVariable('Time', np.int32, 'time')
        Probabilities= tempgrp.createVariable('probabilities', 'f4', ('time','x','y'),zlib=True)
        #Probabilities2 = tempgrp.createVariable('probabilities_11', 'f4', ('time', 'x', 'y'), zlib=True)
        Real_data = tempgrp.createVariable('historic_data', 'i', ('time','x','y'),zlib=True)
        Dist_nested = tempgrp.createVariable('dist_nested', 'f4', ('x','y'),zlib=True)
        TIME[:] = lawr.time[lawr.progStartIdx:lawr.progStartIdx + progTime]
        Probabilities[:,:,:] = lawr.probabilities
        #Probabilities2[:, :, :] = lawr.probabilities2
        Real_data[:, :, :] = real_data
        Dist_nested[:,:] = lawr.dist_nested
        f.close()
        #result=verification(lawr,dwd,year, mon, day, hour, minute,progTime)
        thresholds=[0.1,0.2,0.5,1,2,5,10,20,30]


            #fig, ax = plt.subplots()
            #ax.bar(bin_edges[:-1]-0.1, histX, color='b', width = 0.2)
            #ax.bar(bin_edges[:-1]+0.1, histY, color='r', width = 0.2)
            #plt.show(block=False)
        if 0:
            for t in range(progTime):
                if t == 0:
                    pls, axs = plt.subplots(1,2,figsize=(15,9))
                    ax1=axs[1]
                    prob = axs[0].imshow(lawr.probabilities[t,:,:])
                    cb = plt.colorbar(prob, fraction = 0.046, pad = 0.04, ax=axs[0])
                    imP = ax1.contour(lawr.prog_data[t, :, :],
                                      contours, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=cmap)
                    imR = ax1.imshow(lawr.nested_data[prog + t, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=cmap)
                    radarCircle2 = mpatches.Circle(
                        (int(lawr.probabilities[t, :, :].shape[0] / 2), int(lawr.prog_data[t, :, :].shape[1] / 2)),
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

        print(str(mon) + '_' + str(day) +'_'+str(hour)+ '_' + str(minute)+' finished.')
        print(datetime.now() - startTime)

        #return result
    except Exception as e:
        nan_dummy = np.zeros([progTime, len(rain_thresholds)]) * np.nan

        result = np.nan
        print(str(mon) + '_' + str(day) +'_'+str(hour)+ '_' + str(minute)+' has failed.')
        print(datetime.now() - startTime)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        #return result
year = 2016



rain_thresholds = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30])
num_r = len(rain_thresholds)






months= [5,6,7,8]
#may = [[5,22],[5,23]]
#june = [[6,2],[6,13],[6,14],[6,18],[6,23],[6,24],[6,25]]
#july = [7,21]
#aug = [8,28]
days = [[5,22],[5,23],[6,2],[6,13],[6,25],[7,21],[8,28],[6,14],[6,18],[6,23],[6,24]]
startHour = 0
endHour = 23
minutes=[0,10,20,30,40,50]
hours = np.arange(startHour,endHour)
progTime = 120

dimensions = [len(months), len(days), len(hours),len(minutes),progTime,num_r]


dates = []
for day in days:
    for hour in hours:
        for minute in minutes:
            dates.append([year, day[0],day[1], hour, minute, progTime])
#investigate 13.6 18:20

#dates= []
#for file in filelist:
#    dates.append([2016,int(file.rsplit('_',5)[0]),int(file.rsplit('_', 5)[1]),int(file.rsplit('_', 5)[2]),int(file.rsplit('_', 5)[3])])

dates =[[2016, 6, 24, 3, 30],
 [2016, 6, 13, 12, 0],
 [2016, 6, 14, 9, 30],
 [2016, 6, 14, 22, 50],
 [2016, 6, 14, 11, 10],
 [2016, 7, 21, 18, 20],
 [2016, 6, 13, 0, 50],
 [2016, 8, 28, 18, 0],
 [2016, 6, 25, 3, 20],
 [2016, 6, 14, 12, 50],
 [2016, 8, 28, 4, 20],
 [2016, 8, 28, 2, 50],
 [2016, 6, 13, 8, 30],
 [2016, 6, 14, 17, 30],
 [2016, 6, 23, 22, 20],
 [2016, 6, 13, 4, 0],
 [2016, 5, 23, 1, 50],
 [2016, 8, 28, 4, 0],
 [2016, 7, 21, 2, 30],
 [2016, 6, 18, 22, 50],
 [2016, 5, 23, 16, 0],
 [2016, 6, 14, 22, 40],
 [2016, 5, 23, 14, 40],
 [2016, 7, 21, 22, 30],
 [2016, 7, 21, 15, 30],
 [2016, 6, 13, 6, 30],
 [2016, 5, 22, 21, 0],
 [2016, 6, 14, 10, 30],
 [2016, 6, 13, 10, 40],
 [2016, 8, 28, 3, 0],
 [2016, 8, 28, 3, 10],
 [2016, 6, 24, 18, 0],
 [2016, 6, 25, 6, 20],
 [2016, 6, 13, 15, 30],
 [2016, 6, 25, 11, 20],
 [2016, 6, 25, 3, 10],
 [2016, 6, 13, 9, 20],
 [2016, 6, 13, 9, 0],
 [2016, 6, 23, 19, 40],
 [2016, 5, 23, 17, 50],
 [2016, 6, 2, 18, 40],
 [2016, 6, 18, 16, 30],
 [2016, 5, 22, 12, 50],
 [2016, 6, 13, 4, 10],
 [2016, 7, 21, 6, 40],
 [2016, 7, 21, 16, 10],
 [2016, 6, 13, 16, 30],
 [2016, 6, 24, 6, 50],
 [2016, 6, 24, 12, 30],
 [2016, 7, 21, 17, 10],
 [2016, 5, 23, 12, 40],
 [2016, 6, 14, 18, 10],
 [2016, 6, 2, 17, 40],
 [2016, 6, 13, 10, 50],
 [2016, 6, 14, 17, 40],
 [2016, 6, 25, 20, 30],
 [2016, 7, 21, 4, 50],
 [2016, 6, 23, 16, 20],
 [2016, 6, 24, 19, 30],
 [2016, 6, 24, 4, 10],
 [2016, 5, 22, 22, 30],
 [2016, 6, 25, 4, 20],
 [2016, 5, 22, 18, 0],
 [2016, 6, 24, 16, 50],
 [2016, 6, 24, 13, 0],
 [2016, 6, 2, 18, 30],
 [2016, 5, 23, 4, 20],
 [2016, 6, 14, 9, 0],
 [2016, 6, 14, 0, 40],
 [2016, 8, 28, 19, 30],
 [2016, 6, 14, 20, 50],
 [2016, 6, 23, 19, 30],
 [2016, 6, 13, 22, 0],
 [2016, 6, 24, 7, 30],
 [2016, 8, 28, 15, 0],
 [2016, 6, 2, 18, 20],
 [2016, 6, 2, 10, 0],
 [2016, 6, 14, 9, 50],
 [2016, 6, 18, 17, 0],
 [2016, 6, 13, 5, 50],
 [2016, 6, 13, 17, 50],
 [2016, 6, 24, 0, 40],
 [2016, 6, 25, 19, 0],
 [2016, 6, 13, 22, 10],
 [2016, 6, 25, 13, 10],
 [2016, 6, 13, 9, 10],
 [2016, 6, 25, 7, 40],
 [2016, 6, 25, 10, 50],
 [2016, 6, 25, 8, 20],
 [2016, 6, 18, 19, 10],
 [2016, 6, 25, 15, 30],
 [2016, 6, 24, 16, 40],
 [2016, 6, 2, 20, 10],
 [2016, 6, 2, 20, 20],
 [2016, 6, 18, 19, 40],
 [2016, 6, 24, 2, 40],
 [2016, 7, 21, 16, 20],
 [2016, 6, 24, 14, 0],
 [2016, 6, 2, 8, 10],
 [2016, 7, 21, 19, 0],
 [2016, 5, 22, 19, 40],
 [2016, 6, 25, 0, 30],
 [2016, 6, 25, 4, 40],
 [2016, 6, 18, 15, 50],
 [2016, 5, 23, 2, 20],
 [2016, 6, 25, 18, 0],
 [2016, 6, 13, 15, 20],
 [2016, 6, 25, 16, 40],
 [2016, 6, 13, 6, 10],
 [2016, 5, 23, 3, 10],
 [2016, 6, 25, 17, 20],
 [2016, 6, 13, 18, 30],
 [2016, 6, 14, 21, 30],
 [2016, 6, 25, 19, 30],
 [2016, 6, 25, 2, 20],
 [2016, 6, 24, 7, 10],
 [2016, 6, 13, 21, 20],
 [2016, 5, 22, 20, 30],
 [2016, 6, 14, 3, 30],
 [2016, 6, 25, 6, 10],
 [2016, 6, 2, 9, 20],
 [2016, 5, 23, 17, 0],
 [2016, 7, 21, 18, 30],
 [2016, 8, 28, 18, 50],
 [2016, 6, 18, 20, 20],
 [2016, 6, 23, 11, 30],
 [2016, 6, 24, 16, 20],
 [2016, 5, 23, 14, 0],
 [2016, 5, 22, 22, 50],
 [2016, 6, 14, 19, 10],
 [2016, 6, 14, 17, 0],
 [2016, 6, 14, 16, 50],
 [2016, 6, 13, 11, 30],
 [2016, 6, 2, 8, 50],
 [2016, 7, 21, 19, 20],
 [2016, 6, 14, 21, 40],
 [2016, 6, 23, 16, 30],
 [2016, 6, 2, 9, 30],
 [2016, 6, 14, 9, 10],
 [2016, 6, 25, 21, 40],
 [2016, 7, 21, 15, 10],
 [2016, 6, 13, 1, 10],
 [2016, 7, 21, 18, 0],
 [2016, 6, 13, 0, 30],
 [2016, 6, 25, 22, 40],
 [2016, 8, 28, 17, 50],
 [2016, 5, 23, 4, 30],
 [2016, 6, 24, 6, 0],
 [2016, 7, 21, 4, 0],
 [2016, 6, 24, 5, 50],
 [2016, 6, 13, 20, 10],
 [2016, 6, 24, 18, 30],
 [2016, 5, 23, 14, 50],
 [2016, 6, 13, 17, 20],
 [2016, 6, 14, 21, 20],
 [2016, 6, 25, 4, 10],
 [2016, 6, 24, 17, 40],
 [2016, 6, 13, 5, 0],
 [2016, 8, 28, 6, 0],
 [2016, 6, 13, 10, 30],
 [2016, 6, 25, 14, 30],
 [2016, 6, 24, 4, 30],
 [2016, 6, 13, 3, 10],
 [2016, 6, 14, 12, 10],
 [2016, 6, 13, 22, 50],
 [2016, 6, 24, 18, 40],
 [2016, 7, 21, 9, 40],
 [2016, 6, 25, 14, 40],
 [2016, 6, 14, 2, 20],
 [2016, 7, 21, 17, 20],
 [2016, 6, 25, 6, 0],
 [2016, 6, 14, 18, 50],
 [2016, 6, 23, 20, 40],
 [2016, 5, 23, 3, 50],
 [2016, 5, 23, 0, 50],
 [2016, 6, 18, 20, 10],
 [2016, 5, 23, 4, 10],
 [2016, 6, 13, 16, 50],
 [2016, 5, 23, 15, 50],
 [2016, 5, 23, 13, 0],
 [2016, 6, 13, 17, 0],
 [2016, 6, 13, 2, 50],
 [2016, 6, 13, 16, 10],
 [2016, 5, 23, 15, 0],
 [2016, 6, 18, 17, 10],
 [2016, 6, 25, 5, 30],
 [2016, 6, 24, 19, 0],
 [2016, 6, 14, 20, 10],
 [2016, 6, 18, 20, 30],
 [2016, 6, 18, 17, 20],
 [2016, 6, 13, 3, 20],
 [2016, 5, 23, 19, 30],
 [2016, 6, 25, 21, 10],
 [2016, 6, 24, 17, 10],
 [2016, 5, 23, 3, 20],
 [2016, 6, 23, 17, 10],
 [2016, 6, 14, 12, 40],
 [2016, 6, 13, 7, 30],
 [2016, 6, 14, 14, 30],
 [2016, 5, 22, 20, 0],
 [2016, 6, 2, 10, 20],
 [2016, 6, 14, 19, 40],
 [2016, 6, 2, 11, 30],
 [2016, 6, 2, 16, 30],
 [2016, 5, 23, 13, 30],
 [2016, 6, 14, 1, 30],
 [2016, 6, 13, 18, 20],
 [2016, 8, 28, 13, 40],
 [2016, 5, 23, 3, 0],
 [2016, 8, 28, 3, 50],
 [2016, 6, 14, 11, 0],
 [2016, 6, 24, 11, 30],
 [2016, 6, 25, 4, 50],
 [2016, 6, 14, 17, 50],
 [2016, 5, 22, 8, 10],
 [2016, 6, 14, 8, 10],
 [2016, 6, 25, 0, 40],
 [2016, 6, 13, 15, 40],
 [2016, 6, 25, 7, 20],
 [2016, 8, 28, 4, 40],
 [2016, 6, 24, 22, 20],
 [2016, 6, 23, 22, 10],
 [2016, 6, 2, 19, 0],
 [2016, 6, 25, 2, 10],
 [2016, 6, 2, 17, 50],
 [2016, 6, 13, 9, 50],
 [2016, 6, 25, 19, 20],
 [2016, 6, 13, 7, 0],
 [2016, 7, 21, 19, 10],
 [2016, 6, 18, 22, 30],
 [2016, 6, 24, 1, 0],
 [2016, 6, 2, 20, 30],
 [2016, 6, 14, 0, 30],
 [2016, 6, 14, 3, 0],
 [2016, 6, 24, 15, 20],
 [2016, 6, 24, 22, 10],
 [2016, 6, 25, 12, 30],
 [2016, 6, 25, 11, 10],
 [2016, 6, 13, 21, 0],
 [2016, 6, 13, 20, 30],
 [2016, 8, 28, 14, 30],
 [2016, 6, 24, 7, 0],
 [2016, 6, 14, 3, 10],
 [2016, 6, 25, 12, 20],
 [2016, 8, 28, 15, 40],
 [2016, 6, 18, 20, 40],
 [2016, 6, 24, 16, 30],
 [2016, 6, 25, 17, 40],
 [2016, 6, 14, 8, 0],
 [2016, 6, 14, 8, 40],
 [2016, 6, 25, 10, 40],
 [2016, 8, 28, 5, 20],
 [2016, 6, 25, 11, 30],
 [2016, 5, 23, 17, 20],
 [2016, 6, 14, 2, 30],
 [2016, 7, 21, 16, 40],
 [2016, 5, 23, 15, 40],
 [2016, 6, 2, 7, 30],
 [2016, 6, 25, 11, 50],
 [2016, 6, 13, 19, 20],
 [2016, 6, 24, 9, 0],
 [2016, 6, 24, 6, 20],
 [2016, 6, 24, 6, 30],
 [2016, 6, 23, 19, 50],
 [2016, 8, 28, 12, 30],
 [2016, 6, 14, 3, 20],
 [2016, 6, 14, 16, 20],
 [2016, 6, 14, 14, 20],
 [2016, 5, 23, 13, 20],
 [2016, 5, 23, 2, 0],
 [2016, 6, 14, 15, 30],
 [2016, 8, 28, 10, 10],
 [2016, 6, 14, 14, 40],
 [2016, 6, 2, 21, 0],
 [2016, 5, 23, 18, 0],
 [2016, 6, 24, 4, 50],
 [2016, 6, 23, 16, 0],
 [2016, 8, 28, 2, 40],
 [2016, 6, 24, 9, 20],
 [2016, 6, 14, 1, 10],
 [2016, 7, 21, 5, 20],
 [2016, 6, 18, 16, 40],
 [2016, 5, 23, 15, 30],
 [2016, 5, 22, 19, 50],
 [2016, 5, 23, 9, 30],
 [2016, 6, 24, 3, 50],
 [2016, 5, 23, 10, 0],
 [2016, 6, 14, 11, 30],
 [2016, 6, 2, 18, 0],
 [2016, 8, 28, 13, 20],
 [2016, 7, 21, 17, 0],
 [2016, 8, 28, 14, 10],
 [2016, 8, 28, 18, 10],
 [2016, 6, 25, 9, 10],
 [2016, 6, 24, 2, 30],
 [2016, 6, 13, 4, 20],
 [2016, 6, 23, 20, 20],
 [2016, 6, 18, 18, 0],
 [2016, 8, 28, 3, 40],
 [2016, 6, 2, 15, 40],
 [2016, 6, 23, 21, 0],
 [2016, 5, 22, 19, 30],
 [2016, 6, 23, 21, 40],
 [2016, 6, 14, 13, 50],
 [2016, 5, 23, 17, 10],
 [2016, 5, 23, 14, 10],
 [2016, 6, 13, 2, 10],
 [2016, 8, 28, 1, 10],
 [2016, 6, 2, 22, 10],
 [2016, 5, 22, 20, 20],
 [2016, 6, 18, 18, 30],
 [2016, 6, 24, 11, 10],
 [2016, 6, 23, 21, 10],
 [2016, 6, 18, 18, 10],
 [2016, 6, 13, 22, 20],
 [2016, 6, 13, 17, 40],
 [2016, 6, 23, 22, 0],
 [2016, 6, 25, 4, 30],
 [2016, 6, 13, 6, 20],
 [2016, 8, 28, 13, 30],
 [2016, 6, 13, 3, 40],
 [2016, 7, 21, 18, 50],
 [2016, 6, 13, 10, 0],
 [2016, 6, 13, 17, 10],
 [2016, 7, 21, 17, 50],
 [2016, 6, 18, 22, 40],
 [2016, 7, 21, 10, 50],
 [2016, 6, 18, 17, 30],
 [2016, 6, 25, 7, 0],
 [2016, 5, 23, 0, 40],
 [2016, 5, 22, 19, 20],
 [2016, 6, 24, 11, 40],
 [2016, 6, 18, 8, 40],
 [2016, 6, 24, 12, 50],
 [2016, 6, 25, 1, 0],
 [2016, 6, 24, 13, 50],
 [2016, 6, 2, 8, 20],
 [2016, 6, 13, 18, 10],
 [2016, 6, 23, 22, 40],
 [2016, 6, 25, 21, 30],
 [2016, 6, 25, 6, 30],
 [2016, 6, 2, 18, 50],
 [2016, 6, 2, 17, 20],
 [2016, 6, 25, 2, 30],
 [2016, 6, 13, 20, 40],
 [2016, 6, 13, 8, 40],
 [2016, 5, 22, 21, 30],
 [2016, 6, 13, 14, 50],
 [2016, 6, 25, 19, 50],
 [2016, 6, 13, 5, 30],
 [2016, 8, 28, 14, 40],
 [2016, 6, 25, 22, 0],
 [2016, 6, 25, 8, 50],
 [2016, 6, 25, 10, 0],
 [2016, 5, 23, 1, 10],
 [2016, 7, 21, 15, 50],
 [2016, 5, 22, 19, 10],
 [2016, 6, 13, 21, 50],
 [2016, 6, 13, 21, 30],
 [2016, 5, 23, 3, 40],
 [2016, 5, 23, 4, 40],
 [2016, 5, 22, 20, 10],
 [2016, 6, 24, 8, 30],
 [2016, 6, 25, 13, 30],
 [2016, 8, 28, 5, 10],
 [2016, 6, 14, 10, 0],
 [2016, 5, 23, 17, 40],
 [2016, 6, 25, 3, 50],
 [2016, 5, 23, 19, 20],
 [2016, 6, 23, 16, 10],
 [2016, 6, 13, 9, 30],
 [2016, 6, 25, 8, 30],
 [2016, 6, 14, 2, 0],
 [2016, 6, 14, 1, 20],
 [2016, 6, 13, 15, 50],
 [2016, 6, 2, 8, 0],
 [2016, 6, 24, 9, 40],
 [2016, 6, 25, 5, 0],
 [2016, 6, 23, 17, 0],
 [2016, 6, 2, 21, 50],
 [2016, 6, 13, 19, 0],
 [2016, 6, 13, 16, 0],
 [2016, 6, 25, 20, 20],
 [2016, 6, 2, 7, 10],
 [2016, 6, 24, 9, 10],
 [2016, 6, 24, 4, 40],
 [2016, 6, 13, 19, 40],
 [2016, 6, 25, 2, 40],
 [2016, 6, 24, 18, 20],
 [2016, 6, 14, 13, 0],
 [2016, 6, 2, 9, 10],
 [2016, 6, 13, 15, 0],
 [2016, 6, 2, 15, 30],
 [2016, 6, 13, 22, 40],
 [2016, 6, 2, 7, 40],
 [2016, 6, 14, 10, 10],
 [2016, 6, 24, 5, 40],
 [2016, 6, 13, 9, 40],
 [2016, 6, 18, 16, 50],
 [2016, 6, 14, 12, 0],
 [2016, 6, 25, 3, 40],
 [2016, 6, 24, 21, 50],
 [2016, 6, 2, 17, 10],
 [2016, 6, 2, 21, 10],
 [2016, 8, 28, 15, 50],
 [2016, 8, 28, 18, 20],
 [2016, 6, 25, 20, 50],
 [2016, 6, 24, 3, 10],
 [2016, 6, 13, 4, 40],
 [2016, 6, 24, 13, 40],
 [2016, 6, 24, 11, 0],
 [2016, 6, 25, 2, 0],
 [2016, 6, 25, 12, 50],
 [2016, 6, 25, 5, 50],
 [2016, 5, 23, 1, 0],
 [2016, 6, 25, 22, 10],
 [2016, 6, 14, 15, 20],
 [2016, 6, 14, 3, 50],
 [2016, 6, 24, 8, 0],
 [2016, 6, 13, 1, 0],
 [2016, 6, 18, 17, 50],
 [2016, 8, 28, 16, 10],
 [2016, 6, 13, 1, 20],
 [2016, 6, 25, 5, 40],
 [2016, 8, 28, 2, 30],
 [2016, 6, 24, 13, 20],
 [2016, 6, 14, 20, 30],
 [2016, 6, 25, 10, 30],
 [2016, 6, 18, 10, 40],
 [2016, 6, 25, 11, 40],
 [2016, 6, 24, 18, 50],
 [2016, 6, 2, 15, 50],
 [2016, 6, 14, 13, 10],
 [2016, 6, 2, 19, 20],
 [2016, 6, 24, 6, 10],
 [2016, 8, 28, 4, 10],
 [2016, 6, 24, 14, 10],
 [2016, 6, 25, 17, 50],
 [2016, 6, 25, 18, 30],
 [2016, 6, 13, 19, 50],
 [2016, 6, 14, 20, 20],
 [2016, 6, 14, 8, 20],
 [2016, 6, 13, 5, 40],
 [2016, 6, 25, 22, 50],
 [2016, 5, 23, 2, 40],
 [2016, 5, 23, 20, 30],
 [2016, 6, 2, 6, 30],
 [2016, 6, 25, 6, 40],
 [2016, 6, 25, 22, 20],
 [2016, 6, 13, 1, 30],
 [2016, 6, 14, 16, 10],
 [2016, 6, 13, 11, 50],
 [2016, 6, 25, 17, 10],
 [2016, 6, 24, 16, 10],
 [2016, 6, 25, 12, 0],
 [2016, 5, 23, 4, 50],
 [2016, 6, 13, 18, 40],
 [2016, 6, 24, 17, 20],
 [2016, 6, 13, 17, 30],
 [2016, 6, 24, 22, 50],
 [2016, 6, 18, 10, 0],
 [2016, 6, 25, 15, 0],
 [2016, 6, 2, 8, 40],
 [2016, 5, 23, 14, 20],
 [2016, 6, 25, 16, 50],
 [2016, 6, 25, 9, 30],
 [2016, 6, 14, 18, 0],
 [2016, 8, 28, 5, 30],
 [2016, 6, 14, 4, 40],
 [2016, 7, 21, 18, 40],
 [2016, 6, 24, 22, 30],
 [2016, 5, 23, 18, 20],
 [2016, 6, 14, 20, 0],
 [2016, 5, 23, 18, 30],
 [2016, 6, 23, 22, 50],
 [2016, 6, 13, 6, 50],
 [2016, 6, 14, 16, 40],
 [2016, 7, 21, 14, 40],
 [2016, 6, 2, 21, 30],
 [2016, 6, 25, 9, 40],
 [2016, 6, 25, 18, 10],
 [2016, 6, 24, 7, 20],
 [2016, 6, 24, 7, 40],
 [2016, 6, 18, 19, 20],
 [2016, 6, 25, 1, 40],
 [2016, 5, 23, 18, 40],
 [2016, 6, 14, 0, 50],
 [2016, 6, 25, 13, 20],
 [2016, 6, 25, 6, 50],
 [2016, 5, 22, 22, 20],
 [2016, 7, 21, 19, 30],
 [2016, 5, 22, 22, 40],
 [2016, 6, 25, 9, 0],
 [2016, 6, 13, 21, 40],
 [2016, 6, 2, 16, 50],
 [2016, 6, 13, 1, 40],
 [2016, 6, 2, 16, 20],
 [2016, 6, 14, 19, 50],
 [2016, 6, 25, 7, 10],
 [2016, 5, 23, 16, 30],
 [2016, 6, 14, 11, 20],
 [2016, 5, 23, 16, 40],
 [2016, 6, 2, 16, 10],
 [2016, 6, 24, 8, 20],
 [2016, 5, 23, 16, 50],
 [2016, 6, 2, 7, 20],
 [2016, 6, 25, 3, 30],
 [2016, 8, 28, 14, 50],
 [2016, 6, 25, 9, 20],
 [2016, 6, 25, 18, 40],
 [2016, 8, 28, 12, 50],
 [2016, 6, 24, 5, 30],
 [2016, 6, 24, 9, 30],
 [2016, 6, 2, 19, 50],
 [2016, 6, 25, 12, 40],
 [2016, 7, 21, 18, 10],
 [2016, 6, 24, 5, 10],
 [2016, 6, 14, 20, 40],
 [2016, 6, 18, 21, 10],
 [2016, 6, 13, 2, 0],
 [2016, 5, 22, 20, 50],
 [2016, 5, 23, 20, 0],
 [2016, 8, 28, 13, 10],
 [2016, 6, 23, 5, 10],
 [2016, 5, 23, 14, 30],
 [2016, 7, 21, 16, 50],
 [2016, 6, 13, 10, 20],
 [2016, 5, 23, 5, 0],
 [2016, 6, 14, 17, 10],
 [2016, 6, 24, 3, 40],
 [2016, 6, 23, 21, 50],
 [2016, 6, 13, 0, 40],
 [2016, 7, 21, 17, 40],
 [2016, 6, 2, 19, 30],
 [2016, 6, 24, 8, 40],
 [2016, 5, 23, 13, 50],
 [2016, 6, 25, 2, 50],
 [2016, 6, 2, 10, 40],
 [2016, 6, 2, 21, 20],
 [2016, 6, 14, 9, 20],
 [2016, 6, 24, 6, 40],
 [2016, 6, 2, 9, 0],
 [2016, 6, 25, 1, 50],
 [2016, 6, 13, 2, 40],
 [2016, 6, 13, 4, 50],
 [2016, 6, 2, 12, 40],
 [2016, 6, 13, 6, 40],
 [2016, 6, 25, 7, 50],
 [2016, 6, 24, 11, 50],
 [2016, 7, 21, 2, 20],
 [2016, 6, 13, 20, 20],
 [2016, 5, 23, 18, 50],
 [2016, 6, 14, 5, 20],
 [2016, 6, 14, 2, 40],
 [2016, 6, 14, 12, 30],
 [2016, 6, 2, 9, 40],
 [2016, 6, 24, 22, 0],
 [2016, 6, 14, 18, 20],
 [2016, 6, 23, 16, 40],
 [2016, 6, 18, 18, 20],
 [2016, 6, 25, 19, 10],
 [2016, 5, 23, 0, 30],
 [2016, 6, 13, 18, 50],
 [2016, 6, 2, 15, 10],
 [2016, 6, 13, 8, 50],
 [2016, 6, 14, 10, 20],
 [2016, 6, 14, 15, 10],
 [2016, 6, 13, 22, 30],
 [2016, 6, 18, 18, 50],
 [2016, 6, 13, 14, 40],
 [2016, 5, 22, 20, 40],
 [2016, 6, 13, 5, 20],
 [2016, 6, 24, 12, 10],
 [2016, 6, 24, 8, 50],
 [2016, 6, 23, 22, 30],
 [2016, 6, 13, 5, 10],
 [2016, 6, 23, 17, 30],
 [2016, 6, 23, 21, 30],
 [2016, 5, 23, 6, 0],
 [2016, 6, 2, 21, 40],
 [2016, 6, 24, 7, 50],
 [2016, 6, 2, 22, 0],
 [2016, 6, 13, 3, 30],
 [2016, 6, 14, 8, 30],
 [2016, 6, 25, 12, 10],
 [2016, 6, 14, 9, 40],
 [2016, 6, 13, 7, 10],
 [2016, 6, 23, 20, 30],
 [2016, 5, 23, 6, 10],
 [2016, 6, 25, 20, 10],
 [2016, 5, 23, 3, 30],
 [2016, 7, 21, 16, 30],
 [2016, 6, 14, 14, 0],
 [2016, 5, 23, 20, 40],
 [2016, 6, 2, 18, 10],
 [2016, 8, 28, 14, 0],
 [2016, 6, 18, 19, 50],
 [2016, 6, 25, 19, 40],
 [2016, 6, 25, 13, 40],
 [2016, 6, 25, 10, 20],
 [2016, 6, 25, 13, 50],
 [2016, 5, 23, 5, 50],
 [2016, 6, 25, 14, 50],
 [2016, 6, 25, 8, 10],
 [2016, 7, 21, 3, 10],
 [2016, 6, 25, 20, 0],
 [2016, 6, 25, 10, 10],
 [2016, 6, 13, 2, 20],
 [2016, 6, 13, 7, 50],
 [2016, 7, 21, 9, 0],
 [2016, 8, 28, 12, 20],
 [2016, 6, 25, 7, 30],
 [2016, 6, 25, 13, 0],
 [2016, 6, 25, 14, 20],
 [2016, 6, 23, 17, 40],
 [2016, 8, 28, 17, 40],
 [2016, 6, 24, 1, 20],
 [2016, 6, 24, 13, 10],
 [2016, 5, 23, 1, 20],
 [2016, 6, 24, 5, 20],
 [2016, 6, 25, 8, 40],
 [2016, 6, 23, 16, 50],
 [2016, 6, 24, 17, 30],
 [2016, 5, 23, 5, 20],
 [2016, 6, 13, 7, 40],
 [2016, 6, 14, 18, 30],
 [2016, 5, 23, 15, 20],
 [2016, 6, 24, 8, 10],
 [2016, 6, 18, 20, 0],
 [2016, 6, 13, 15, 10],
 [2016, 6, 2, 10, 50],
 [2016, 6, 25, 11, 0],
 [2016, 5, 22, 21, 10],
 [2016, 6, 14, 16, 30],
 [2016, 6, 14, 1, 40],
 [2016, 6, 13, 7, 20],
 [2016, 6, 13, 11, 10],
 [2016, 6, 14, 14, 50],
 [2016, 6, 14, 12, 20],
 [2016, 6, 13, 11, 0],
 [2016, 5, 22, 21, 40],
 [2016, 8, 28, 15, 30],
 [2016, 6, 13, 19, 30],
 [2016, 6, 13, 3, 50],
 [2016, 5, 23, 2, 10],
 [2016, 6, 25, 1, 30],
 [2016, 6, 14, 19, 20],
 [2016, 6, 25, 21, 50],
 [2016, 6, 14, 18, 40],
 [2016, 6, 25, 18, 50],
 [2016, 6, 13, 8, 0],
 [2016, 6, 24, 0, 30],
 [2016, 6, 14, 8, 50],
 [2016, 6, 14, 17, 20],
 [2016, 6, 13, 19, 10],
 [2016, 6, 25, 14, 0],
 [2016, 6, 13, 20, 50],
 [2016, 8, 28, 17, 20],
 [2016, 5, 23, 5, 10],
 [2016, 8, 28, 4, 30],
 [2016, 6, 25, 0, 50],
 [2016, 6, 13, 11, 40],
 [2016, 6, 24, 17, 0],
 [2016, 6, 18, 18, 40],
 [2016, 5, 23, 16, 20],
 [2016, 7, 21, 11, 50],
 [2016, 6, 25, 4, 0],
 [2016, 6, 14, 19, 30],
 [2016, 6, 23, 20, 0],
 [2016, 5, 23, 19, 40],
 [2016, 8, 28, 13, 50],
 [2016, 5, 23, 1, 40],
 [2016, 6, 25, 8, 0],
 [2016, 6, 23, 19, 10],
 [2016, 6, 25, 3, 0],
 [2016, 6, 24, 13, 30],
 [2016, 6, 13, 10, 10],
 [2016, 6, 2, 20, 40],
 [2016, 6, 13, 18, 0],
 [2016, 6, 18, 19, 30],
 [2016, 8, 28, 15, 20],
 [2016, 8, 28, 13, 0],
 [2016, 6, 18, 17, 40],
 [2016, 5, 22, 22, 0],
 [2016, 6, 14, 3, 40],
 [2016, 8, 28, 17, 0],
 [2016, 6, 24, 18, 10],
 [2016, 8, 28, 3, 20],
 [2016, 6, 14, 15, 50],
 [2016, 6, 2, 15, 20],
 [2016, 6, 14, 15, 40],
 [2016, 6, 14, 10, 40],
 [2016, 6, 25, 21, 0],
 [2016, 6, 2, 8, 30],
 [2016, 6, 25, 17, 30],
 [2016, 7, 21, 17, 30],
 [2016, 6, 23, 20, 50],
 [2016, 6, 14, 13, 20],
 [2016, 6, 14, 15, 0],
 [2016, 6, 13, 6, 0],
 [2016, 8, 28, 1, 20],
 [2016, 6, 23, 19, 0],
 [2016, 7, 21, 14, 0],
 [2016, 6, 25, 21, 20],
 [2016, 6, 13, 20, 0],
 [2016, 6, 25, 15, 20],
 [2016, 6, 24, 17, 50],
 [2016, 6, 14, 16, 0],
 [2016, 6, 14, 1, 50],
 [2016, 8, 28, 16, 40],
 [2016, 6, 2, 19, 40],
 [2016, 6, 23, 20, 10],
 [2016, 6, 2, 16, 40],
 [2016, 6, 14, 11, 50],
 [2016, 6, 24, 21, 10],
 [2016, 6, 2, 6, 50],
 [2016, 6, 13, 16, 20],
 [2016, 8, 28, 18, 30],
 [2016, 6, 24, 2, 50],
 [2016, 6, 14, 21, 0],
 [2016, 5, 23, 19, 10],
 [2016, 5, 23, 18, 10],
 [2016, 6, 14, 1, 0],
 [2016, 6, 24, 4, 20],
 [2016, 5, 23, 5, 40],
 [2016, 8, 28, 2, 0],
 [2016, 6, 24, 21, 40],
 [2016, 6, 13, 21, 10],
 [2016, 6, 18, 19, 0],
 [2016, 5, 23, 17, 30],
 [2016, 6, 25, 14, 10],
 [2016, 8, 28, 14, 20],
 [2016, 6, 23, 21, 20],
 [2016, 6, 25, 20, 40],
 [2016, 6, 2, 15, 0],
 [2016, 6, 13, 11, 20],
 [2016, 6, 14, 6, 50],
 [2016, 6, 24, 14, 20],
 [2016, 6, 18, 9, 20],
 [2016, 6, 2, 20, 0],
 [2016, 8, 28, 16, 20],
 [2016, 6, 14, 2, 10],
 [2016, 6, 2, 17, 30],
 [2016, 6, 2, 16, 0],
 [2016, 6, 14, 4, 30],
 [2016, 6, 24, 5, 0],
 [2016, 5, 23, 2, 50],
 [2016, 5, 23, 4, 0],
 [2016, 6, 2, 20, 50],
 [2016, 6, 2, 10, 30],
 [2016, 8, 28, 16, 0],
 [2016, 6, 2, 7, 50],
 [2016, 6, 2, 9, 50],
 [2016, 6, 14, 14, 10],
 [2016, 8, 28, 15, 10],
 [2016, 6, 2, 12, 20],
 [2016, 6, 14, 10, 50],
 [2016, 5, 23, 19, 0],
 [2016, 5, 23, 2, 30],
 [2016, 6, 13, 3, 0],
 [2016, 6, 25, 1, 10],
 [2016, 6, 25, 1, 20],
 [2016, 5, 22, 22, 10],
 [2016, 6, 13, 4, 30],
 [2016, 6, 25, 18, 20],
 [2016, 5, 23, 15, 10],
 [2016, 7, 21, 4, 20],
 [2016, 8, 28, 16, 30],
 [2016, 6, 13, 1, 50],
 [2016, 8, 28, 16, 50],
 [2016, 5, 23, 19, 50],
 [2016, 6, 18, 9, 10],
 [2016, 5, 23, 5, 30],
 [2016, 6, 13, 8, 10],
 [2016, 6, 14, 13, 40],
 [2016, 6, 14, 2, 50],
 [2016, 5, 23, 16, 10],
 [2016, 5, 23, 1, 30],
 [2016, 6, 14, 19, 0],
 [2016, 6, 14, 11, 40],
 [2016, 6, 23, 18, 10],
 [2016, 6, 24, 22, 40],
 [2016, 6, 14, 13, 30],
 [2016, 8, 28, 5, 0],
 [2016, 6, 13, 8, 20],
 [2016, 6, 2, 19, 10],
 [2016, 6, 2, 7, 0],
 [2016, 6, 14, 21, 10],
 [2016, 6, 2, 10, 10],
 [2016, 6, 2, 17, 0],
 [2016, 6, 13, 2, 30],
 [2016, 8, 28, 4, 50],
 [2016, 6, 25, 22, 30]]

#prognosis([2016,6,24,18,0,120],0)
#prognosis([2016,3,1,20,0,120],0)

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
t = np.arange(len(dates))

startTime = datetime.now()
pool = mp.Pool(4)
pool.starmap(prognosis, zip(dates,t))
pool.close()
pool.join()
print(datetime.now() - startTime)

#np.save('/scratch/local1/radardata/results.npy',results)
# #
