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
    z2rainrate, findRadarSite, getFiles, nesting, booDisplacement, verification,dwdFileSelector,lawrFileSelector,get_Grid, Results
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



        rainThreshold = 0.5
        prog = int(minute*2)
        trainTime = 6
        probabilityFlag = 1
        timeSteps = prog + progTime
        contours = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
        booFileList = sorted(os.listdir(dwdDirectoryPath))
        dwdTime = list((mon,day,hour,prog))

        dwdSelectedFiles = dwdFileSelector(dwdDirectoryPath, dwdTime, trainTime)



        dwd = DWDData(dwdDirectoryPath + '/' + dwdSelectedFiles[0])

        for i,file in enumerate(dwdSelectedFiles[1:]):
            dwd.addTimestep(dwdDirectoryPath + '/' + file)
            #print(file)
        dwd.initial_maxima()
        dwd.find_displacement(0)

        lawrSelectedFiles = lawrFileSelector(lawrDirectoryPath, date)
        lawr = LawrData(lawrDirectoryPath+'/'+lawrSelectedFiles[0])

        for i,file in enumerate(lawrSelectedFiles[1:]):
            lawr.addTimestep(lawrDirectoryPath+'/'+file)

        lawr.setStart(date)
        lawr.initial_maxima()


        if np.any(np.abs([x/10*(dwd.resolution/lawr.resolution) for x in dwd.gaussMeans])<1):
            lawr.progField.deltaT=int(np.ceil(np.min(1/np.min(np.abs([x/10*(dwd.resolution/lawr.resolution) for x in dwd.gaussMeans])))))+1
        lawr.find_displacement(prog)

        if np.any(np.isnan(lawr.covNormAngle)) or lawr.normEqualOneSum<3*len(lawr.progField.activeIds) or len(lawr.progField.activeFields):
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

        dwd.nested_data[:, (dwd.dist_nested > dwd.r.max())] = 0
        lawr.extrapolation(dwd,progTime,prog,probabilityFlag)
        return
year = 2016



rain_thresholds = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30])
num_r = len(rain_thresholds)



progTime = 90



months= [6]
startHour = 0
days =[2,13,14,18,23,24,25]
endHour = 23
minutes=np.arange(0,59,10)
runs = len(minutes)
hours = np.arange(startHour,endHour)

dimensions = [len(months), len(days), len(hours),len(minutes),progTime,num_r]


dates = []
for mon in months:
    for day in days:
        for hour in hours:
            for minute in minutes:
                dates.append([year, mon, day, hour, minute, progTime])
t = np.arange(len(dates))
#investigate 13.6 18:20

#result = prognosis([2016,6,2,7,40,90],0)
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

startTime = datetime.now()
pool = mp.Pool(4)
results = pool.starmap(prognosis, zip(dates,t))
pool.close()
pool.join()
# #
import os
import shutil
path = '/home/zmaw/u231126/radar/public_html/temp_data'
homepath = '/scratch/local1/temp_radar_data'
newdata = 0
for root, dirs, files in os.walk(path):
    params = [[[], []] for i in range(10)]
    for i, filename in enumerate(files):
        # I use absolute path, case you want to move several dirs.
        base, extension = os.path.splitext(filename)
        params[i][0] = filename
        params[i][1] = datetime.datetime.utcfromtimestamp(os.path.getmtime(root + '/' + filename))
while True:
    for root, dirs, files in os.walk(path):
        for i, filename in enumerate(files):
            # I use absolute path, case you want to move several dirs.
            base, extension = os.path.splitext(filename)
            oldname = os.path.join(os.path.abspath(root), filename)
            if ((filename == 'lawr_latest.nc') | (filename == 'dwd_latest.nc')) & (params[i][0] == filename) & ~(
                    params[i][1] == datetime.datetime.utcfromtimestamp(os.path.getmtime(root + '/' + filename))):
                print(filename + ' is changed')
                params[i][0] = filename
                params[i][1] = datetime.datetime.utcfromtimestamp(os.path.getmtime(root + '/' + filename))
                newname = base + '_' + str(params[i][1].minute) + '_' + str(params[i][1].second) + extension
                newpath = os.path.join(homepath, base)
                newname = os.path.join(newpath, newname)
                shutil.copy(oldname, newname)
                print('copied ' + newname)
                newdata = 1
            else:
                params[i][0] = filename
                params[i][1] = datetime.datetime.utcfromtimestamp(os.path.getmtime(root + '/' + filename))

    time.sleep(1)

