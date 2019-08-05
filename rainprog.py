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
import subprocess
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

        lawr.startTime = -lawr.trainTime
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


        outf = 'test.mp4'
        cmdstring = ('ffmpeg',
                     '-y', '-r', '5',  # overwrite, 30fps
                     '-s', '%dx%d' % (700, 700),  # size of image string
                     '-pix_fmt', 'argb',  # format
                     '-f', 'rawvideo', '-i', '-', '-b:v', '3M', '-crf', '14', # input from pipe, bitrate, compression
                     # tell ffmpeg to expect raw video from the pipe
                     '-vcodec', 'mpeg4', outf)  # output encoding


        f = plt.figure(frameon=True, figsize=(7, 7))
        ax1 = f.add_subplot(111)
        p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)
        for t in range(progTime):
            im = lawr.probabilities[t, :, :]
            im[lawr.dist_nested >= np.max(lawr.r)] = 0
            if t == 0:
                imP = ax1.imshow(im,cmap=cmap)
                #plt.show(block=False)
                s1 = plt.colorbar(imP)
                s1.set_clim(0, 1)
                s1.draw_all()
            else:
                imP.set_data(im)
            f.canvas.draw()

            string = f.canvas.tostring_argb()

            p.stdin.write(string)

        p.communicate()
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

result = prognosis([2016,6,2,7,40,90],0)
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

#startTime = datetime.now()
#pool = mp.Pool(4)
#results = pool.starmap(prognosis, zip(dates,t))
#pool.close()
#pool.join()
# # #
from init import DWDData, LawrData, findRadarSite
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib
import os
import datetime
import subprocess
import os
import shutil
import time
import datetime

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'cmaptest',cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('BuPu')
newcmap=truncate_colormap(cmap,0.2,1)
newcmap.set_under('1')
path = '/home/zmaw/u231126/radar/public_html/temp_data'

lawr_dir = '/scratch/local1/temp_radar_data/lawr_dual_latest/'
dwd_dir = '/scratch/local1/temp_radar_data/dwd_latest/'

homepath = '/scratch/local1/temp_radar_data'
newdata = 0
progTime = 50
frameRate = 5
timer = 0
params = [[[], []] for i in range(20)]

def serial_date_to_string(srl_no):
    new_date = datetime.datetime(1970,1,1,0,0) + datetime.timedelta(srl_no - 1)
    return new_date.strftime("%Y-%m-%d-%h-%m-%s")

for i, filename in enumerate(os.listdir(path)):
    # I use absolute path, case you want to move several dirs.
    base, extension = os.path.splitext(filename)
    params[i][0] = filename
    params[i][1] = datetime.datetime.utcfromtimestamp(os.path.getmtime(path+ '/' + filename))
    oldname = os.path.join(os.path.abspath(path), filename)
    if ((filename == 'lawr_dual_latest.nc') | (filename == 'dwd_latest.nc')):
        newname = base + '_' + str(params[i][1].minute) + '_' + str(params[i][1].second) + extension
        newpath = os.path.join(homepath, base)
        newname = os.path.join(newpath, newname)
        shutil.copy(oldname, newname)
        print('copied ' + newname)
while True:
    for i, filename in enumerate(os.listdir(path)):
        # I use absolute path, case you want to move several dirs.
        base, extension = os.path.splitext(filename)
        oldname = os.path.join(os.path.abspath(path), filename)
        if ((filename == 'lawr_dual_latest.nc') | (filename == 'dwd_latest.nc')) & (params[i][0] == filename) & ~(
                params[i][1] == datetime.datetime.utcfromtimestamp(os.path.getmtime(path + '/' + filename))):
            print(filename + ' is changed')
            params[i][0] = filename
            params[i][1] = datetime.datetime.utcfromtimestamp(os.path.getmtime(path+ '/' + filename))
            newname = base + '_' + str(params[i][1].minute).zfill(2) + '_' + str(params[i][1].second).zfill(2) + extension
            newpath = os.path.join(homepath, base)
            newname = os.path.join(newpath, newname)
            shutil.copy(oldname, newname)
            print('copied ' + newname)
            newdata = 1
            timer += 1

        else:
            params[i][0] = filename
            params[i][1] = datetime.datetime.utcfromtimestamp(os.path.getmtime(path + '/' + filename))


        if (newdata == 1) & (len(os.listdir(lawr_dir)) > 10) & (len(os.listdir(dwd_dir)) > 8):
            try:
                lawrTraintime = -10
                dwdTraintime = -8

                os.chdir(lawr_dir)
                lawr_files = sorted(os.listdir(lawr_dir), key=os.path.getmtime)

                os.chdir(dwd_dir)
                dwd_files= sorted(os.listdir(dwd_dir), key=os.path.getmtime)

                lawr = LawrData(lawr_dir+lawr_files[lawrTraintime])
                dwd = DWDData(dwd_dir+dwd_files[dwdTraintime])

                for t in range(len(lawr_files[lawrTraintime + 1:])):
                    try:
                        lawr.addTimestep(lawr_dir + lawr_files[lawrTraintime + 1 + t])
                    except:
                        print(lawr_dir + lawr_files[lawrTraintime + 1 + t]+' is faulty.')

                for t in range(len(dwd_files[dwdTraintime + 1:])):
                    try:
                        dwd.addTimestep(dwd_dir + dwd_files[dwdTraintime + 1 + t])
                    except:
                        print(dwd_dir + dwd_files[dwdTraintime + 1 + t] + ' is faulty.')

                dwd.initial_maxima()
                dwd.find_displacement()
                dwd.covNormAngle_norm = np.cov(dwd.progField.return_fieldHistX().flatten() / 10,
                                               dwd.progField.return_fieldHistY().flatten() / 10)
                dwd.gaussMeans_norm = [x / 10 for x in dwd.gaussMeans]

                dwd.extrapolation(progTime+15)
                lawr.startTime = -lawr.trainTime
                lawr.initial_maxima()
                lawr.find_displacement()

                dwd.HHGPos = findRadarSite(lawr, dwd)
                dwd.set_auxillary_geoData(dwd, lawr, dwd.HHGPos)

                lawr.extrapolation(dwd, progTime, 3)
                lawr.probabilities[:, lawr.dist_nested >= np.max(lawr.r)] = 0

                outf = '/data/share/u231/pattern_mp4/prognosis.mp4'
                #cmdstring = ('ffmpeg',
                #             '-y', '-r', '5',  # overwrite, 30fps
                #             '-s', '%dx%d' % (700, 700),  # size of image string
                #             '-pix_fmt', 'argb',  # format
                #             '-f', 'rawvideo', '-i', '-', '-b:v', '3M', '-crf', '14',
                #             # input from pipe, bitrate, compression
                #             # tell ffmpeg to expect raw video from the pipe
                #             '-vcodec', 'libx264','mpeg4', outf)  # output encoding
                fig = plt.figure()
                fig.set_dpi(100)
                fig.set_size_inches(7, 7)
                ax1 = fig.add_subplot(111)
                fig.gca().set_axis_off()
                ax1.margins(0, 0)
                ax1.axis('off')
                fig.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
                ax1.set_frame_on(False)
                imP = ax1.imshow(lawr.probabilities[0, :, :], cmap=plt.get_cmap('BuPu', 10))
                time_text = ax1.text(30, 30, serial_date_to_string(lawr.time[0]))


                def animate(i):
                    imP.set_data(lawr.probabilities[i, :, :])
                    time_text.set_text(serial_date_to_string(lawr.time[0]))
                    return [imP]


                anim = animation.FuncAnimation(fig, animate,
                                               frames=len(lawr.probabilities),
                                               interval=200, repeat=1,
                                               blit=True)
                anim.save(outf, fps=5,
                          extra_args=['-vcodec', 'h264',
                                      '-pix_fmt', 'yuv420p'])
                plt.close(fig)

                os.chmod('/data/share/u231/pattern_mp4/prognosis.mp4', 0o755)

                newdata=0


            except Exception as e:
                print(e)
                outf = '/data/share/u231/pattern_mp4/prognosis.mp4'
                #cmdstring = ('ffmpeg',
                #             '-y', '-r', '5',  # overwrite, 30fps
                #             '-s', '%dx%d' % (700, 700),  # size of image string
                #             '-pix_fmt', 'argb',  # format
                #             '-f', 'rawvideo', '-i', '-', '-b:v', '3M', '-crf', '14',
                #             # input from pipe, bitrate, compression
                #             # tell ffmpeg to expect raw video from the pipe
                #             '-vcodec', 'libx264','mpeg4', outf)  # output encoding
                fig = plt.figure()
                fig.set_dpi(100)
                fig.set_size_inches(7, 7)
                ax1 = fig.add_subplot(111)
                fig.gca().set_axis_off()
                ax1.margins(0, 0)
                ax1.axis('off')
                fig.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
                ax1.set_frame_on(False)
                norain = np.zeros([2,441,441])
                imP = ax1.imshow(norain[0, :, :], cmap=plt.get_cmap('BuPu', 10))

                def animate(i):
                    imP.set_data(norain[i, :, :])
                    return [imP]


                anim = animation.FuncAnimation(fig, animate,
                                               frames=len(norain),
                                               interval=200,repeat=1,
                                               blit=True)
                anim.save(outf, fps=1,
                          extra_args=['-vcodec', 'h264',
                                      '-pix_fmt', 'yuv420p'])
                plt.close(fig)

                os.chmod('/data/share/u231/pattern_mp4/prognosis.mp4', 0o755)
                print('Prognosis failed')

    if timer%10==0:
        now = time.time()
        for f in os.listdir(lawr_dir):

            f = os.path.join(lawr_dir, f)
            if os.stat(f).st_mtime < now - 7200:

                if os.path.isfile(f):
                    os.remove(os.path.join(lawr_dir, f))
        for f in os.listdir(dwd_dir):

            f = os.path.join(dwd_dir, f)
            if os.stat(f).st_mtime < now - 7200:

                if os.path.isfile(f):
                    os.remove(os.path.join(dwd_dir, f))
        if timer > 10000:
            timer = 0
    time.sleep(1)

import matplotlib.pyplot as plt
import numpy as np

test = np.random.random([20,50,50])
cmdstring = ('ffmpeg',
             '-f', 'rawvideo',
             '-s', '%dx%d' % (700, 700),  # size of image string
             '-i', '-',             # tell ffmpeg to expect raw video from the pipe
             '-pix_fmt', 'yuv420p',  # format
             '-y', # overwrite, 30fps
             '-r', '5', #'-crf', '0',
             # input from pipe, bitrate, compression

             'test.mp4')  # output encoding '-b:v', '5M',

f = plt.figure(frameon=True, figsize=(7, 7))
ax1 = f.add_subplot(111)
for t in range(len(test)):
    im = test[t, :, :]
    if t == 0:
        imP = ax1.imshow(imyuv, cmap=plt.get_cmap('BuPu', 10))
        ax1.axis('off')
        # plt.show(block=False)
    else:
        imP.set_data(im)
    f.canvas.draw()

    string = f.canvas.tostring_argb()

    p.stdin.write(string)

p.communicate()

import os
import shutil
import time
import datetime
path = '/home/zmaw/u231126/radar/public_html/temp_data'
homepath = '/scratch/local1/temp_radar_data'
newdata = 0
newprogflag = 1
while True:

    if newprogflag:
        HHGTime = datetime.datetime.utcfromtimestamp(os.path.getmtime(path + '/' + 'lawr_dual_latest.nc'))
        DWDTime = datetime.datetime.utcfromtimestamp(os.path.getmtime(path + '/' + 'dwd_latest.nc'))
        lawr = LawrData(path + '/' + 'lawr_dual_latest.nc')
        dwd = DWDData(path + '/' + 'dwd_latest.nc')
        newprogflag = 0
        print('Started new prognosis')

    if datetime.datetime.utcfromtimestamp(os.path.getmtime(path + '/' + 'lawr_dual_latest.nc'))!=HHGTime:

        lawr.addTimestep(path + '/' + 'lawr_dual_latest.nc')
        HHGTime = datetime.datetime.utcfromtimestamp(os.path.getmtime(path + '/' + 'lawr_dual_latest.nc'))
        print('added new lawr data')

    if datetime.datetime.utcfromtimestamp(os.path.getmtime(path + '/' + 'dwd_latest.nc'))!=DWDTime:
        dwd.addTimestep(path + '/' + 'dwd_latest.nc')
        DWDTime = datetime.datetime.utcfromtimestamp(os.path.getmtime(path + '/' + 'dwd_latest.nc'))
        print('added new dwd data')

    if (len(lawr.nested_data)>8) & (len(dwd.nested_data)>10):
        dwd.initial_maxima()
        dwd.find_displacement(0)
        dwd.extrapolation(60)

        lawr.startTime = -lawr.trainTime
        lawr.initial_maxima()
        if np.any(np.abs([x/10*(dwd.resolution/lawr.resolution) for x in dwd.gaussMeans])<1):
            lawr.progField.deltaT=int(np.ceil(np.min(1/np.min(np.abs([x/10*(dwd.resolution/lawr.resolution) for x in dwd.gaussMeans])))))+1
        lawr.find_displacement(-lawr.trainTime)

        if np.any(np.isnan(lawr.covNormAngle)) or lawr.normEqualOneSum<3*len(lawr.progField.activeIds) or len(lawr.progField.activeFields):
            lawr.covNormAngle = dwd.covNormAngle
            lawr.gaussMeans = [x/10*(dwd.resolution/lawr.resolution) for x in dwd.gaussMeans]

        if np.any(np.isnan(dwd.covNormAngle)):
            dwd.covNormAngle = lawr.covNormAngle
            dwd.gaussMeans = [x*10/(dwd.resolution/lawr.resolution) for x in lawr.gaussMeans]

        progTime = 60
        dwd.extrapolation(progTime +12)
        dwd.HHGPos = findRadarSite(lawr,dwd)
        dwd.set_auxillary_geoData(dwd,lawr,dwd.HHGPos)
        dwd.nested_data[:, (dwd.dist_nested > dwd.r.max())] = 0
        lawr.extrapolation(dwd,progTime, 1, 7)

        contours = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]

        outf = 'test.mp4'
        cmdstring = ('ffmpeg',
                     '-y', '-r', '5',  # overwrite, 30fps
                     '-s', '%dx%d' % (700, 700),  # size of image string
                     '-pix_fmt', 'argb',  # format
                     '-f', 'rawvideo', '-i', '-', '-b:v', '3M', '-crf', '14', # input from pipe, bitrate, compression
                     # tell ffmpeg to expect raw video from the pipe
                     '-vcodec', 'mpeg4', outf)  # output encoding


        f = plt.figure(frameon=True, figsize=(7, 7))
        ax1 = f.add_subplot(111)
        p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)
        for t in range(progTime):
            im = lawr.probabilities[t, :, :]
            im[lawr.dist_nested >= np.max(lawr.r)] = 0
            if t == 0:
                imP = ax1.imshow(im,cmap=cmap)
                #plt.show(block=False)
                s1 = plt.colorbar(imP)
                s1.set_clim(0, 1)
                s1.draw_all()
            else:
                imP.set_data(im)
            f.canvas.draw()

            string = f.canvas.tostring_argb()

            p.stdin.write(string)

        p.communicate()
        newprogflag=1
        print('issued new prognosis')

    time.sleep(1)

cmdstring = ('ffmpeg',
             '-y', '-r', '5',  # overwrite, 30fps
             '-s', '%dx%d' % (1000, 1000),  # size of image string
             '-pix_fmt', 'argb',  # format
             '-f', 'rawvideo', '-i', '-', '-b:v', '3M', '-crf', '14',  # input from pipe, bitrate, compression
             # tell ffmpeg to expect raw video from the pipe
             '-vcodec', 'mpeg4', outf)  # output encoding
f = plt.figure(frameon=True, figsize=(10, 10))
ax1 = f.add_subplot(111)
p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)
for t in range(len(ta)):
    im = lawr.nested_data[t, :, :]
    im[lawr.dist_nested >= np.max(lawr.r)] = 0
    if t == 0:
        imP = ax1.imshow(im, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=newcmap)
        # plt.show(block=False)
        s = plt.colorbar(imP, format=matplotlib.ticker.ScalarFormatter())
        s.set_label('Precipitation in mm/h')
        s.set_clim(0.1, 100)
        s.set_ticks(contours)
        s.draw_all()

    else:
        imP.set_data(im)
    plt.title((datetime.datetime(1970, 1, 1, 0, 0) + datetime.timedelta(lawr.time[t])).ctime())
    f.canvas.draw()

    string = f.canvas.tostring_argb()

    p.stdin.write(string)

p.communicate()




plt.rcParams["figure.figsize"] = (10, 9)
plt.rcParams.update({'font.size': 14})

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
alpha = 0.8
gray = 204
cols=[(50/255,200/255,0/255,.4),
      (255/255,255/255,42/255,.6),
      (254/255,161/255,42/255,alpha)]
cm = LinearSegmentedColormap.from_list(
    'test', cols, N=10)
test = np.copy(lawr.probabilities)
test[:, lawr.dist_nested >= np.max(lawr.r)] = 0
test[test > 1] = 1
test[test<0.05] =-0.01
fig, axes = plt.subplots(nrows=2, ncols=2)
extent = [lawr.Lon_nested.min(), lawr.Lon_nested.max(), lawr.Lat_nested.min(), lawr.Lat_nested.max()]
t = [20, 40, 60, 80]
levels = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
img = mpimg.imread('/home/zmaw/u300675/background_100dpi.png')
for i, ax in enumerate(axes.flat):
    ax.imshow(img,extent=extent)
    im = ax.contourf(lawr.Lon_nested,lawr.Lat_nested,np.max(test[:t[i], :, :], axis=0), levels, cmap=cm)
    ax.set_title(str(int(t[i] / 2)) + ' Minuten Vorhersage')
    ax.tick_params(axis='both', which='both', bottom=0, top=0, labelbottom=0, right=0, left=0, labelleft=0)
    ax.set_aspect(1.62)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.123, 0.05, 0.745])
cb = fig.colorbar(im, cax=cbar_ax, fraction=0.046, pad=0.04)
cb.set_ticklabels((cb.get_ticks() * 100).astype(int))
cb.set_label('Regenwahrscheinlichkeit in %')
plt.savefig('/home/zmaw/u300675/Overview.jpeg',dpi=150)

img = mpimg.imread('/home/zmaw/u300675/background.png')

plt.rcParams["figure.figsize"] = (10, 9)
plt.rcParams.update({'font.size': 14})
for i in range(len(t)):
    fig, ax = plt.subplots(1)
    ax.imshow(img,extent=extent)
    im = ax.contourf(lawr.Lon_nested,lawr.Lat_nested,np.max(test[:t[i], :, :], axis=0), levels, cmap=cm)
    ax.set_title(str(int(t[i] / 2)) + ' Minuten Vorhersage')
    ax.tick_params(axis='both', which='both', bottom=0, top=0, labelbottom=0, right=0, left=0, labelleft=0)
    ax.set_aspect(1.65)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.123, 0.05, 0.745])
    cb = fig.colorbar(im, cax=cbar_ax, fraction=0.046, pad=0.04)
    cb.set_ticklabels((cb.get_ticks() * 100).astype(int))
    cb.set_label('Regenwahrscheinlichkeit in %')