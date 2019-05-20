from init import DWDData, LawrData, findRadarSite
from createblob import createblob
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


def serial_date_to_string(srl_no):
    new_date = datetime.datetime(1970,1,1,0,0) + datetime.timedelta(srl_no)
    return new_date.strftime("%m/%d/%Y, %H:%M:%S")

def main():
    newcmap = truncate_colormap(cmap, 0.2, 1)
    newcmap.set_under('1')
    path = '/home/zmaw/u231126/radar/public_html/temp_data'

    lawr_dir = '/scratch/local1/temp_radar_data/lawr_dual_latest/'
    dwd_dir = '/scratch/local1/temp_radar_data/dwd_latest/'

    homepath = '/scratch/local1/temp_radar_data'
    newdata = 0
    progTime = 120
    frameRate = 5
    timer = 0
    params = [[[], []] for i in range(20)]
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
        try:
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


                if (newdata == 1) & (len(os.listdir(lawr_dir)) > 11) & (len(os.listdir(dwd_dir)) > 9):
                    try:
                        lawrTraintime = -10
                        dwdTraintime = -8

                        os.chdir(lawr_dir)
                        lawr_files = sorted(os.listdir(lawr_dir), key=os.path.getmtime)

                        os.chdir(dwd_dir)
                        dwd_files= sorted(os.listdir(dwd_dir), key=os.path.getmtime)

                        lawr = LawrData(lawr_dir+lawr_files[lawrTraintime-1])
                        dwd = DWDData(dwd_dir+dwd_files[dwdTraintime-1])

                        for t in range(len(lawr_files[lawrTraintime:])):
                            try:
                                lawr.addTimestep(lawr_dir + lawr_files[lawrTraintime + t])
                            except:
                                print(lawr_dir + lawr_files[lawrTraintime + t]+' is faulty.')

                        for t in range(len(dwd_files[dwdTraintime:])):
                            try:
                                dwd.addTimestep(dwd_dir + dwd_files[dwdTraintime + t])
                            except:
                                print(dwd_dir + dwd_files[dwdTraintime + t ] + ' is faulty.')

                        dwd.initial_maxima()
                        dwd.find_displacement()
                        dwd.covNormAngle_norm = np.cov(dwd.progField.return_fieldHistX().flatten() / 10,
                                                       dwd.progField.return_fieldHistY().flatten() / 10)
                        dwd.gaussMeans_norm = [x / 10 for x in dwd.gaussMeans]

                        dwd.extrapolation(progTime+15)

                        lawr.startTime = -lawr.trainTime
                        lawr.initial_maxima()
                        lawr.find_displacement()
                        resFactor = (dwd.resolution / lawr.resolution)  # resolution factor  between dwd and lawr

                        if np.any(np.isnan(lawr.covNormAngle)) or lawr.progField.deltaT > 8 or len(
                                lawr.progField.activeFields):
                            lawr.covNormAngle = np.cov((dwd.progField.return_fieldHistX().flatten() / 10) * resFactor,
                                                       dwd.progField.return_fieldHistY().flatten() / 10 * resFactor)
                            lawr.gaussMeans = [x / 10 * resFactor for x in dwd.gaussMeans]

                        dwd.HHGPos = findRadarSite(lawr, dwd)
                        dwd.set_auxillary_geoData(dwd, lawr, dwd.HHGPos)

                        lawr.extrapolation(dwd, progTime, 3)
                        lawr.probabilities[:, lawr.dist_nested >= np.max(lawr.r)] = 0

                        if np.sum(dwd.prog_data[:, (dwd.dist_nested >= lawr.r[-1])]) >  50:

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
                            ax1.invert_yaxis()
                            fig.gca().set_axis_off()
                            ax1.margins(0, 0)
                            ax1.axis('off')
                            fig.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                                hspace=0, wspace=0)
                            ax1.set_frame_on(False)
                            imP = ax1.imshow(np.flipud(lawr.probabilities[0, :, :]), cmap=plt.get_cmap('BuPu', 10))
                            imP.set_clim(0,1)
                            time_text = ax1.text(30, 30, serial_date_to_string(lawr.time[0]+2/24))


                            def animate(i):
                                imP.set_data(np.flipud(lawr.probabilities[i, :, :]))
                                time_text.set_text(serial_date_to_string(lawr.time[i]+2/24))
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
                            print('Prognosis successful.')

                            newdata=0
                        else:
                            outf = '/data/share/u231/pattern_mp4/prognosis.mp4'
                            # cmdstring = ('ffmpeg',
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
                            norain = np.zeros([2, 441, 441])
                            imP = ax1.imshow(norain[0, :, :], cmap=plt.get_cmap('BuPu', 10))
                            norain_text = ax1.text(200, 220, 'No precipitation')

                            def animate(i):
                                imP.set_data(norain[i, :, :])
                                return [imP]

                            anim = animation.FuncAnimation(fig, animate,
                                                           frames=len(norain),
                                                           interval=200, repeat=1,
                                                           blit=True)
                            anim.save(outf, fps=1,
                                      extra_args=['-vcodec', 'h264',
                                                  '-pix_fmt', 'yuv420p'])
                            plt.close(fig)

                            os.chmod('/data/share/u231/pattern_mp4/prognosis.mp4', 0o755)
                            print('Prognosis successful, no rain.')

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
                        norain_text = ax1.text(160, 220, 'Failed to create prognosis or no precipitation')
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


        except Exception as e:
            print(e)


if __name__ == '__main__':

    main()