from init import DWDData, LawrData, findRadarSite
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
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
    outf = '/home/zmaw/u300675/prognosis.mp4'
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


    lawrDisplacementParameters = []
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


                if (newdata == 1) & (len(os.listdir(lawr_dir)) >= 11) & (len(os.listdir(dwd_dir)) >= 7):
                    try:
                        lawrTraintime = -10
                        dwdTraintime = -6

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

                        lawr.startTime = -lawr.trainTime
                        lawr.initial_maxima()
                        lawr.find_displacement()
                        resFactor = (dwd.resolution / lawr.resolution)  # resolution factor  between dwd and lawr

                        if np.any(np.isnan(lawr.covNormAngle)) or lawr.progField.deltaT > 8 or len(
                                lawr.progField.activeFields)<2:
                            lawr.covNormAngle = np.cov((dwd.progField.return_fieldHistX().flatten() / 10) * resFactor,
                                                       dwd.progField.return_fieldHistY().flatten() / 10 * resFactor)
                            lawr.gaussMeans = [x / 10 * resFactor for x in dwd.gaussMeans]

                        if np.any(np.isnan(dwd.covNormAngle)) or len(dwd.progField.activeFields) < 2:
                            dwd.covNormAngle_norm = np.cov(lawr.progField.return_fieldHistX().flatten() / resFactor,
                                                            lawr.progField.return_fieldHistY().flatten() / resFactor)

                            dwd.gaussMeans_norm = [x/resFactor for x in lawr.gaussMeans]

                        dwd.HHGPos = findRadarSite(lawr, dwd)
                        dwd.set_auxillary_geoData(dwd, lawr, dwd.HHGPos)

                        dwd.extrapolation(progTime+15)
                        lawr.extrapolation(dwd, progTime, 3)
                        lawr.probabilities[:, lawr.dist_nested >= np.max(lawr.r)] = 0
                        if np.sum(dwd.prog_data[:, (dwd.dist_nested >= lawr.r[-1])]) > 50:
                            f=open("/home/zmaw/u300675/py_rainprog/displacement_parameters.txt","a+")
                            f.write(serial_date_to_string(lawr.time[0]+2/24))
                            f.write("\r\n")
                            f.write('LAWR Coviances: ' + str(np.round(lawr.covNormAngle[0, 0], 3)) + " " + str(
                                np.round(lawr.covNormAngle[0, 1], 3)) + " " + str(np.round(lawr.covNormAngle[1, 1], 3)))
                            f.write("\r\n")
                            f.write('LAWR Displacements: ' + str(np.round(lawr.gaussMeans, 3)))
                            f.write("\r\n")
                            f.write('DWD Coviances: ' + str(np.round(dwd.covNormAngle_norm[0, 0], 3)) + " " + str(
                                np.round(dwd.covNormAngle_norm[0, 1], 3)) + " " + str(
                                np.round(dwd.covNormAngle_norm[1, 1], 3)))
                            f.write("\r\n")
                            f.write('DWD Displacements: ' + str(np.round(dwd.gaussMeans_norm, 3) * 2))
                            f.write("\r\n")
                            f.close()
                            #cmdstring = ('ffmpeg',
                            #             '-y', '-r', '5',  # overwrite, 30fps
                            #             '-s', '%dx%d' % (700, 700),  # size of image string
                            #             '-pix_fmt', 'argb',  # format
                            #             '-f', 'rawvideo', '-i', '-', '-b:v', '3M', '-crf', '14',
                            #             # input from pipe, bitrate, compression
                            #             # tell ffmpeg to expect raw video from the pipe
                            #             '-vcodec', 'libx264','mpeg4', outf)  # output encoding
                            plt.ioff()
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
                            anim.save(outf, fps=frameRate,
                                      extra_args=['-vcodec', 'h264',
                                                  '-pix_fmt', 'yuv420p'])
                            plt.close(fig)
                            shutil.copy(outf, '/data/share/u231/pattern_mp4/prognosis.mp4')
                            os.chmod('/data/share/u231/pattern_mp4/prognosis.mp4', 0o755)
                            print('Prognosis successful.')

                            plt.rcParams["figure.figsize"] = (10, 9)
                            plt.rcParams.update({'font.size': 14})

                            from matplotlib.colors import LinearSegmentedColormap
                            import matplotlib.image as mpimg
                            alpha = 0.8
                            gray = 204
                            cols = [(50 / 255, 200 / 255, 0 / 255, .4),
                                    (255 / 255, 255 / 255, 42 / 255, .6),
                                    (254 / 255, 161 / 255, 42 / 255, alpha)]
                            cm = LinearSegmentedColormap.from_list(
                                'test', cols, N=10)
                            test = np.copy(lawr.probabilities)
                            test[:, lawr.dist_nested >= np.max(lawr.r)] = 0
                            test[test > 1] = 1
                            test[test < 0.05] = -0.01
                            fig, axes = plt.subplots(nrows=2, ncols=2)
                            extent = [lawr.Lon_nested.min(), lawr.Lon_nested.max(), lawr.Lat_nested.min(),
                                      lawr.Lat_nested.max()]
                            t = [20, 40, 60, 80]
                            levels = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                            img = mpimg.imread('/home/zmaw/u300675/background_100dpi.png')
                            for i, ax in enumerate(axes.flat):
                                ax.imshow(img, extent=extent)
                                im = ax.contourf(lawr.Lon_nested, lawr.Lat_nested, np.max(test[:t[i], :, :], axis=0),
                                                 levels, cmap=cm)
                                ax.set_title(str(int(t[i] / 2)) + ' Minuten Vorhersage')
                                ax.tick_params(axis='both', which='both', bottom=0, top=0, labelbottom=0, right=0,
                                               left=0, labelleft=0)
                                ax.set_aspect(1.62)
                            fig.subplots_adjust(right=0.8)
                            cbar_ax = fig.add_axes([0.85, 0.123, 0.05, 0.745])
                            cb = fig.colorbar(im, cax=cbar_ax, fraction=0.046, pad=0.04)
                            cb.set_ticklabels((cb.get_ticks() * 100).astype(int))
                            cb.set_label('Regenwahrscheinlichkeit in %')
                            plt.savefig('/home/zmaw/u300675/Overview.jpeg', dpi=150)
                            plt.close(fig)
                            img = mpimg.imread('/home/zmaw/u300675/background.png')

                            plt.rcParams["figure.figsize"] = (10, 9)
                            plt.rcParams.update({'font.size': 14})
                            for i in range(len(t)):
                                fig, ax = plt.subplots(1)
                                ax.imshow(img, extent=extent)
                                im = ax.contourf(lawr.Lon_nested, lawr.Lat_nested, np.max(test[:t[i], :, :], axis=0),
                                                 levels, cmap=cm)
                                ax.set_title(str(int(t[i] / 2)) + ' Minuten Vorhersage')
                                ax.tick_params(axis='both', which='both', bottom=0, top=0, labelbottom=0, right=0,
                                               left=0, labelleft=0)
                                ax.set_aspect(1.65)
                                fig.subplots_adjust(right=0.8)
                                cbar_ax = fig.add_axes([0.85, 0.123, 0.05, 0.745])
                                cb = fig.colorbar(im, cax=cbar_ax, fraction=0.046, pad=0.04)
                                cb.set_ticklabels((cb.get_ticks() * 100).astype(int))
                                cb.set_label('Regenwahrscheinlichkeit in %')
                                plt.savefig('/home/zmaw/u300675/Panel_'+str(int(t[i] / 2)) + '_Mins.jpeg', dpi=150)
                                plt.close(fig)

                            newdata=0
                        else:
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
                            time_text = ax1.text(30, 30, serial_date_to_string(lawr.time[0]+2/24))

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
                            shutil.copy(outf, '/data/share/u231/pattern_mp4/prognosis.mp4')

                            os.chmod('/data/share/u231/pattern_mp4/prognosis.mp4', 0o755)
                            print('Prognosis successful, no rain.')
                            newdata=0

                    except Exception as e:
                        print(e)
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
                        time_text = ax1.text(30, 30, datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))

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
                        shutil.copy(outf, '/data/share/u231/pattern_mp4/prognosis.mp4')

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