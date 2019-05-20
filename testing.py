from init import DWDData, LawrData, findRadarSite
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib
import os
import datetime
import subprocess

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'cmaptest',cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('BuPu')
newcmap=truncate_colormap(cmap,0.2,1)
newcmap.set_under('1')
contours = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]

dir = '/scratch/local1/temp_radar_data/lawr_single_attcorr_latest/'

lawr_dir = '/scratch/local1/temp_radar_data/lawr_dual_latest/'
os.chdir(lawr_dir)
ta = sorted(os.listdir(lawr_dir),key=os.path.getmtime)
dwd_dir = '/scratch/local1/temp_radar_data/dwd_latest/'
os.chdir(dwd_dir)
da = sorted(os.listdir(dwd_dir),key=os.path.getmtime)


lawr = LawrData('/scratch/local1/temp_radar_data/lawr_dual_latest/'+ta[0])
dwd = DWDData('/scratch/local1/temp_radar_data/dwd_latest/'+da[0])

for t in range(len(da[1:])):
    dwd.addTimestep('/scratch/local1/temp_radar_data/dwd_latest/'+da[t+1])

for t in range(len(ta[1:])):
    lawr.addTimestep('/scratch/local1/temp_radar_data/lawr_dual_latest/'+ta[t+1])

dwd.initial_maxima()
dwd.find_displacement()
#dwd.gaussMeans = [10,2]
dwd.covNormAngle_norm = np.cov(dwd.progField.return_fieldHistX().flatten() / 10,
                                           dwd.progField.return_fieldHistY().flatten() / 10)
dwd.gaussMeans_norm = [x/10 for x in dwd.gaussMeans]
dwd.extrapolation(50+15)

lawr.startTime = -10
lawr.initial_maxima()
lawr.find_displacement()

dwd.HHGPos = findRadarSite(lawr, dwd)
dwd.set_auxillary_geoData(dwd, lawr, dwd.HHGPos)

lawr.extrapolation(dwd, 50, 1)
lawr.probabilities[:, lawr.dist_nested >= np.max(lawr.r)] = 0
print('ding')