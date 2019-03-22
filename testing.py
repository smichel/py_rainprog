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

da = sorted(os.listdir('/scratch/local1/temp_radar_data/dwd_latest/'))
ta = sorted(os.listdir('/scratch/local1/temp_radar_data/lawr_dual_latest/'))

os.chdir(dir)
tes = sorted(os.listdir(dir),key=os.path.getmtime)
te = tes[1:-43]
lawr = LawrData('/scratch/local1/temp_radar_data/lawr_dual_latest/'+ta[50])
ta = ta[51:62]
dwd = DWDData('/scratch/local1/temp_radar_data/dwd_latest/dwd_latest_31_16.nc')

for t in range(len(da[6:12])):
    dwd.addTimestep('/scratch/local1/temp_radar_data/dwd_latest/'+da[6+t])
    print(da[6+t])

for t in range(len(ta)):
    lawr.addTimestep('/scratch/local1/temp_radar_data/lawr_dual_latest/'+ta[t])

dwd.initial_maxima()
dwd.find_displacement(0)
dwd.gaussMeans = [10,2]
dwd.extrapolation(50)

lawr.startTime = -lawr.trainTime
lawr.initial_maxima()
lawr.find_displacement(-lawr.trainTime)
lawr.covNormAngle = np.array([[0,0],[0,0]])
lawr.gaussMeans = [x / 10 * (dwd.resolution / lawr.resolution) for x in dwd.gaussMeans]

dwd.HHGPos = findRadarSite(lawr, dwd)
dwd.set_auxillary_geoData(dwd, lawr, dwd.HHGPos)

lawr.extrapolation(dwd, 50, 1)
print('ding')