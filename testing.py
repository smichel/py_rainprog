from init import DWDData, LawrData, findRadarSite
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

da = sorted(os.listdir('/scratch/local1/temp_radar_data/dwd_latest/'))
ta = sorted(os.listdir('/scratch/local1/temp_radar_data/lawr_dual_latest/'))
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