import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4

dir = '/scratch/local1/radardata/prognosis/'

filelist = os.listdir(dir)

nc = netCDF4.Dataset(dir + filelist[0])

time = len(nc.groups['Prognosis Data'].variables['Time'][:])
dist_nested = nc.groups['Prognosis Data'].variables['dist_nested'][:][:]
rain_threshold = 0.5
prob_thresholds = np.arange(0, 1.01, 0.1)
num_prob = len(prob_thresholds)
max_dist = 19966.141
inverse_number_of_forecasts = 1/np.sum(dist_nested<max_dist)


hit = np.zeros([len(filelist), time, num_prob])
miss = np.copy(hit)
f_alert = np.copy(hit)
corr_zero = np.copy(hit)
event = np.copy(hit)
nonevent = np.copy(hit)
total = np.copy(hit)
roc_hr = np.copy(hit)
roc_far = np.copy(hit)
briers = np.zeros([len(filelist),time])
prog_data_total = np.zeros([time,np.sum(dist_nested<max_dist)])
real_data_total = np.copy(prog_data_total)
rse = np.copy(real_data_total)
for j,file in enumerate(filelist[0:50]):
    with netCDF4.Dataset(dir + file) as nc:
        try:
            prog_data = nc.groups['Prognosis Data'].variables['probabilities'][:][:][:]
            prog_data = prog_data[:,dist_nested < max_dist]
            prog_data_total += np.nan_to_num(prog_data)
            real_data = nc.groups['Prognosis Data'].variables['historic_data'][:][:][:]
            real_data = real_data[:,dist_nested < max_dist]
            real_data_total += np.nan_to_num(real_data)
            time = nc.groups['Prognosis Data'].variables['Time'][:]
            for i, p in enumerate(prob_thresholds):
                p_dat = prog_data >= p
                r_dat = real_data == 1
                #for t in range(len(time)):
                hit[j, :, i] = np.sum(r_dat[:, :] & p_dat[:, :],axis=1)
                miss[j, :, i] = np.sum(r_dat[:, :] & ~p_dat[:, :],axis=1)
                f_alert[j, :, i] = np.sum(~r_dat[:, :] & p_dat[:, :],axis=1)
                corr_zero[j, :, i] = np.sum(~r_dat[:, :] & ~p_dat[:, :],axis=1)

                event[j, :, i] = hit[j, :, i] + miss[j, :, i]
                nonevent[j, :, i] = f_alert[j, :, i] + corr_zero[j, :, i]
                total[j, :, i] = hit[j, :, i] + miss[j, :, i] + f_alert[j, :, i] + corr_zero[j, :, i]

                roc_hr[j, :, i] = hit[j, :, i] / event[j, :, i]
                roc_far[j, :, i] = f_alert[j, :, i] / nonevent[j, :, i]
            print('Finished ' + file)
            rse += (np.nan_to_num(prog_data)-np.nan_to_num(real_data))**2
            briers[j,:] = inverse_number_of_forecasts*np.sum((prog_data[:,:]-real_data[:,:])**2,axis=1)
        except Exception as e:
            print(e)
    print(str(np.round(j/len(filelist),2)))
print('Finished')
hit_total = np.sum(hit,0)
miss_total = np.sum(miss,0)
f_alert_total=np.sum(f_alert,0)
corr_zero_total = np.sum(corr_zero)


event_total = np.sum(event,0)
nonevent_total = np.sum(nonevent,0)

roc_hr_total = hit_total/event_total
roc_far_total = f_alert_total/nonevent_total

bs = np.mean(briers,axis=0)
bs2= np.sum(rse/len(filelist),axis=1)*inverse_number_of_forecasts

plt.figure()
for t in range(60):
    plt.plot(np.hstack([0, roc_far_total[t, ::-1], 1]), np.hstack([0, roc_hr_total[t, ::-1], 1]),
             color=[1 / 60 * t, 0, 1 - 1 / 60 * t])
plt.xlabel('FAR')
plt.ylabel('HR')

ROC_AREA=np.zeros(60)
for t in range(60):
    ROC_AREA[t]=(np.trapz(np.hstack([0,roc_hr_total[t,::-1],1]),np.hstack([0,roc_far_total[t,::-1],1]),dx=0.1))