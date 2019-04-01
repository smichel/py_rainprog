import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from datetime import datetime
from init import reliability_curve

params = {"pgf.texsystem": "pdflatex"}
plt.rcParams.update(params)
dir = '/scratch/local1/radardata/prognosis3/'
days = np.array([[5,22],[5,23],[6,2],[6,13],[6,25],[7,21],[8,28],[6,14],[6,18],[6,23],[6,24]])
col = [np.random.rand(3, len(days))]
filelist = os.listdir(dir)

nc = netCDF4.Dataset(dir + filelist[0])

time = len(nc.groups['Prognosis Data'].variables['Time'][:])
dist_nested = nc.groups['Prognosis Data'].variables['dist_nested'][:][:]
rain_threshold = .5
bin_width = 0.1
prob_thresholds = np.arange(0, 1.01, bin_width)
num_prob = len(prob_thresholds)
max_dist = 19966.141
inverse_number_of_forecasts = 1/np.sum(dist_nested<max_dist)
number_of_forecasts = np.sum(dist_nested<max_dist)
selectedFiles = filelist

hit = np.zeros([len(selectedFiles), time, num_prob])
miss = np.copy(hit)
f_alert = np.copy(hit)
f_alert_bin = np.copy(hit)
corr_zero_bin = np.copy(hit)
bin_total= np.copy(hit)
corr_zero = np.copy(hit)
event = np.copy(hit)
nonevent = np.copy(hit)
total = np.copy(hit)
roc_hr = np.copy(hit)
roc_far = np.copy(hit)
briers = np.zeros([len(selectedFiles),time])
prog_data_total = np.zeros([time,np.sum(dist_nested<max_dist)])
real_data_total = np.copy(prog_data_total)
rse = np.copy(real_data_total)
y_score_bin_mean = np.zeros([len(selectedFiles),time,10])
empirical_prob_pos = np.copy(y_score_bin_mean)
sample_num = np.copy(empirical_prob_pos)
timestamps = np.zeros([len(selectedFiles),time])

for j,file in enumerate(selectedFiles):
    starttime = datetime.now()
    with netCDF4.Dataset(dir + file) as nc:
        try:
            prog_data = nc.groups['Prognosis Data'].variables['probabilities'][:][:][:]
            prog_data = prog_data[:,dist_nested < max_dist]
            prog_data_total += np.nan_to_num(prog_data)
            real_data = nc.groups['Prognosis Data'].variables['historic_data'][:][:][:]
            real_data = real_data[:,dist_nested < max_dist]
            real_data_total += np.nan_to_num(real_data)
            timestamps[j,:] = nc.groups['Prognosis Data'].variables['Time'][:]
            for t in range(time):
                y_score_bin_mean[j,t,:],empirical_prob_pos[j,t,:],sample_num[j,t,:],bins = reliability_curve(real_data[t,:],prog_data[t,:])
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
                bin_idx = np.logical_and(p - bin_width / 2 < prog_data,
                                         prog_data <= p + bin_width / 2)
                #for t in range(len(time)):
                #    f_alert_bin[j,t,i] = np.sum(real_data[t,bin_idx[t,:]])
                #    corr_zero_bin[j,t,i] = np.sum(real_data[t,bin_idx[t,:]]==0)
                #    bin_total[j,t,i] = np.sum(bin_idx[t,:])

            print('Finished ' + file)
            rse += (np.nan_to_num(prog_data)-np.nan_to_num(real_data))**2
            briers[j,:] = inverse_number_of_forecasts*np.sum((prog_data[:,:]-real_data[:,:])**2,axis=1)
        except Exception as e:
            print(e)
    print(str(np.round(j/len(selectedFiles),2)))
    print(datetime.now()-starttime)
print('Finished')
hit_total = np.sum(hit,0)
miss_total = np.sum(miss,0)
f_alert_total=np.sum(f_alert,0)
corr_zero_total = np.sum(corr_zero)


#np.sum(real_data[t,bin_idx[t,:]]==0)/np.sum(bin_idx[t,:])
#np.sum(real_data[t,bin_idx[t,:]])/np.sum(bin_idx[t,:])


event_total = np.sum(event,0)
nonevent_total = np.sum(nonevent,0)

roc_hr_total = hit_total/event_total
roc_far_total = f_alert_total/nonevent_total

bs = np.nanmean(briers,axis=0)
bs2= np.sum(rse/len(selectedFiles),axis=1)*inverse_number_of_forecasts

bsref = (np.mean(event_total[:,0])/(np.mean(event_total[:,0])+np.mean(nonevent_total[:,0])))
bss = 1- bs2 / bsref
y_score_bin_mean_total =np.nansum(y_score_bin_mean[:,-1,:]*sample_num[:,-1,:],axis=0)/np.nansum(sample_num[:,-1,:],axis=0)
plt.rcParams.update({'font.size': 11})
plt.rcParams["figure.figsize"] = (6,4)
day = np.zeros([len(selectedFiles)])
month = np.copy(day)
for i,file in enumerate(selectedFiles):
    day[i] =datetime.fromtimestamp(timestamps[i,0]-2*60*60).day
    month[i] = datetime.fromtimestamp(timestamps[i,0]-2*60*60).month
timearray= np.arange(0,121)

plt.figure()
for t in range(time):
    plt.plot(np.hstack([0, roc_far_total[t, ::-1], 1]), np.hstack([0, roc_hr_total[t, ::-1], 1]),
             color=[1 / time * t, 0, 1 - 1 / time * t])
plt.xlabel('FAR')
plt.ylabel('HR')

fig,ax = plt.subplots(1)
ROC_AREA=np.zeros(time)
for t in range(time):
    ROC_AREA[t]=(np.trapz(np.hstack([0,roc_hr_total[t,::-1],1]),np.hstack([0,roc_far_total[t,::-1],1]),dx=0.1))
ROC_AREA = np.hstack([1,ROC_AREA])
ax.plot(timearray, ROC_AREA)
ax.set_xlim([timearray[0],timearray[-1]])
ax.set_ylim([0.6,ROC_AREA.max()])
ax.set_xticklabels((ax.get_xticks() * 0.5).astype(int))
ax.set_xlabel('Leadtime in minutes')
ax.set_ylabel('ROC Area')
ax.grid()
fig.savefig('/home/zmaw/u300675/ma_rainprog/roc_area.pgf')


fig,ax = plt.subplots(1)
ax.plot(timearray,np.hstack([0,bs2]))
ax.set_xlim([0,timearray[-1]])
ax.set_ylim([0,0.2])
ax.set_xticklabels((ax.get_xticks() * 0.5).astype(int))
ax.set_xlabel('Leadtime in minutes')
ax.grid()
ax.set_ylabel('Brier score')
fig.savefig('/home/zmaw/u300675/ma_rainprog/mean_brier_score.pgf')



fig,ax = plt.subplots(1)
ax.plot(timearray,np.hstack([1,bss]))
ax.set_xlim([0,timearray[-1]])
ax.set_ylim([0,1])
ax.set_xticklabels((ax.get_xticks() * 0.5).astype(int))
ax.set_xlabel('Leadtime in minutes')
ax.grid()
ax.set_ylabel('Brier skill score')
fig.savefig('/home/zmaw/u300675/ma_rainprog/mean_brier_skill_score.pgf')

fig,ax = plt.subplots(1)
for t in range(19,time,20):
    ax.plot(bins,np.nansum(empirical_prob_pos[:,t,:]*sample_num[:,t,:],axis=0)/np.nansum(sample_num[:,t,:],axis=0),color=[1 / time * t, 0, 1 - 1 / time * t])
    ax.annotate(str('+'+str(int((t+1)/2))),xy=(bins[-1],np.nansum(empirical_prob_pos[:,t,-1]*sample_num[:,t,-1],axis=0)/np.nansum(sample_num[:,t,-1],axis=0)-0.02),size = 8)
ax.plot([0,1],'k')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_xlabel('FST')
ax.set_ylabel('OBS')
ax.grid()
fig.savefig('/home/zmaw/u300675/ma_rainprog/mean_rel.pgf')
