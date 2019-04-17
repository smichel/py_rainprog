import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from datetime import datetime
from init import reliability_curve

params = {"pgf.texsystem": "pdflatex"}
plt.rcParams.update(params)
dir = '/scratch/local1/radardata/prognosis12_variance3/'
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
reliability = np.copy(briers)
resolution = np.copy(briers)
uncertainty = np.copy(briers)
y_score_bin_mean = np.zeros([len(selectedFiles),time,len(prob_thresholds)])
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

            rel_prob = np.zeros([len(prob_thresholds),time])
            res_prob = np.copy(rel_prob)
            overall_frequency = np.mean(real_data,axis=1)

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
                #for t in range(len(time)):
                #    f_alert_bin[j,t,i] = np.sum(real_data[t,bin_idx[t,:]])
                #    corr_zero_bin[j,t,i] = np.sum(real_data[t,bin_idx[t,:]]==0)
                #    bin_total[j,t,i] = np.sum(bin_idx[t,:])
                prog_bin_idx = np.logical_and(p - bin_width / 2 < prog_data,
                                              prog_data <= p + bin_width / 2)

                prog_sum_of_forecasts = np.nansum(prog_bin_idx, axis=1)
                for t in range(120):
                    rel_prob[i, t] = prog_sum_of_forecasts[t] * (p - np.nanmean(real_data[t, prog_bin_idx[t, :]])) ** 2
                    res_prob[i, t] = prog_sum_of_forecasts[t] * (
                            np.nanmean(real_data[t, prog_bin_idx[t, :]]) - overall_frequency[t]) ** 2

            reliability[j, :] = inverse_number_of_forecasts * np.nansum(rel_prob, axis=0)
            resolution[j, :] = inverse_number_of_forecasts * np.nansum(res_prob, axis=0)
            uncertainty[j, :] = overall_frequency * (1 - overall_frequency)

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

bsref = (np.mean(event_total[:,0])/(np.mean(event_total[:,0])+np.mean(nonevent_total[:,0])))
bs = np.nanmean(briers[np.nanmax(event[:,:,0],axis=1)>1000],axis=0)
bs2= np.sum(rse/len(selectedFiles),axis=1)*inverse_number_of_forecasts

bss = 1- bs2 / bsref
y_score_bin_mean_total =np.nansum(empirical_prob_pos[:,-1,:]*sample_num[:,-1,:],axis=0)/np.nansum(sample_num[:,-1,:],axis=0)






dir2 = '/scratch/local1/radardata/prognosis12_variance1/'
filelist2 = os.listdir(dir2)

nc = netCDF4.Dataset(dir2 + filelist2[0])
time2 = len(nc.groups['Prognosis Data'].variables['Time'][:])
dist_nested2 = nc.groups['Prognosis Data'].variables['dist_nested'][:][:]
rain_threshold = .5
bin_width = 0.1
prob_thresholds = np.arange(0, 1.01, bin_width)
num_prob = len(prob_thresholds)
max_dist = 19966.141
inverse_number_of_forecasts = 1/np.sum(dist_nested<max_dist)
number_of_forecasts = np.sum(dist_nested<max_dist)
selectedFiles2 = filelist2

hit2 = np.zeros([len(selectedFiles2), time, num_prob])
miss2 = np.copy(hit2)
f_alert2 = np.copy(hit2)
f_alert_bin2 = np.copy(hit2)
corr_zero_bin2 = np.copy(hit2)
bin_total2= np.copy(hit2)
corr_zero2 = np.copy(hit2)
event2 = np.copy(hit2)
nonevent2 = np.copy(hit2)
total2 = np.copy(hit2)
roc_hr2 = np.copy(hit2)
roc_far2 = np.copy(hit2)
briers2 = np.zeros([len(selectedFiles2),time])
prog_data_total2 = np.zeros([time,np.sum(dist_nested<max_dist)])
real_data_total2 = np.copy(prog_data_total2)
rse2 = np.copy(real_data_total2)
reliability2 = np.copy(briers2)
resolution2 = np.copy(briers2)
uncertainty2 = np.copy(briers2)
y_score_bin_mean2 = np.zeros([len(selectedFiles2),time,len(prob_thresholds)])
empirical_prob_pos2 = np.copy(y_score_bin_mean2)
sample_num2 = np.copy(empirical_prob_pos2)
timestamps2 = np.zeros([len(selectedFiles2),time])

for j,file in enumerate(selectedFiles2):
    starttime = datetime.now()
    with netCDF4.Dataset(dir2 + file) as nc:
        try:
            prog_data = nc.groups['Prognosis Data'].variables['probabilities'][:][:][:]
            prog_data = prog_data[:,dist_nested < max_dist]
            prog_data_total2 += np.nan_to_num(prog_data)
            real_data = nc.groups['Prognosis Data'].variables['historic_data'][:][:][:]
            real_data = real_data[:,dist_nested < max_dist]
            real_data_total2 += np.nan_to_num(real_data)
            timestamps2[j,:] = nc.groups['Prognosis Data'].variables['Time'][:]

            rel_prob = np.zeros([len(prob_thresholds),time])
            res_prob = np.copy(rel_prob)
            overall_frequency = np.mean(real_data,axis=1)

            for t in range(time):
                y_score_bin_mean2[j,t,:],empirical_prob_pos2[j,t,:],sample_num2[j,t,:],bins = reliability_curve(real_data[t,:],prog_data[t,:])
            for i, p in enumerate(prob_thresholds):
                p_dat = prog_data >= p
                r_dat = real_data == 1
                #for t in range(len(time)):
                hit2[j, :, i] = np.sum(r_dat[:, :] & p_dat[:, :],axis=1)
                miss2[j, :, i] = np.sum(r_dat[:, :] & ~p_dat[:, :],axis=1)
                f_alert2[j, :, i] = np.sum(~r_dat[:, :] & p_dat[:, :],axis=1)
                corr_zero2[j, :, i] = np.sum(~r_dat[:, :] & ~p_dat[:, :],axis=1)

                event2[j, :, i] = hit2[j, :, i] + miss2[j, :, i]
                nonevent2[j, :, i] = f_alert2[j, :, i] + corr_zero2[j, :, i]
                total2[j, :, i] = hit2[j, :, i] + miss2[j, :, i] + f_alert2[j, :, i] + corr_zero2[j, :, i]

                roc_hr2[j, :, i] = hit2[j, :, i] / event2[j, :, i]
                roc_far2[j, :, i] = f_alert2[j, :, i] / nonevent2[j, :, i]
                #bin_idx = np.logical_and(p - bin_width / 2 < prog_data,
                #                                         prog_data <= p + bin_width / 2)
                #for t in range(len(time)):
                #    f_alert_bin[j,t,i] = np.sum(real_data[t,bin_idx[t,:]])
                #    corr_zero_bin[j,t,i] = np.sum(real_data[t,bin_idx[t,:]]==0)
                #    bin_total[j,t,i] = np.sum(bin_idx[t,:])
                prog_bin_idx = np.logical_and(p - bin_width / 2 < prog_data,
                                              prog_data <= p + bin_width / 2)

                prog_sum_of_forecasts = np.sum(prog_bin_idx, axis=1)
                for t in range(120):
                    rel_prob[i, t] = prog_sum_of_forecasts[t] * (p - np.nanmean(real_data[t, prog_bin_idx[t, :]])) ** 2
                    res_prob[i, t] = prog_sum_of_forecasts[t] * (
                                np.nanmean(real_data[t, prog_bin_idx[t, :]]) - overall_frequency[t]) ** 2

            reliability2[j, :] = inverse_number_of_forecasts * np.nansum(rel_prob, axis=0)
            resolution2[j, :] = inverse_number_of_forecasts * np.nansum(res_prob, axis=0)
            uncertainty2[j, :] = overall_frequency * (1 - overall_frequency)

            print('Finished ' + file)
            rse2 += (np.nan_to_num(prog_data)-np.nan_to_num(real_data))**2
            briers2[j,:] = inverse_number_of_forecasts*np.sum((prog_data[:,:]-real_data[:,:])**2,axis=1)
        except Exception as e:
            print(e)
    print(str(np.round(j/len(selectedFiles2),2)))
    print(datetime.now()-starttime)
print('Finished')
hit_total2 = np.sum(hit2,0)
miss_total2 = np.sum(miss2,0)
f_alert_total2=np.sum(f_alert2,0)
corr_zero_total2 = np.sum(corr_zero2)


#np.sum(real_data[t,bin_idx[t,:]]==0)/np.sum(bin_idx[t,:])
#np.sum(real_data[t,bin_idx[t,:]])/np.sum(bin_idx[t,:])


event_total2 = np.sum(event2,0)
nonevent_total2 = np.sum(nonevent2,0)

roc_hr_total2 = hit_total2/event_total2
roc_far_total2 = f_alert_total2/nonevent_total2

bsref2 = (np.mean(event_total2[:,0])/(np.mean(event_total2[:,0])+np.mean(nonevent_total2[:,0])))
bs_2 = np.nanmean(briers2[np.nanmax(event2[:,:,0],axis=1)>1000],axis=0)
bs2_2= np.sum(rse2/len(selectedFiles2),axis=1)*inverse_number_of_forecasts

bss2 = 1- bs2_2 / bsref2
y_score_bin_mean_total2 =np.nansum(empirical_prob_pos2[:,-1,:]*sample_num2[:,-1,:],axis=0)/np.nansum(sample_num2[:,-1,:],axis=0)




dir5 = '/scratch/local1/radardata/prognosis12_variance5/'
filelist5 = os.listdir(dir5)


nc = netCDF4.Dataset(dir5 + filelist5[0])

time5 = len(nc.groups['Prognosis Data'].variables['Time'][:])
dist_nested5 = nc.groups['Prognosis Data'].variables['dist_nested'][:][:]
rain_threshold = .5
bin_width = 0.1
prob_thresholds = np.arange(0, 1.01, bin_width)
num_prob = len(prob_thresholds)
max_dist = 19966.141
inverse_number_of_forecasts = 1/np.sum(dist_nested<max_dist)
number_of_forecasts = np.sum(dist_nested<max_dist)
selectedFiles5 = filelist5

hit5 = np.zeros([len(selectedFiles5), time, num_prob])
miss5 = np.copy(hit5)
f_alert5 = np.copy(hit5)
f_alert_bin5 = np.copy(hit5)
corr_zero_bin5 = np.copy(hit5)
bin_total5= np.copy(hit5)
corr_zero5 = np.copy(hit5)
event5 = np.copy(hit5)
nonevent5 = np.copy(hit5)
total5 = np.copy(hit5)
roc_hr5 = np.copy(hit5)
roc_far5 = np.copy(hit5)
briers5 = np.zeros([len(selectedFiles5),time])
prog_data_total5 = np.zeros([time,np.sum(dist_nested<max_dist)])
real_data_total5 = np.copy(prog_data_total5)
rse5 = np.copy(real_data_total5)
reliability5 = np.copy(briers5)
resolution5 = np.copy(briers5)
uncertainty5 = np.copy(briers5)
y_score_bin_mean5 = np.zeros([len(selectedFiles5),time,len(prob_thresholds)])
empirical_prob_pos5 = np.copy(y_score_bin_mean5)
sample_num5 = np.copy(empirical_prob_pos5)
timestamps5 = np.zeros([len(selectedFiles5),time])

for j,file in enumerate(selectedFiles5):
    starttime = datetime.now()
    with netCDF4.Dataset(dir5 + file) as nc:
        try:
            prog_data = nc.groups['Prognosis Data'].variables['probabilities'][:][:][:]
            prog_data = prog_data[:,dist_nested < max_dist]
            prog_data_total5 += np.nan_to_num(prog_data)
            real_data = nc.groups['Prognosis Data'].variables['historic_data'][:][:][:]
            real_data = real_data[:,dist_nested < max_dist]
            real_data_total5 += np.nan_to_num(real_data)
            timestamps5[j,:] = nc.groups['Prognosis Data'].variables['Time'][:]

            rel_prob = np.zeros([len(prob_thresholds),time])
            res_prob = np.copy(rel_prob)
            overall_frequency = np.mean(real_data,axis=1)

            for t in range(time):
                y_score_bin_mean5[j,t,:],empirical_prob_pos5[j,t,:],sample_num5[j,t,:],bins = reliability_curve(real_data[t,:],prog_data[t,:])
            for i, p in enumerate(prob_thresholds):
                p_dat = prog_data >= p
                r_dat = real_data == 1
                #for t in range(len(time)):
                hit5[j, :, i] = np.sum(r_dat[:, :] & p_dat[:, :],axis=1)
                miss5[j, :, i] = np.sum(r_dat[:, :] & ~p_dat[:, :],axis=1)
                f_alert5[j, :, i] = np.sum(~r_dat[:, :] & p_dat[:, :],axis=1)
                corr_zero5[j, :, i] = np.sum(~r_dat[:, :] & ~p_dat[:, :],axis=1)

                event5[j, :, i] = hit5[j, :, i] + miss5[j, :, i]
                nonevent5[j, :, i] = f_alert5[j, :, i] + corr_zero5[j, :, i]
                total5[j, :, i] = hit5[j, :, i] + miss5[j, :, i] + f_alert5[j, :, i] + corr_zero5[j, :, i]

                roc_hr5[j, :, i] = hit5[j, :, i] / event5[j, :, i]
                roc_far5[j, :, i] = f_alert5[j, :, i] / nonevent5[j, :, i]
                #bin_idx = np.logical_and(p - bin_width / 5 < prog_data,
                #                                         prog_data <= p + bin_width / 2)
                #for t in range(len(time)):
                #    f_alert_bin[j,t,i] = np.sum(real_data[t,bin_idx[t,:]])
                #    corr_zero_bin[j,t,i] = np.sum(real_data[t,bin_idx[t,:]]==0)
                #    bin_total[j,t,i] = np.sum(bin_idx[t,:])
                prog_bin_idx = np.logical_and(p - bin_width / 2 < prog_data,
                                              prog_data <= p + bin_width / 2)

                prog_sum_of_forecasts = np.sum(prog_bin_idx, axis=1)
                for t in range(120):
                    rel_prob[i, t] = prog_sum_of_forecasts[t] * (p - np.nanmean(real_data[t, prog_bin_idx[t, :]])) ** 2
                    res_prob[i, t] = prog_sum_of_forecasts[t] * (
                                np.nanmean(real_data[t, prog_bin_idx[t, :]]) - overall_frequency[t]) ** 2

            reliability5[j, :] = inverse_number_of_forecasts * np.nansum(rel_prob, axis=0)
            resolution5[j, :] = inverse_number_of_forecasts * np.nansum(res_prob, axis=0)
            uncertainty5[j, :] = overall_frequency * (1 - overall_frequency)

            print('Finished ' + file)
            rse5 += (np.nan_to_num(prog_data)-np.nan_to_num(real_data))**2
            briers5[j,:] = inverse_number_of_forecasts*np.sum((prog_data[:,:]-real_data[:,:])**2,axis=1)
        except Exception as e:
            print(e)
    print(str(np.round(j/len(selectedFiles5),2)))
    print(datetime.now()-starttime)
print('Finished')
hit_total5 = np.sum(hit5,0)
miss_total5 = np.sum(miss5,0)
f_alert_total5=np.sum(f_alert5,0)
corr_zero_total5 = np.sum(corr_zero5)


#np.sum(real_data[t,bin_idx[t,:]]==0)/np.sum(bin_idx[t,:])
#np.sum(real_data[t,bin_idx[t,:]])/np.sum(bin_idx[t,:])


event_total5 = np.sum(event5,0)
nonevent_total5 = np.sum(nonevent5,0)

roc_hr_total5 = hit_total5/event_total5
roc_far_total5 = f_alert_total5/nonevent_total5

bsref5 = (np.mean(event_total5[:,0])/(np.mean(event_total5[:,0])+np.mean(nonevent_total5[:,0])))
bs_5 = np.nanmean(briers5[np.nanmax(event5[:,:,0],axis=1)>1000],axis=0)
bs2_5= np.sum(rse5/len(selectedFiles5),axis=1)*inverse_number_of_forecasts

bss5 = 1- bs2_5 / bsref5
y_score_bin_mean_total5 =np.nansum(empirical_prob_pos5[:,-1,:]*sample_num5[:,-1,:],axis=0)/np.nansum(sample_num5[:,-1,:],axis=0)



dir6 = '/scratch/local1/radardata/prognosis12_variance6/'
filelist6 = os.listdir(dir6)


nc = netCDF4.Dataset(dir6 + filelist6[0])

time6 = len(nc.groups['Prognosis Data'].variables['Time'][:])
dist_nested6 = nc.groups['Prognosis Data'].variables['dist_nested'][:][:]
rain_threshold = .5
bin_width = 0.1
prob_thresholds = np.arange(0, 1.01, bin_width)
num_prob = len(prob_thresholds)
max_dist = 19966.141
inverse_number_of_forecasts = 1/np.sum(dist_nested<max_dist)
number_of_forecasts = np.sum(dist_nested<max_dist)
selectedFiles6 = filelist6

hit6 = np.zeros([len(selectedFiles6), time, num_prob])
miss6 = np.copy(hit6)
f_alert6 = np.copy(hit6)
f_alert_bin6 = np.copy(hit6)
corr_zero_bin6 = np.copy(hit6)
bin_total6= np.copy(hit6)
corr_zero6 = np.copy(hit6)
event6 = np.copy(hit6)
nonevent6 = np.copy(hit6)
total6 = np.copy(hit6)
roc_hr6 = np.copy(hit6)
roc_far6 = np.copy(hit6)
briers6 = np.zeros([len(selectedFiles6),time])
reliability6 = np.copy(briers6)
resolution6 = np.copy(briers6)
uncertainty6 = np.copy(briers6)
prog_data_total6 = np.zeros([time,np.sum(dist_nested<max_dist)])
real_data_total6 = np.copy(prog_data_total6)
rse6 = np.copy(real_data_total6)
y_score_bin_mean6 = np.zeros([len(selectedFiles6),time,len(prob_thresholds)])
empirical_prob_pos6 = np.copy(y_score_bin_mean6)
sample_num6 = np.copy(empirical_prob_pos6)
timestamps6 = np.zeros([len(selectedFiles6),time])

for j,file in enumerate(selectedFiles6):
    starttime = datetime.now()
    with netCDF4.Dataset(dir6 + file) as nc:
        try:
            prog_data = nc.groups['Prognosis Data'].variables['probabilities'][:][:][:]
            prog_data = prog_data[:,dist_nested < max_dist]
            prog_data_total6 += np.nan_to_num(prog_data)
            real_data = nc.groups['Prognosis Data'].variables['historic_data'][:][:][:]
            real_data = real_data[:,dist_nested < max_dist]
            real_data_total6 += np.nan_to_num(real_data)
            timestamps6[j,:] = nc.groups['Prognosis Data'].variables['Time'][:]
            for t in range(time):
                y_score_bin_mean6[j,t,:],empirical_prob_pos6[j,t,:],sample_num6[j,t,:],bins = reliability_curve(real_data[t,:],prog_data[t,:])

            rel_prob = np.zeros([len(prob_thresholds),time])
            res_prob = np.copy(rel_prob)
            overall_frequency = np.mean(real_data,axis=1)

            for i, p in enumerate(prob_thresholds):
                p_dat = prog_data >= p
                r_dat = real_data == 1
                #for t in range(len(time)):
                hit6[j, :, i] = np.sum(r_dat[:, :] & p_dat[:, :],axis=1)
                miss6[j, :, i] = np.sum(r_dat[:, :] & ~p_dat[:, :],axis=1)
                f_alert6[j, :, i] = np.sum(~r_dat[:, :] & p_dat[:, :],axis=1)
                corr_zero6[j, :, i] = np.sum(~r_dat[:, :] & ~p_dat[:, :],axis=1)

                event6[j, :, i] = hit6[j, :, i] + miss6[j, :, i]
                nonevent6[j, :, i] = f_alert6[j, :, i] + corr_zero6[j, :, i]
                total6[j, :, i] = hit6[j, :, i] + miss6[j, :, i] + f_alert6[j, :, i] + corr_zero6[j, :, i]

                roc_hr6[j, :, i] = hit6[j, :, i] / event6[j, :, i]
                roc_far6[j, :, i] = f_alert6[j, :, i] / nonevent6[j, :, i]
                prog_bin_idx = np.logical_and(p - bin_width / 2 < prog_data,
                                                         prog_data <= p + bin_width / 2)


                prog_sum_of_forecasts = np.sum(prog_bin_idx,axis=1)
                for t in range(120):
                    rel_prob[i,t] = prog_sum_of_forecasts[t] * (p - np.nanmean(real_data[t,prog_bin_idx[t,:]]))**2
                    res_prob[i,t] = prog_sum_of_forecasts[t] * (np.nanmean(real_data[t,prog_bin_idx[t,:]]) - overall_frequency[t])**2


            reliability6[j,:] = inverse_number_of_forecasts*np.nansum(rel_prob,axis=0)
            resolution6[j,:] = inverse_number_of_forecasts*np.nansum(res_prob,axis=0)
            uncertainty6[j,:] = overall_frequency*(1-overall_frequency)
            print('Finished ' + file)
            rse6 += (np.nan_to_num(prog_data)-np.nan_to_num(real_data))**2
            briers6[j,:] = inverse_number_of_forecasts*np.sum((prog_data[:,:]-real_data[:,:])**2,axis=1)
        except Exception as e:
            print(e)
    print(str(np.round(j/len(selectedFiles6),2)))
    print(datetime.now()-starttime)
print('Finished')
hit_total6 = np.sum(hit6,0)
miss_total6 = np.sum(miss6,0)
f_alert_total6=np.sum(f_alert6,0)
corr_zero_total6 = np.sum(corr_zero6)


#np.sum(real_data[t,bin_idx[t,:]]==0)/np.sum(bin_idx[t,:])
#np.sum(real_data[t,bin_idx[t,:]])/np.sum(bin_idx[t,:])


event_total6 = np.sum(event6,0)
nonevent_total6 = np.sum(nonevent6,0)

roc_hr_total6 = hit_total6/event_total6
roc_far_total6 = f_alert_total6/nonevent_total6

bsref6 = (np.mean(event_total6[:,0])/(np.mean(event_total6[:,0])+np.mean(nonevent_total6[:,0])))
bs_6 = np.nanmean(briers6[np.nanmax(event6[:,:,0],axis=1)>1000],axis=0)
bs2_6= np.sum(rse6/len(selectedFiles6),axis=1)*inverse_number_of_forecasts

bss6 = 1- bs2_6 / bsref6
y_score_bin_mean_total6 =np.nansum(empirical_prob_pos6[:,-1,:]*sample_num6[:,-1,:],axis=0)/np.nansum(sample_num6[:,-1,:],axis=0)



plt.rcParams.update({'font.size': 10.5})
plt.rcParams["figure.figsize"] = (5,3)
day = np.zeros([len(selectedFiles)])
month = np.copy(day)
for i,file in enumerate(selectedFiles):
    day[i] =datetime.fromtimestamp(timestamps[i,0]-2*60*60).day
    month[i] = datetime.fromtimestamp(timestamps[i,0]-2*60*60).month
timearray= np.arange(0,121)

# plt.figure()
# for t in range(time):
#     plt.plot(np.hstack([0, roc_far_total[t, ::-1], 1]), np.hstack([0, roc_hr_total[t, ::-1], 1]),
#              color=[1 / time * t, 0, 1 - 1 / time * t])
# plt.xlabel('FAR')
# plt.ylabel('HR')

plt.rcParams["figure.figsize"] = (5,3)
fig,ax = plt.subplots(1)
ROC_AREA=np.zeros(time)
ROC_AREA2=np.zeros(time)
ROC_AREA5=np.zeros(time)
for t in range(time):
    ROC_AREA[t]=(np.trapz(np.hstack([0,roc_hr_total[t,::-1],1]),np.hstack([0,roc_far_total[t,::-1],1]),dx=0.1))
    ROC_AREA2[t] = (
        np.trapz(np.hstack([0, roc_hr_total2[t, ::-1], 1]), np.hstack([0, roc_far_total2[t, ::-1], 1]), dx=0.1))
    ROC_AREA5[t] =(np.trapz(np.hstack([0,roc_hr_total5[t,::-1],1]),np.hstack([0,roc_far_total5[t,::-1],1]),dx=0.1))
ROC_AREA = np.hstack([1,ROC_AREA])
ROC_AREA2 = np.hstack([1,ROC_AREA2])
ROC_AREA5 = np.hstack([1,ROC_AREA5])
roc1, = ax.plot(timearray, ROC_AREA,color = 'b')
roc2, = ax.plot(timearray, ROC_AREA2,color = 'g')
roc5, = ax.plot(timearray, ROC_AREA5,color = 'k')
ax.set_xlim([timearray[0],timearray[-1]])
ax.set_ylim([0.6,1])
ax.set_xticklabels((ax.get_xticks() * 0.5).astype(int))
ax.set_xlabel('Leadtime in minutes')
ax.set_ylabel('${SS}_{ROC}$')
ax.grid()
ax.legend((roc2,roc1,roc5),('STD x1','STD x3','STD x5'))
fig.tight_layout()
fig.savefig('/home/zmaw/u300675/ma_rainprog/roc_area_var_vgl.pgf')

#
# fig,ax = plt.subplots(1)
# ax.plot(timearray,np.hstack([0,bs]))
# ax.set_xlim([0,timearray[-1]])
# ax.set_ylim([0,np.max(bs)])
# ax.set_xticklabels((ax.get_xticks() * 0.5).astype(int))
# ax.set_xlabel('Leadtime in minutes')
# ax.grid()
# ax.set_ylabel('Brier score')
# fig.tight_layout()
# fig.savefig('/home/zmaw/u300675/ma_rainprog/mean_brier_score.pgf')
plt.rcParams["figure.figsize"] = (5,3)
sokolTime = np.arange(20,121,20)
sokolVals = [0.55,0.405,0.31,0.27,0.18,0.21]

fig,ax = plt.subplots(1)
own, = ax.plot(timearray,np.hstack([1,bss]),color='b')
own2, = ax.plot(timearray,np.hstack([1,bss2]),color='g')
own5, = ax.plot(timearray,np.hstack([1,bss5]),color='k')
sokol, = ax.plot(sokolTime,sokolVals,color='r')
ax.set_xlim([0,timearray[-1]])
ax.set_ylim([0,1])
ax.set_xticklabels((ax.get_xticks() * 0.5).astype(int))
ax.set_xlabel('Leadtime in minutes')
ax.grid()
ax.set_ylabel('BSS')
ax.legend((sokol,own2,own,own5),('Sokol et al.,2017','HHG STD x1','HHG STD x3','HHG STD x5'),numpoints=1)
fig.tight_layout()
fig.savefig('/home/zmaw/u300675/ma_rainprog/mean_brier_skill_score_var_vgl.pgf')

plt.rcParams["figure.figsize"] = (4,3.8)
fig,ax = plt.subplots(1)
for t in range(19,time,20):
    ax.plot(bins,np.nansum(empirical_prob_pos[:,t,:]*sample_num[:,t,:],axis=0)/np.nansum(sample_num[:,t,:],axis=0),color=[1 / time * t, 0, 1 - 1 / time * t])
    ax.annotate(str('+'+str(int((t+1)/2))),xy=(bins[-1],np.nansum(empirical_prob_pos[:,t,-1]*sample_num[:,t,-1],axis=0)/np.nansum(sample_num[:,t,-1],axis=0)-0.02),size = 8)
ax.plot([0,1],'k')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_xlabel('Forecast probability')
ax.set_ylabel('Observed relative frequency')
ax.set_aspect('equal')
ax.grid()
fig.tight_layout()
fig.savefig('/home/zmaw/u300675/ma_rainprog/mean_rel_var3.pgf')


plt.rcParams["figure.figsize"] = (4,3.8)
fig,ax = plt.subplots(1)
for t in range(19,time,20):
    ax.plot(bins,np.nansum(empirical_prob_pos2[:,t,:]*sample_num2[:,t,:],axis=0)/np.nansum(sample_num2[:,t,:],axis=0),color=[1 / time * t, 0, 1 - 1 / time * t])
    ax.annotate(str('+'+str(int((t+1)/2))),xy=(bins[-1],np.nansum(empirical_prob_pos2[:,t,-1]*sample_num2[:,t,-1],axis=0)/np.nansum(sample_num2[:,t,-1],axis=0)-0.02),size = 8)
ax.plot([0,1],'k')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_xlabel('Forecast probability')
ax.set_ylabel('Observed relative frequency')
ax.set_aspect('equal')
ax.grid()
fig.tight_layout()
fig.savefig('/home/zmaw/u300675/ma_rainprog/mean_rel_var1.pgf')

plt.rcParams["figure.figsize"] = (5,5)
fig,ax = plt.subplots(1)
for t in range(19,time,20):
    var1, = ax.plot(bins,np.nansum(empirical_prob_pos2[:,t,:]*sample_num2[:,t,:],axis=0)/np.nansum(sample_num2[:,t,:],axis=0),color=[1 / time * t, 0, 1 - 1 / time * t],linestyle='--',linewidth = 1)
    ax.annotate(str('+'+str(int((t+1)/2))),xy=(bins[-1],np.nansum(empirical_prob_pos[:,t,-1]*sample_num[:,t,-1],axis=0)/np.nansum(sample_num[:,t,-1],axis=0)-0.02),size = 8)
    var3, = ax.plot(bins,np.nansum(empirical_prob_pos[:,t,:]*sample_num[:,t,:],axis=0)/np.nansum(sample_num[:,t,:],axis=0),color=[1 / time * t, 0, 1 - 1 / time * t])
    var5, = ax.plot(bins,np.nansum(empirical_prob_pos5[:,t,:]*sample_num5[:,t,:],axis=0)/np.nansum(sample_num5[:,t,:],axis=0),color=[1 / time * t, 0, 1 - 1 / time * t],linestyle='-.',linewidth = 1)
ax.plot([0,1],'k')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_xlabel('Forecast probability')
ax.set_ylabel('Observed relative frequency')
ax.set_aspect('equal')
ax.grid()
fig.tight_layout()
ax.legend((var1,var3,var5),('STD x1','STD x3','STD x5'),numpoints=1)

fig.savefig('/home/zmaw/u300675/ma_rainprog/mean_rel_var_vgl.pgf')
