import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from datetime import datetime
from init import reliability_curve

params = {"pgf.texsystem": "pdflatex"}
plt.rcParams.update(params)
days = np.array([[5,22],[5,23],[6,2],[6,13],[6,25],[7,21],[8,28],[6,14],[6,18],[6,23],[6,24]])
col = [np.random.rand(3, len(days))]

filelist = os.listdir('/scratch/local1/radardata/prognosis12_variance1/')
nc = netCDF4.Dataset('/scratch/local1/radardata/prognosis12_variance1/' + filelist[0])

paths = ['/scratch/local1/radardata/prognosis12_variance1/','/scratch/local1/radardata/prognosis12_variance3/',
         '/scratch/local1/radardata/prognosis12_variance5/','/scratch/local1/radardata/prognosis12_variance6/',
         '/scratch/local1/radardata/prognosis12_variance7/','/scratch/local1/radardata/prognosis12_variance8/',
         '/scratch/local1/radardata/prognosis12_variance9/','/scratch/local1/radardata/prognosis12_variance10/']

paths = [paths[4]]

time = len(nc.groups['Prognosis Data'].variables['Time'][:])
dist_nested = nc.groups['Prognosis Data'].variables['dist_nested'][:][:]
rain_threshold = .5
bin_width = 0.1
prob_thresholds = np.arange(0, 1.01, bin_width)
num_prob = len(prob_thresholds)
max_dist = 19966.141
inverse_number_of_forecasts = 1/np.sum(dist_nested<max_dist)
number_of_forecasts = np.sum(dist_nested<max_dist)
Hit = []
Miss = []
F_alert = []
F_alert_bin = []
Corr_zero_bin = []
Bin_total = []
Corr_zero = []
Event = []
Nonevent = []
Total = []
Roc_hr = []
Roc_far = []
Briers = []
Prog_data_total = []
Real_data_total = []
Rse = []
Reliability = []
Resolution = []
Uncertainty = []
Y_score_bin_mean = []
Empirical_prob_pos = []
Sample_num = []
Timestamps = []
Std = []
Hit_total = []
Miss_total = []
F_alert_total=[]
Corr_zero_total = []


#np.sum(real_data[t,bin_idx[t,:]]==0)/np.sum(bin_idx[t,:])
#np.sum(real_data[t,bin_idx[t,:]])/np.sum(bin_idx[t,:])


Event_total = []
Nonevent_total = []

Roc_hr_total = []
Roc_far_total = []

Bsref = []
Bs = []
Bs2= []

Bss = []
Y_score_bin_mean_total =[]

for path in paths:
    filelist = os.listdir(path)
    selectedFiles = filelist
    Std.append(int(path.rsplit('/',4)[3][20:]))
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
        with netCDF4.Dataset(path + file) as nc:
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
    print('Finished ' + path)
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
    Roc_hr.append(roc_hr)
    Roc_far.append(roc_far)
    Resolution.append(resolution)
    Uncertainty.append(uncertainty)
    Reliability.append(reliability)
    Rse.append(rse)
    Briers.append(briers)

    Hit_total.append(hit_total)
    Miss_total.append(miss_total)
    F_alert_total.append(f_alert_total)
    Corr_zero_total.append(corr_zero_total)

    Nonevent_total.append(nonevent_total)
    Event_total.append(event_total)
    Roc_far_total.append(roc_far_total)
    Roc_hr_total.append(roc_hr_total)
    Bsref.append(bsref)
    Bs.append(bs)
    Bs2.append(bs2)
    Bss.append(bss)
    Y_score_bin_mean_total.append(y_score_bin_mean_total)
    Empirical_prob_pos.append(empirical_prob_pos)
    Sample_num.append(sample_num)

plt.rcParams.update({'font.size': 18})
plt.rcParams["figure.figsize"] = (5,5)
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

plt.rcParams["figure.figsize"] = (5,3.5)

fig,ax = plt.subplots(1)
unc, = ax.plot(timearray, np.hstack([np.nanmean(Uncertainty[0],axis=0)[0], np.nanmean(Uncertainty[0],axis=0)]))
rel, = ax.plot(timearray, np.hstack([0,np.nanmean(Reliability[0],axis=0)]))
res, = ax.plot(timearray,np.hstack([np.nanmean(Uncertainty[0],axis=0)[0], np.nanmean(Resolution[0],axis=0)]))
brier, = ax.plot(timearray,np.hstack([0,np.nanmean(Briers[0],axis=0)]))
ax.set_xlim([0,timearray[-1]])
ax.set_ylim([0,0.14])
ax.set_xlabel('Lead time in minutes')
ax.set_ylabel('Units')
ax.set_xticklabels((ax.get_xticks() * 0.5).astype(int))
ax.legend((unc,rel,res,brier),('Uncertainty','Reliability','Resolution','Brier Score'),numpoints=1)
ax.grid()
fig.tight_layout()
fig.savefig('/home/zmaw/u300675/ma_rainprog/bss_decomposition_std1.pgf')

fig,ax = plt.subplots(1)
unc, = ax.plot(timearray, np.hstack([np.nanmean(Uncertainty[0],axis=0)[0], np.nanmean(Uncertainty[0],axis=0)]),color=[31/255, 119/255, 180/255])
rel, = ax.plot(timearray, np.hstack([0,np.nanmean(Reliability[0],axis=0)]),color=[100/100,50/100,5/100])
res, = ax.plot(timearray,np.hstack([np.nanmean(Uncertainty[0],axis=0)[0], np.nanmean(Resolution[0],axis=0)]),color=[17/100,63/100,17/100])
brier, = ax.plot(timearray,np.hstack([0,np.nanmean(Briers[0],axis=0)]),color=[84/100,15/100,16/100])

rel, = ax.plot(timearray, np.hstack([0,np.nanmean(Reliability[4],axis=0)]),Linestyle='--',color=[100/100,50/100,5/100])
res, = ax.plot(timearray,np.hstack([np.nanmean(Uncertainty[4],axis=0)[0], np.nanmean(Resolution[4],axis=0)]),Linestyle='--',color=[17/100,63/100,17/100])
brier, = ax.plot(timearray,np.hstack([0,np.nanmean(Briers[4],axis=0)]),Linestyle='--',color=[84/100,15/100,16/100])

ax.set_xlim([0,timearray[-1]])
ax.set_ylim([0,0.14])
ax.set_xlabel('Lead time in minutes')
ax.set_ylabel('Units')
ax.set_xticklabels((ax.get_xticks() * 0.5).astype(int))
ax.legend((unc,rel,res,brier),('Uncertainty','Reliability','Resolution','Brier Score'),numpoints=1)
ax.grid()
fig.tight_layout()
fig.savefig('/home/zmaw/u300675/ma_rainprog/bss_decomposition_std1_std7.pgf')



plt.rcParams["figure.figsize"] = (5,3.5)

Roc_area = []

for std in range(len(Roc_hr_total)):
    Roc_area.append(np.hstack([1,np.trapz(np.column_stack([np.zeros(120),Roc_hr_total[std][:,::-1],np.ones(120)]),
                                          np.column_stack([np.zeros(120),Roc_far_total[std][:,::-1],np.ones(120)]),axis=1,dx=0.1)]))

cols = ['b','g',[203/255,133/255,0], 'c','k','m','y','r']
roc={}
fig,ax = plt.subplots(1)
for std in range(len(Roc_area)):
    roc[std], = ax.plot(timearray, Roc_area[std], color=cols[std],label='STD x'+str(Std[std]))

ax.set_xlim([timearray[0],timearray[-1]])
ax.set_ylim([0.6,1])
ax.set_xticklabels((ax.get_xticks() * 0.5).astype(int))
ax.set_xlabel('Lead time in minutes')
ax.set_ylabel('${SS}_{ROC}$')
ax.grid()
fig.tight_layout()
#fig.savefig('/home/zmaw/u300675/ma_rainprog/roc_area_std1.pgf')

ax.legend(ncol=2)
fig.savefig('/home/zmaw/u300675/ma_rainprog/roc_area_allstd.pgf')

plt.rcParams["figure.figsize"] = (5,4)
fig,ax = plt.subplots(1)
rocdiff={}
for std in range(1,len(Roc_area)):
    rocdiff[std], = ax.plot(timearray, Roc_area[std]-Roc_area[0], color=cols[std], label='STD x' + str(Std[std]) + ' - STD x1',linewidth=1.5)


ax.set_xlim([0,timearray[-1]])
ax.set_ylim([0,0.09])
ax.set_yticks([0,0.02,0.04,0.06,0.08])
ax.set_xticklabels((ax.get_xticks() * 0.5).astype(int))
ax.set_xlabel('Lead time in minutes')
ax.grid()
ax.set_ylabel('$\Delta$ ${SS}_{ROC}$')
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig('/home/zmaw/u300675/ma_rainprog/roc_area_diff_allstd.pgf')

#
# fig,ax = plt.subplots(1)
# ax.plot(timearray,np.hstack([0,bs]))
# ax.set_xlim([0,timearray[-1]])
# ax.set_ylim([0,np.max(bs)])
# ax.set_xticklabels((ax.get_xticks() * 0.5).astype(int))
# ax.set_xlabel('Lead time in minutes')
# ax.grid()
# ax.set_ylabel('Brier score')
# fig.tight_layout()
# fig.savefig('/home/zmaw/u300675/ma_rainprog/mean_brier_score.pgf')
plt.rcParams["figure.figsize"] = (5,3.5)
sokolTime = np.arange(20,121,20)
sokolVals = [0.55,0.405,0.31,0.27,0.18,0.21]
sokolRel = np.array([0.5, 0.8,1.2,1.7,1.95,2.4])*10**(-5)
sokolRes = [0.012,0.0085,0.007,0.006,0.0045,0.004]
bss_corr = np.zeros([11,120])

bss_ = {}
fig,ax = plt.subplots(1)
for std in range(0,len(Bss)):
    bss_corr[std] = 1-(Bs2[std]/0.141)#(Bsref[std]*(1-Bsref[std])))
    bss_[std], = ax.plot(timearray,np.hstack([1,bss_corr[std]]),color=cols[std],label='STD x' + str(Std[std]))

sokol, = ax.plot(sokolTime,sokolVals,color='lime',label='Sokol et al.,2017')
ax.set_xlim([0,timearray[-1]])
ax.set_ylim([0,1])
ax.set_xticklabels((ax.get_xticks() * 0.5).astype(int))
ax.set_xlabel('Lead time in minutes')
ax.grid()
ax.set_ylabel('BSS')
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig('/home/zmaw/u300675/ma_rainprog/bss_allstd.pgf')

plt.rcParams["figure.figsize"] = (5,4)
bss_diff = {}
fig,ax = plt.subplots(1)
for std in range(1,len(Bss)):
    bss_diff[std], = ax.plot(timearray,np.hstack([1,bss_corr[std]])-np.hstack([1,bss_corr[0]]),color=cols[std],label='STD x'+ str(Std[std])+' - STD x1')
ax.set_xlim([0,timearray[-1]])
ax.set_ylim([-0.01,0.20])
ax.set_xticklabels((ax.get_xticks() * 0.5).astype(int))
ax.set_yticks((0,0.05,0.1,0.15,0.2))
ax.set_xlabel('Lead time in minutes')
ax.grid()
ax.set_ylabel('$\Delta$ BSS')
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig('/home/zmaw/u300675/ma_rainprog/bss_diff_allstd.pgf')


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
#fig.savefig('/home/zmaw/u300675/ma_rainprog/mean_rel_var3.pgf')


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
#fig.savefig('/home/zmaw/u300675/ma_rainprog/mean_rel_var1.pgf')

plt.rcParams["figure.figsize"] = (4.5,4.5)
fig,ax = plt.subplots(1)
for t in range(19,time,20):
    var1, = ax.plot(bins,Y_score_bin_mean_total[0,t,],color=[1 / time * t, 0, 1 - 1 / time * t],linestyle='--',linewidth = 1)
    ax.annotate(str('+'+str(int((t+1)/2))),xy=(bins[-1],np.nansum(empirical_prob_pos6[:,t,-1]*sample_num6[:,t,-1],axis=0)/np.nansum(sample_num6[:,t,-1],axis=0)-0.01),size = 8)
    #var3, = ax.plot(bins,np.nansum(empirical_prob_pos[:,t,:]*sample_num[:,t,:],axis=0)/np.nansum(sample_num[:,t,:],axis=0),color=[1 / time * t, 0, 1 - 1 / time * t])
    #var5, = ax.plot(bins,np.nansum(empirical_prob_pos5[:,t,:]*sample_num5[:,t,:],axis=0)/np.nansum(sample_num5[:,t,:],axis=0),color=[1 / time * t, 0, 1 - 1 / time * t],linestyle='-.',linewidth = 1)
    var6, = ax.plot(bins,np.nansum(empirical_prob_pos6[:,t,:]*sample_num6[:,t,:],axis=0)/np.nansum(sample_num6[:,t,:],axis=0),color=[1 / time * t, 0, 1 - 1 / time * t],linewidth = 1)
    #var7, = ax.plot(bins,np.nansum(empirical_prob_pos7[:,t,:]*sample_num7[:,t,:],axis=0)/np.nansum(sample_num7[:,t,:],axis=0),color=[1 / time * t, 0, 1 - 1 / time * t],linestyle='--',linewidth = 1)

ax.plot([0,1],'k',alpha=0.6)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_xlabel('Forecast probability')
ax.set_ylabel('Observed relative frequency')
ax.set_aspect('equal')
ax.grid()
fig.tight_layout()
ax.legend((var1,var6),('STD x1','STD x6'),numpoints=1)
#fig.savefig('/home/zmaw/u300675/ma_rainprog/mean_rel_var_vgl.pgf')


plt.rcParams["figure.figsize"] = (4,3.8)
fig,ax = plt.subplots(1)
for t in range(19,time,20):
    std1, = ax.plot(bins,np.nansum(Empirical_prob_pos[0][:,t,:]*
                                   Sample_num[0][:,t,:],axis=0)/np.nansum(Sample_num[0][:,t,:],axis=0),color=[1 / time * t, 0, 1 - 1 / time * t],Linewidth=1.2)
    ax.annotate(str('+' + str(int((t + 1) / 2))),
                xy=(float(bins[-1]),float(np.nansum(Empirical_prob_pos[4][:, t, -1] * Sample_num[4][:, t, -1],
                                        axis=0) / np.nansum(Sample_num[4][:, t, -1], axis=0)) - 0.01),size=8)
    std7, = ax.plot(bins,np.nansum(Empirical_prob_pos[4][:,t,:]*
                                   Sample_num[4][:,t,:],axis=0)/np.nansum(Sample_num[4][:,t,:],axis=0),color=[1 / time * t, 0, 1 - 1 / time * t],Linestyle='--',Linewidth=1.2)

ax.plot([0,1],'k')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_xlabel('Forecast probability')
ax.set_ylabel('Observed relative frequency')
ax.set_aspect('equal')
ax.grid()
ax.legend((std1,std7),('STD x1','STD x7'),numpoints=1)
fig.tight_layout()
fig.savefig('/home/zmaw/u300675/ma_rainprog/rel_std1_std7.pgf')


plt.rcParams["figure.figsize"] = (5,4)

bss_ = {}
days_ = np.array([[5,22],[5,23],[6,2],[6,13],[6,14],[6,18],[6,23],[6,24],[6,25],[7,21],[8,28]])


bss_day = np.zeros([11,120])
bss_count = np.zeros(11)
for run in range(len(Briers[4])):

    bss_day[np.where((day[run]==days_[:,1])&(month[run]==days_[:,0]))[0][0]]+=1-(Briers[4][run]/0.14)
    bss_count[np.where((day[run]==days_[:,1])&(month[run]==days_[:,0]))[0][0]]+=1

bss_day_mean = np.copy(bss_day)
for day_ in range(len(days_)):
    bss_day_mean[day_,:] = bss_day[day_,:]/bss_count[day_]

linestyles= ['-','-','-','-','-','-','-','-','-','-','--']
fig,ax = plt.subplots(1)
bss_day_mean_={}
for day_ in range(0,len(days_)):
    bss_day_mean_[day_], = ax.plot(timearray,np.hstack([1,bss_day_mean[day_]]),label=str(days_[day_,1]).zfill(2)+'.'+str(days_[day_,0]).zfill(2),linestyle=linestyles[day_])

ax.set_xlim([0,timearray[-1]])
ax.set_ylim([-1,1])
ax.set_xticklabels((ax.get_xticks() * 0.5).astype(int))
ax.set_xlabel('Lead time in minutes')
ax.grid()
ax.set_ylabel('BSS')
ax.legend(ncol=2)
fig.tight_layout()

fig.savefig('/home/zmaw/u300675/ma_rainprog/bss_day_std7.pgf')

bss_ = {}
days_ = np.array([[5,22],[5,23],[6,2],[6,13],[6,14],[6,18],[6,23],[6,24],[6,25],[7,21],[8,28]])


bss_day = np.zeros([11,120])
bss_count = np.zeros(11)
for run in range(len(Briers[0])):
    try:
        bss_day[np.where((day[run]==days_[:,1])&(month[run]==days_[:,0]))[0][0]]+=1-(Briers[0][run]/0.14)
        bss_count[np.where((day[run]==days_[:,1])&(month[run]==days_[:,0]))[0][0]]+=1
    except:
        print(1)

bss_day_mean = np.copy(bss_day)
for day_ in range(len(days_)):
    bss_day_mean[day_,:] = bss_day[day_,:]/bss_count[day_]

fig,ax = plt.subplots(1)
bss_day_mean_={}
for day_ in range(0,len(days_)):
    bss_day_mean_[day_], = ax.plot(timearray,np.hstack([1,bss_day_mean[day_]]),label=str(days_[day_,1]).zfill(2)+'.'+str(days_[day_,0]).zfill(2),linestyle=linestyles[day_])

ax.set_xlim([0,timearray[-1]])
ax.set_ylim([-1,1])
ax.set_xticklabels((ax.get_xticks() * 0.5).astype(int))
ax.set_xlabel('Lead time in minutes')
ax.grid()
ax.set_ylabel('BSS')
ax.legend(ncol=2)
fig.tight_layout()

fig.savefig('/home/zmaw/u300675/ma_rainprog/bss_day_std1.pgf')


colos = plt.rcParams['axes.prop_cycle'].by_key()['color']
colos.append(plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
roc_far_day = [[] for i in range(11)]
roc_hr_day = [[] for i in range(11)]
#roc_hr_count = []
#roc_far_count = np.zeros([11,120,11])

run_num = 4
for run in range(len(Briers[run_num ])):
    roc_far_day[np.where((day[run]==days_[:,1])&(month[run]==days_[:,0]))[0][0]].append(Roc_far[run_num ][run])
    roc_hr_day[np.where((day[run] == days_[:, 1]) & (month[run] == days_[:, 0]))[0][0]].append(Roc_hr[run_num ][run])

roc_far_day_mean =np.zeros([11,120,11])
roc_hr_day_mean = np.copy(roc_far_day_mean)

for day_ in range(len(days_)):
    roc_far_day_mean[day_,:,:] = np.nanmean(np.asarray(roc_far_day[day_]),axis=0)
    roc_hr_day_mean[day_,:,:]= np.nanmean(np.asarray(roc_hr_day[day_]),axis=0)


roc_area_day = []

for day_ in range(len(days_)):
    roc_area_day.append(np.hstack([1,np.trapz(np.column_stack([np.zeros(120),roc_hr_day_mean[day_,:,::-1],np.ones(120)]),
                                          np.column_stack([np.zeros(120),roc_far_day_mean[day_,:,::-1],np.ones(120)]),axis=1,dx=0.1)]))

plt.rcParams["figure.figsize"] = (5,4)

fig,ax = plt.subplots(1)
roc_ = {}
linestyles= ['-','-','-','-','-','-','-','-','-','-','--']

for day_ in range(len(days_)):
    roc_[day_], =ax.plot(timearray, roc_area_day[day_],label=str(days_[day_,1]).zfill(2)+'.'+str(days_[day_,0]).zfill(2),linestyle=linestyles[day_])


ax.set_xlim([0,timearray[-1]])
ax.set_ylim([0.25,1])
ax.set_xticklabels((ax.get_xticks() * 0.5).astype(int))
ax.set_xlabel('Lead time in minutes')
ax.grid()
ax.set_ylabel('${SS}_{ROC}$')
ax.legend(ncol=3)
fig.tight_layout()

fig.savefig('/home/zmaw/u300675/ma_rainprog/roc_days_std7.pgf')




daystrings = []
for t in range(len(days)):
    daystrings.append(str(days_[t,1]).zfill(2)+'.'+str(days_[t,0]).zfill(2))


plt.rcParams["figure.figsize"] = (5,3.5)
fig,ax = plt.subplots(1)
ax.bar(np.arange(0,10), bss_count[:10],color=plt.rcParams['axes.prop_cycle'].by_key()['color'])
ax.bar(10, bss_count[10],color=plt.rcParams['axes.prop_cycle'].by_key()['color'],edgecolor='white', hatch="/")
ticks = np.arange(0,11)
plt.xticks(ticks,daystrings,rotation=45)
ax.set_xlabel('Date')
ax.set_ylabel('Number of forecasts')
plt.tight_layout()
ax.grid(linewidth=0.5)
fig.savefig('/home/zmaw/u300675/ma_rainprog/forecasts_day.pgf')


fig,ax = plt.subplots()
ax.set_yscale('log')
width =0.25
xvals = np.arange(0,11,1)
dist10 = ax.bar(xvals-width, np.nansum(Sample_num[0][:,20,:],axis=0),width,label='10 minutes lead time')
dist30 = ax.bar(xvals, np.nansum(Sample_num[0][:,60,:],axis=0),width,label='30 minutes lead time')
dist60 = ax.bar(xvals+width, np.nansum(Sample_num[0][:,-1,:],axis=0),width,label='60 minutes lead time')
ax.set_xlabel('Probability')
ax.set_xticklabels(ax.get_xticks()/10)
ax.set_ylabel('Number of forecasted pixels')
ax.legend()
plt.tight_layout()