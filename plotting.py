import matplotlib.patches as mpatches

import matplotlib.colors as colors
contours = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'cmaptest',cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('BuPu')
newcmap=truncate_colormap(cmap,0.2,1)
newcmap.set_under('1')
#plt.rcParams["figure.figsize"] = (9,6)
params = {"pgf.texsystem": "pdflatex"}
plt.rcParams.update(params)


plt.rcParams.update({'font.size': 10.5})
plt.rcParams["figure.figsize"] = (3,3)
fig, ax = plt.subplots(1)
dataArea1 =self.nested_data[t+1,
                        (int(field.maxima[0, 1]) - self.cRange * 2):(int(field.maxima[0, 1]) + self.cRange * 2),
                        (int(field.maxima[0, 2]) - self.cRange * 2):(int(field.maxima[0, 2]) + self.cRange * 2)]
dat = ax.imshow(dataArea1, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1),cmap=newcmap)
plt.plot(48,48,marker='X',color='red',markersize=8)
plt.plot(48+int(cIdx[1] - 0.5 * len(c)),48+int(cIdx[0] - 0.5 * len(c)),marker='X',color='black',markersize=8)
s = plt.colorbar(dat, format=matplotlib.ticker.ScalarFormatter(),fraction=0.046, pad=0.04)
s.set_label('Precipitation in mm/h')
s.set_clim(0.1, 100)
s.set_ticks(contours)
s.draw_all()
ax.grid(linewidth=0.5)
ax.set_xlabel('Extent in km')
ax.set_ylabel('Extent in km')
ax.set_xticklabels(ax.get_xticks()*0.1)
ax.set_yticklabels(ax.get_yticks()*0.1)
ax.invert_yaxis()
plt.tight_layout(pad=0)
fig.savefig('/home/zmaw/u300675/ma_rainprog/displacement_t1.pgf')

fig, ax = plt.subplots(1)
dataArea1 =self.nested_data[t,
                       (int(field.maxima[0, 1]) - self.cRange * 2):(int(field.maxima[0, 1]) + self.cRange * 2),
                       (int(field.maxima[0, 2]) - self.cRange * 2):(int(field.maxima[0, 2]) + self.cRange * 2)]
dat = ax.imshow(dataArea1, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1),cmap=newcmap)
corrAreaRect = mpatches.Rectangle((24,24),48,48,color='r',linewidth=1,fill=0)
ax.add_patch(corrAreaRect)
plt.plot(48,48,marker='X',color='red',markersize=8)
s = plt.colorbar(dat, format=matplotlib.ticker.ScalarFormatter(),fraction=0.046, pad=0.04)
s.set_label('Precipitation in mm/h')
s.set_clim(0.1, 100)
s.set_ticks(contours)
s.draw_all()
ax.set_xticklabels(ax.get_xticks()*0.1)
ax.set_yticklabels(ax.get_yticks()*0.1)
ax.set_xlabel('Extent in km')
ax.set_ylabel('Extent in km')
ax.grid(linewidth=0.5)
ax.invert_yaxis()
plt.tight_layout(pad=0)
fig.savefig('/home/zmaw/u300675/ma_rainprog/displacement_t0.pgf')




plt.rcParams.update({'font.size': 10.5})
plt.rcParams["figure.figsize"] = (4,4)
fig, ax = plt.subplots(1)
dataArea1 =self.nested_data[t+1,
                        (int(field.maxima[0, 1]) - self.cRange * 2):(int(field.maxima[0, 1]) + self.cRange * 2),
                        (int(field.maxima[0, 2]) - self.cRange * 2):(int(field.maxima[0, 2]) + self.cRange * 2)]
dat = ax.imshow(dataArea1, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1),cmap=newcmap)
plt.plot(48,48,marker='X',color='red',markersize=8)
plt.plot(48+int(cIdx[1] - 0.5 * len(c)),48+int(cIdx[0] - 0.5 * len(c)),marker='X',color='black',markersize=8)
s = plt.colorbar(dat, format=matplotlib.ticker.ScalarFormatter())
s.set_label('Precipitation in mm/h')
s.set_clim(0.1, 100)
s.set_ticks(contours)
s.draw_all()
ax.grid(linewidth=0.5)
ax.set_xlabel('Extent in km')
ax.set_ylabel('Extent in km')
ax.set_xticklabels(ax.get_xticks()*0.1)
ax.set_yticklabels(ax.get_yticks()*0.1)
ax.invert_yaxis()
plt.show()

fig, ax = plt.subplots(1)
dataArea1 =self.nested_data[t,
                       (int(field.maxima[0, 1]) - self.cRange * 2):(int(field.maxima[0, 1]) + self.cRange * 2),
                       (int(field.maxima[0, 2]) - self.cRange * 2):(int(field.maxima[0, 2]) + self.cRange * 2)]
dat = ax.imshow(dataArea1, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1),cmap=newcmap)
corrAreaRect = mpatches.Rectangle((24,24),48,48,color='r',linewidth=1,fill=0)
ax.add_patch(corrAreaRect)
plt.plot(48,48,marker='X',color='red',markersize=8)
s = plt.colorbar(dat, format=matplotlib.ticker.ScalarFormatter())
s.set_label('Precipitation in mm/h')
s.set_clim(0.1, 100)
s.set_ticks(contours)
s.draw_all()
ax.set_xticklabels(ax.get_xticks()*0.1)
ax.set_yticklabels(ax.get_yticks()*0.1)
ax.grid(linewidth=0.5)
ax.invert_yaxis()
plt.show()



plt.rcParams["figure.figsize"] = (5,5)
fig,ax = plt.subplots(1)
corr = ax.imshow(c, cmap=plt.get_cmap('inferno_r'))
s = plt.colorbar(corr,fraction=0.046, pad=0.04)
s.set_clim(0,np.max(c))
s.set_label('Inverse Similarity in $\mathrm{mm^{2}/h^{2}}$')
s.draw_all()
plt.plot(25,25,marker='X',color='ghostwhite',markersize=8)
plt.plot(cIdx[1],cIdx[0],marker='X',color='black',markersize=8)
ax.set_xticklabels(ax.get_xticks()*0.1)
ax.set_yticklabels(ax.get_yticks()*0.1)
ax.set_xlabel('Extent in km')
ax.set_ylabel('Extent in km')
ax.grid(linewidth=0.5)
ax.invert_yaxis()
plt.tight_layout()
plt.show()
fig.savefig('/home/zmaw/u300675/ma_rainprog/correlation_matrix.pgf')


import matplotlib.patches as mpatches
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM,GoogleTiles
import matplotlib.pyplot as plt
import matplotlib.dates
import matplotlib.patches as mpatches
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

fig = plt.figure()
osm_tiles = OSM()
plt.rcParams["figure.figsize"] = (9,6)
plt.rcParams.update({'font.size': 10.5})
ax = fig.add_subplot(111,projection=ccrs.PlateCarree())
ax.coastlines()
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_left = False
rect = mpatches.Rectangle((6,0),12,84,linewidth=2,fill=False,color='red')
ax.add_patch(rect)
google = GoogleTiles()
mercator = google.crs

fig2 = plt.figure()
ax2 = fig2.add_subplot(111,projection=mercator)

ax2.set_extent([7.5, 12.6, 52.4, 55.5])
ax2.add_image(osm_tiles, 8, interpolation='bilinear')
xy_dwd = [10.04683,54.0044]
xy_dwd_T = mercator.transform_point(xy_dwd[0],xy_dwd[1],ccrs.PlateCarree())
radarCircle_dwd = mpatches.Circle(xy=xy_dwd_T, radius=260000, color='r', linewidth=1, fill=0,transform = mercator)
ax2.add_patch(radarCircle_dwd)
ax2.scatter(xy_dwd_T[0],xy_dwd_T[1],color='r',marker='x')

xy_pat = [9.9734,53.56833]
xy_pat_T = mercator.transform_point(xy_pat[0],xy_pat[1],ccrs.PlateCarree())
radarCircle_pat = mpatches.Circle(xy=xy_pat_T, radius=20000*1.73, color='k', linewidth=1, fill=0,transform = mercator)
ax2.add_patch(radarCircle_pat)
x_rect = np.array([9.0671469606965474,10.90388580413939])
y_rect = np.array([53.018473372584012,54.111564718254151])
xy_rect=mercator.transform_points(ccrs.PlateCarree(),x_rect,y_rect)
radarRect = mpatches.Rectangle(xy_rect[0],width=xy_rect[1,0]-xy_rect[0,0],height=xy_rect[1,1]-xy_rect[0,1], color='b',fill=0)
ax2.add_patch(radarRect)
ax2.scatter(xy_pat_T[0],xy_pat_T[1],color='k',marker='x')
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax2.xaxis.set_major_formatter(lon_formatter)
ax2.yaxis.set_major_formatter(lat_formatter)
ax2.set_xticks([8,9,10,11,12], crs=ccrs.PlateCarree())
ax2.set_yticks([53,54,55], crs=ccrs.PlateCarree())
fig2.savefig('/home/zmaw/u300675/ma_rainprog/radaroverview.pgf')

#gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                  linewidth=0.5, color='gray', alpha=0.2, linestyle='--')
#gl2.xlabels_top = False
#gl2.ylabels_right = False
plt.show()



plt.rcParams.update({'font.size': 22})
t = 0
fig, ax = plt.subplots(1, figsize=(10, 8))
im = ax.imshow(dwd.nested_data[0, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=newcmap)
plt.show(block=False)
o, = plt.plot(*np.transpose(dwd.progField.return_histMaxima(t)[:, 2:0:-1]), 'rX')
s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter())
s.set_clim(0.1, 100)
s.set_ticks(contours)
ax.set_xlim([50,448])
ax.set_ylim([448,50])
ax.grid(linewidth=0.5)
s.draw_all()
ax.set_xticklabels((ax.get_xticks()-50) * 0.25)
ax.set_yticklabels((ax.get_yticks()-50) * 0.25)
ax.invert_yaxis()

t = 0
fig, ax = plt.subplots(1, figsize=(10, 8))
im = ax.imshow(dwd.nested_data[0, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=newcmap)
plt.show(block=False)
o, = plt.plot(*np.transpose(dwd.progField.return_histMaxima(t)[:, 2:0:-1]), 'rX')
maxima = dwd.progField.return_histMaxima(t)[:, 2:0:-1]
for u in range(len(maxima)):
    proxCircle = mpatches.Circle((maxima[u,0],maxima[u,1]),3000/dwd.resolution,color='k',linewidth=1,fill=0)
    ax.add_patch(proxCircle)
s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter())
s.set_clim(0.1, 100)
s.set_ticks(contours)
ax.set_xlim([100,300])
ax.set_ylim([300,100])
ax.grid(linewidth=0.5)
s.draw_all()
ax.set_xticklabels((ax.get_xticks()-50) * 0.25)
ax.set_yticklabels((ax.get_yticks()-50) * 0.25)
ax.invert_yaxis()
plt.rcParams.update({'font.size': 22})




col = np.concatenate(
    [np.zeros([1, np.max(dwd.progField.activeIds)])+0.5, np.random.rand(2, np.max(dwd.progField.activeIds))])
plt.figure(figsize=(8, 8))
ax = plt.axes()
for i, field in enumerate(dwd.progField.activeFields):
    for t in field.histMaxima:
        pnt, = plt.plot(*np.transpose(t[0][2:0:-1]), color=col[:, field.id - 1], marker='o',linestyle='None')
pnt, = plt.plot(0,0,color='black',marker='o',linestyle='None')
for i, field in enumerate(dwd.progField.inactiveFields):
    for t in field.histMaxima:
        cross, = plt.plot(*np.transpose(t[0][2:0:-1]), color=(1, 0, 0), marker='x',linestyle='None')
plt.gca().invert_yaxis()
plt.show(block=False)
ax.set_xlim([100,500])
ax.set_ylim([500,100])
ax.grid(linewidth=0.5)
ax.set_xticklabels(ax.get_xticks() * 0.25)
ax.set_yticklabels(ax.get_yticks() * 0.25)
ax.legend((pnt,cross),('Accepted Track','Rejected Track'),numpoints=1)
ax.invert_yaxis()


col = np.concatenate(
    [np.random.rand(1,np.max(lawr.progField.activeIds))*0.7, np.random.rand(2, np.max(lawr.progField.activeIds))])
plt.rcParams.update({'font.size': 10.5})
plt.rcParams["figure.figsize"] = (5.5,5.5)
fig,ax = plt.subplots(1)
for i, field in enumerate(lawr.progField.activeFields):
    for t in field.histMaxima:
        pnt, = plt.plot(*np.transpose(t[0][2:0:-1]), color=col[:, field.id - 1], marker='o',linestyle='None',markersize=4)
pnt, = plt.plot(-100,-100,color='black',marker='o',linestyle='None',markersize=4)
for i, field in enumerate(lawr.progField.inactiveFields):
    for t in field.histMaxima:
        cross, = plt.plot(*np.transpose(t[0][2:0:-1]), color=(1, 0, 0), marker='x',linestyle='None',markersize=4)
plt.gca().invert_yaxis()
plt.show(block=False)
ax.grid(linewidth=0.5)
ax.set_xlim([20,420])
ax.set_ylim([20,420])
ax.set_xlabel('Extent in km')
ax.set_ylabel('Extent in km')
ax.set_xticks([20,120,220,320,420])
ax.set_yticks([20,120,220,320,420])
ax.set_xticklabels((ax.get_xticks()-20) * 0.1)
ax.set_yticklabels((ax.get_yticks()-20) * 0.1)
ax.legend((pnt,cross),('Accepted Track','Rejected Track'),numpoints=1)
#fig.savefig('/home/zmaw/u300675/ma_rainprog/tracks.pgf')



#ax.invert_yaxis()

import cv2
import numpy as np
import matplotlib.pyplot as plt
dummy=np.zeros([30,30])
dummy[15:16,15:16]=1
dummy[10:12,20:22]=1
dummy[20:23,10:13]=1
plt.rcParams.update({'font.size': 22})
fig, ax1 = plt.subplots(1, figsize=(10, 8))
im = ax1.imshow(dummy)
cb = plt.colorbar(im)
dummy2 = cv2.filter2D(dummy,-1,np.flipud(self.kernel))
fig, ax2 = plt.subplots(1, figsize=(10, 8))
ax2.imshow(dummy2)
cb = plt.colorbar(im)
ax2.set_xticks([0,5,10,15,20,25])
ax1.set_xticks([0,5,10,15,20,25])
ax1.grid()
ax2.grid()


import matplotlib.patches as mpatches
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM,GoogleTiles
import matplotlib.pyplot as plt
import matplotlib.dates
import matplotlib.patches as mpatches
import numpy as np

xy_dwd = [10.04683,54.0044]
xy_emden = [7.0238,53.3387]
xy_dwd_radars=[[12.0581,54.1757],[9.6945,52.4601],[11.1761,53.3387],[13.8582,52.6487],[6.9671,51.4056]
    ,[8.802,51.3112],[13.7687,51.1245],[11.1350,50.5001],[6.5485,50.1097],[8.7129,49.9847],[12.4028,49.5407],[9.7828,48.5853],[12.1018,48.1747],[8.0036,47.8736],[10.2192,48.0421]]

google = GoogleTiles()
mercator = google.crs
xy_dwd_radars_T=['']*len(xy_dwd_radars)

for t in range(len(xy_dwd_radars)):
    xy_dwd_radars_T[t] = mercator.transform_point(xy_dwd_radars[t][0],xy_dwd_radars[t][1],ccrs.PlateCarree())

xy_emden_T = mercator.transform_point(xy_emden[0],xy_emden[1],ccrs.PlateCarree())
xy_boo_T = mercator.transform_point(xy_dwd[0],xy_dwd[1],ccrs.PlateCarree())
google = GoogleTiles()
mercator = google.crs
osm_tiles = OSM()
plt.rcParams["figure.figsize"] = (11,7)
plt.rcParams.update({'font.size': 10.5})
fig2 = plt.figure()
ax2 = fig2.add_subplot(111,projection=mercator)

ax2.set_extent([4, 17, 46, 56])
ax2.add_image(osm_tiles, 6, interpolation='bilinear')

for i in range(len(xy_dwd_radars_T)):
    radarCircle_dwd = mpatches.Circle(xy=xy_dwd_radars_T[i], radius=250000, color='k', linewidth=0.5, fill=0, transform=mercator)
    ax2.add_patch(radarCircle_dwd)
    ax2.scatter(xy_dwd_radars_T[i][0], xy_dwd_radars_T[i][1], color='k', marker='x')

radarCircle_dwd = mpatches.Circle(xy=xy_emden_T, radius=250000, color='k', linewidth=0.5, fill=0, transform=mercator)
ax2.add_patch(radarCircle_dwd)
ax2.scatter(xy_emden_T[0], xy_emden_T[1], color='k', marker='x')

radarCircle_dwd = mpatches.Circle(xy=xy_boo_T, radius=250000, color='r', linewidth=1, fill=0, transform=mercator)
ax2.add_patch(radarCircle_dwd)
ax2.scatter(xy_boo_T[0], xy_boo_T[1], color='r', marker='x')
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax2.xaxis.set_major_formatter(lon_formatter)
ax2.yaxis.set_major_formatter(lat_formatter)
ax2.set_xticks([5,7,9,11,13,15], crs=ccrs.PlateCarree())
ax2.set_yticks([47,49,51,53,55], crs=ccrs.PlateCarree())
ax2.grid()
fig2.savefig('/home/zmaw/u300675/ma_rainprog/dwdradars.pgf')

plt.show()
plt.rcParams.update({'font.size': 11})
plt.rcParams["figure.figsize"] = (6,6)


np.trapz(results[t].roc_hr[-1,::-1],results[t].roc_far[-1,::-1])


plt.rcParams.update({'font.size': 10.5})
fig, ax = plt.subplots(1)
dataArea1 =self.nested_data[t+1,
                        (int(field.maxima[0, 1]) - self.cRange * 2):(int(field.maxima[0, 1]) + self.cRange * 2),
                        (int(field.maxima[0, 2]) - self.cRange * 2):(int(field.maxima[0, 2]) + self.cRange * 2)]
dat = ax.imshow(dataArea1, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1),cmap=newcmap)
plt.plot(dataArea1.shape[0]/2,dataArea1.shape[0]/2,marker='X',color='red',markersize=8)
plt.plot(dataArea1.shape[0]/2+int(cIdx[1] - 0.5 * len(c)),dataArea1.shape[0]/2+int(cIdx[0] - 0.5 * len(c)),marker='X',color='black',markersize=8)
s = plt.colorbar(dat, format=matplotlib.ticker.ScalarFormatter(),fraction=0.046, pad=0.04)
s.set_label('Precipitation in mm/h')
s.set_clim(0.1, 100)
s.set_ticks(contours)
s.draw_all()
ax.set_xticks([0,10,20,30])
ax.set_yticks([0,10,20,30])
ax.set_xticklabels((ax.get_xticks()*0.1).astype(int))
ax.set_yticklabels((ax.get_yticks()*0.1).astype(int))
ax.set_xlabel('Extent in km')
ax.set_ylabel('Extent in km')
ax.grid(linewidth=0.5)
ax.invert_yaxis()
plt.tight_layout(pad=0)
fig.savefig('/home/zmaw/u300675/ma_rainprog/displacement_t1.pgf')

fig, ax = plt.subplots(1)
dataArea1 =self.nested_data[t,
                       (int(field.maxima[0, 1]) - self.cRange * 2):(int(field.maxima[0, 1]) + self.cRange * 2),
                       (int(field.maxima[0, 2]) - self.cRange * 2):(int(field.maxima[0, 2]) + self.cRange * 2)]
dat = ax.imshow(dataArea1, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1),cmap=newcmap)
corrAreaRect = mpatches.Rectangle((dataArea1.shape[0]/4,dataArea1.shape[0]/4),dataArea1.shape[0]/2,dataArea1.shape[0]/2,color='r',linewidth=2,fill=0)
ax.add_patch(corrAreaRect)
plt.plot(dataArea1.shape[0]/2,dataArea1.shape[0]/2,marker='X',color='red',markersize=8)
s = plt.colorbar(dat, format=matplotlib.ticker.ScalarFormatter(),fraction=0.046, pad=0.04)
s.set_label('Precipitation in mm/h')
s.set_clim(0.1, 100)
s.set_ticks(contours)
s.draw_all()
ax.set_xticks([0,10,20,30])
ax.set_yticks([0,10,20,30])
ax.set_xticklabels((ax.get_xticks()*0.1).astype(int))
ax.set_yticklabels((ax.get_yticks()*0.1).astype(int))
ax.set_xlabel('Extent in km')
ax.set_ylabel('Extent in km')
ax.grid(linewidth=0.5)
ax.invert_yaxis()
plt.tight_layout(pad=0)
fig.savefig('/home/zmaw/u300675/ma_rainprog/displacement_t0.pgf')

t = 0
fig, ax = plt.subplots(1)
im = ax.imshow(lawr.nested_data[lawr.startTime + t, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1),
               cmap=newcmap)
plt.show(block=False)
maximas = ['']
for x in range(len(lawr.progField.activeFields)):
    o, = plt.plot(lawr.progField.activeFields[x].maxima[0][2], lawr.progField.activeFields[x].maxima[0][1], 'rX')
    proxCircle = mpatches.Circle(
        (lawr.progField.activeFields[x].maxima[0][2], lawr.progField.activeFields[x].maxima[0][1]),
        3000 / lawr.resolution, color='k', linewidth=1, fill=0)
    ax.add_patch(proxCircle)
s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter())
s.set_clim(0.1, 100)
s.set_label('Precipitation in mm/h')
s.set_ticks(contours)
ax.grid(linewidth=0.5)
ax.set_xticks([0,100,200,300,400])
ax.set_yticks([0,100,200,300,400])
ax.set_xlabel('Extent in km')
ax.set_ylabel('Extent in km')
ax.set_xticklabels((ax.get_xticks()*0.1).astype(int))
ax.set_yticklabels((ax.get_yticks()*0.1).astype(int))
s.draw_all()
ax.invert_yaxis()
fig.savefig('/home/zmaw/u300675/ma_rainprog/maxima_overview_zoom.pgf')

t = 0
fig, ax = plt.subplots(1)
im = ax.imshow(dwd.nested_data[0, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=newcmap)
plt.show(block=False)
maximas = ['']
for x in range(len(dwd.progField.activeFields)):
    maximas.append(dwd.progField.activeFields[x].maxima[0][2:0:-1])
    o, = plt.plot(dwd.progField.activeFields[x].maxima[0][2],dwd.progField.activeFields[x].maxima[0][1], 'rX')
s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter())
s.set_clim(0.1, 100)
s.set_ticks(contours)
s.set_label('Precipitation in mm/h')
ax.set_xlim([50,448])
ax.set_ylim([448,50])
ax.grid(linewidth=0.5)
s.draw_all()
ax.set_xticks([50,150,250,350,450])
ax.set_yticks([50,150,250,350,450])
ax.set_xlabel('Extent in km')
ax.set_ylabel('Extent in km')
ax.set_xticklabels((ax.get_xticks()-50) * 0.25)
ax.set_yticklabels((ax.get_yticks()-50) * 0.25)
ax.invert_yaxis()
fig.savefig('/home/zmaw/u300675/ma_rainprog/maximaOverview.pgf')


plt.rcParams["figure.figsize"] = (2.8,2.8)
testcmap = plt.get_cmap('jet', 11)    # 11 discrete colors
testcmap.set_under('white')
test2 = np.zeros([20,20])
test2[6,13] = 1
test2[11:13,11:13] =1
fig, ax = plt.subplots(1)
dat = ax.imshow(test2, cmap=testcmap)
ax.set_xlabel('Pixels')
ax.set_xticks([0,5,10,15])
ax.set_ylabel('Pixels')
s = plt.colorbar(dat, fraction=0.046, pad=0.04)
s.set_clim(0.001,1)
s.set_ticks([0.0, 0.2,0.4,0.6,0.8,1.0])
s.set_ticklabels([0.0, 0.2,0.4,0.6,0.8,1.0])
s.draw_all()
s.set_label('Arbitrary Units')
ax.grid()
ax.invert_yaxis()
plt.tight_layout()
fig.savefig('/home/zmaw/u300675/ma_rainprog/prognosis_t0.pgf')

fig, ax = plt.subplots(1)
dat = ax.imshow(cv2.filter2D(test2,-1,lawr.kernel), cmap=testcmap)
ax.set_xlabel('Pixels')
ax.set_xticks([0,5,10,15])
ax.set_ylabel('Pixels')
s = plt.colorbar(dat, fraction=0.046, pad=0.04)
s.set_clim(0.001,1)
s.set_ticks([0.0, 0.2,0.4,0.6,0.8,1.0])
s.set_ticklabels([0.0, 0.2,0.4,0.6,0.8,1.0])
s.draw_all()
s.set_label('Arbitrary Units')
ax.grid()
ax.invert_yaxis()
plt.tight_layout()
fig.savefig('/home/zmaw/u300675/ma_rainprog/prognosis_t1.pgf')






plt.rcParams["figure.figsize"] = (4,4)
fig, ax = plt.subplots(1)
im = ax.imshow(test3, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1),cmap=newcmap)
s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter(),fraction=0.046, pad=0.04)
s.set_clim(0.1, 100)
s.set_label('Precipitation in mm/h')
s.set_ticks(contours)
ax.grid(linewidth=0.5)
ax.set_xticks([0,100,200,300,400])
ax.set_yticks([0,100,200,300,400])
ax.set_xlabel('Extent in km')
ax.set_ylabel('Extent in km')
ax.set_xticklabels((ax.get_xticks()*0.1).astype(int))
ax.set_yticklabels((ax.get_yticks()*0.1).astype(int))
s.draw_all()
ax.invert_yaxis()
plt.tight_layout()

fig.savefig('/home/zmaw/u300675/ma_rainprog/attenuation_boo.pgf')

fig, ax = plt.subplots(1)
im = ax.imshow(lawr.nested_data[80,:,:], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1),cmap=newcmap)
ellipse = mpatches.Ellipse((110,320),120,240,angle=-40,color='r',linewidth=1,fill=0)
ax.add_patch(ellipse)
ellipse = mpatches.Ellipse((220,350),70,140,angle=-40,color='r',linewidth=1,fill=0)
ax.add_patch(ellipse)
s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter(),fraction=0.046, pad=0.04)
s.set_clim(0.1, 100)
s.set_label('Precipitation in mm/h')
s.set_ticks(contours)
ax.grid(linewidth=0.5)
ax.set_xticks([0,100,200,300,400])
ax.set_yticks([0,100,200,300,400])
ax.set_xlabel('Extent in km')
ax.set_ylabel('Extent in km')
ax.set_xticklabels((ax.get_xticks()*0.1).astype(int))
ax.set_yticklabels((ax.get_yticks()*0.1).astype(int))
s.draw_all()
ax.invert_yaxis()
plt.tight_layout()

fig.savefig('/home/zmaw/u300675/ma_rainprog/attenuation_lawr.pgf')






plt.rcParams["figure.figsize"] = (3.2,3.2)
plt.rcParams.update({'font.size': 10.5})
##prog 30 min
progdata =  np.copy(lawr.probabilities[60,:,:])
progdata[lawr.dist_nested > lawr.r[-1]]=0
test = np.copy(progdata)
test[test<0.01]=np.nan

fig, ax = plt.subplots(1)
im = ax.imshow(test, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1),cmap=plt.get_cmap('BuPu',10))
s = plt.colorbar(im, fraction=0.046, pad=0.04)
ellipse = mpatches.Ellipse((400,205),60,80,angle=0,color='g',linewidth=1,fill=0)
ax.add_patch(ellipse)
ellipse = mpatches.Ellipse((370,280),60,60,angle=-40,color='g',linewidth=1,fill=0)
ax.add_patch(ellipse)
#ellipse = mpatches.Ellipse((270,110),160,100,angle=90,color='r',linewidth=1,fill=0)
#ax.add_patch(ellipse)
s.set_clim(0, 1)
s.set_label('Precipitation probability')
s.set_ticks([0,0.2,0.4,0.6,0.8,1])
s.set_ticklabels([0.0,0.2,0.4,0.6,0.8,1.0])
ax.grid(linewidth=0.5)
ax.set_xticks([0,100,200,300,400])
ax.set_yticks([0,100,200,300,400])
ax.set_xlabel('Extent in km')
ax.set_ylabel('Extent in km')
ax.set_xticklabels((ax.get_xticks()*0.1).astype(int))
ax.set_yticklabels((ax.get_yticks()*0.1).astype(int))
s.draw_all()
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('/home/zmaw/u300675/ma_rainprog/prog30_std7.pgf')

hhg_60 = np.copy(lawr.nested_data[lawr.progStartIdx+60,:,:])
hhg_60[hhg_60<0.5]=0
hhg_60[hhg_60>0.5]=1
fig, ax = plt.subplots(1)
im = ax.imshow(hhg_60,cmap=plt.get_cmap('Greys',2))
ellipse = mpatches.Ellipse((370,280),60,60,angle=-40,color='g',linewidth=1,fill=0)
ax.add_patch(ellipse)
ellipse = mpatches.Ellipse((400,205),60,80,angle=0,color='g',linewidth=1,fill=0)
ax.add_patch(ellipse)
#ellipse = mpatches.Ellipse((200,110),110,100,angle=-40,color='b',linewidth=1,fill=0)
#ax.add_patch(ellipse)
s = plt.colorbar(im,fraction=0.046, pad=0.04)
s.set_clim(0.0001,1)
s.set_ticks([0,1])
s.set_ticklabels([0.0,1.0])
s.set_label('Precipitation > 0.5 mm/h')
ax.grid(linewidth=0.5)
ax.set_xticks([0,100,200,300,400])
ax.set_yticks([0,100,200,300,400])
ax.set_xlabel('Extent in km')
ax.set_ylabel('Extent in km')
ax.set_xticklabels((ax.get_xticks()*0.1).astype(int))
ax.set_yticklabels((ax.get_yticks()*0.1).astype(int))
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('/home/zmaw/u300675/ma_rainprog/real30_std7.pgf')

hhg_0 = np.copy(lawr.nested_data[lawr.progStartIdx,:,:])
hhg_0[hhg_0<0.5]=0
hhg_0[hhg_0>0.5]=1
fig, ax = plt.subplots(1)
im = ax.imshow(hhg_0,cmap=plt.get_cmap('Greys',2))
#ellipse = mpatches.Ellipse((370,230),110,100,angle=-40,color='b',linewidth=1,fill=0)
#ax.add_patch(ellipse)
s = plt.colorbar(im,fraction=0.046, pad=0.04)
s.set_clim(0.0001,1)
s.set_ticks([0,1])
s.set_ticklabels([0.0,1.0])
s.set_label('Precipitation > 0.5 mm/h')
ax.grid(linewidth=0.5)
ax.set_xticks([0,100,200,300,400])
ax.set_yticks([0,100,200,300,400])
ax.set_xlabel('Extent in km')
ax.set_ylabel('Extent in km')
ax.set_xticklabels((ax.get_xticks()*0.1).astype(int))
ax.set_yticklabels((ax.get_yticks()*0.1).astype(int))
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('/home/zmaw/u300675/ma_rainprog/real0_std7.pgf')


hhg_120 = np.copy(lawr.nested_data[lawr.progStartIdx+119,:,:])
hhg_120[hhg_120<0.5]=0
hhg_120[hhg_120>0.5]=1
fig, ax = plt.subplots(1)
im = ax.imshow(hhg_120,cmap=plt.get_cmap('Greys',2))
s = plt.colorbar(im,fraction=0.046, pad=0.04)
s.set_clim(0.0001,1)
s.set_ticks([0,1])
s.set_ticklabels([0.0,1.0])
s.set_label('Precipitation > 0.5 mm/h')
ax.grid(linewidth=0.5)
ax.set_xticks([0,100,200,300,400])
ax.set_yticks([0,100,200,300,400])
ax.set_xlabel('Extent in km')
ax.set_ylabel('Extent in km')
ax.set_xticklabels((ax.get_xticks()*0.1).astype(int))
ax.set_yticklabels((ax.get_yticks()*0.1).astype(int))
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('/home/zmaw/u300675/ma_rainprog/real60_std7.pgf')


progdata =  np.copy(lawr.probabilities[-1,:,:])
progdata[lawr.dist_nested > lawr.r[-1]]=0
test = np.copy(progdata)
test[test<0.01]=np.nan

fig, ax = plt.subplots(1)
im = ax.imshow(test, norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1),cmap=plt.get_cmap('BuPu',10))
s = plt.colorbar(im, fraction=0.046, pad=0.04)
s.set_clim(0, 1)
s.set_label('Precipitation probability')
s.set_ticks([0,0.2,0.4,0.6,0.8,1])
s.set_ticklabels([0.0,0.2,0.4,0.6,0.8,1.0])
ax.grid(linewidth=0.5)
ax.set_xticks([0,100,200,300,400])
ax.set_yticks([0,100,200,300,400])
ax.set_xlabel('Extent in km')
ax.set_ylabel('Extent in km')
ax.set_xticklabels((ax.get_xticks()*0.1).astype(int))
ax.set_yticklabels((ax.get_yticks()*0.1).astype(int))
s.draw_all()
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('/home/zmaw/u300675/ma_rainprog/prog60_std7.pgf')


##rse error
probs = np.copy(lawr.probabilities[60,:,:])
probs[lawr.dist_nested>lawr.r[-1]] = 0

rse =  probs-(lawr.nested_data[lawr.progStartIdx+60,:,:]>0.5)
rse[lawr.dist_nested>lawr.r[-1]] = np.nan
rse_s = np.square(rse)
rse_s[rse_s<0.01]=np.nan

fig, ax = plt.subplots(1)
im = ax.imshow(rse_s, cmap=plt.get_cmap('Reds',10))
ellipse = mpatches.Ellipse((370,280),60,60,angle=-40,color='g',linewidth=1,fill=0)
ax.add_patch(ellipse)
ellipse = mpatches.Ellipse((400,205),60,80,angle=0,color='g',linewidth=1,fill=0)
ax.add_patch(ellipse)
s = plt.colorbar(im, fraction=0.046, pad=0.04)
s.set_clim(0, 1)
s.set_label('Squared difference')
ax.grid(linewidth=0.5)
ax.set_xticks([0,100,200,300,400])
ax.set_yticks([0,100,200,300,400])
ax.set_xlabel('Extent in km')
ax.set_ylabel('Extent in km')
ax.set_xticklabels((ax.get_xticks()*0.1).astype(int))
ax.set_yticklabels((ax.get_yticks()*0.1).astype(int))
s.draw_all()
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('/home/zmaw/u300675/ma_rainprog/diff30_std7.pgf')

probs = np.copy(lawr.probabilities[-1,:,:])
probs[lawr.dist_nested>lawr.r[-1]] = 0

rse =  probs-(lawr.nested_data[lawr.progStartIdx+119,:,:]>0.5)
rse[lawr.dist_nested>lawr.r[-1]] = np.nan
rse_s = np.square(rse)
rse_s[rse_s<0.01]=np.nan

fig, ax = plt.subplots(1)
im = ax.imshow(rse_s, cmap=plt.get_cmap('Reds',10))
s = plt.colorbar(im, fraction=0.046, pad=0.04)
s.set_clim(0, 1)
s.set_label('Squared difference')
ax.grid(linewidth=0.5)
ax.set_xticks([0,100,200,300,400])
ax.set_yticks([0,100,200,300,400])
ax.set_xlabel('Extent in km')
ax.set_ylabel('Extent in km')
ax.set_xticklabels((ax.get_xticks()*0.1).astype(int))
ax.set_yticklabels((ax.get_yticks()*0.1).astype(int))
s.draw_all()
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('/home/zmaw/u300675/ma_rainprog/diff60_std7.pgf')


import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM,GoogleTiles
import matplotlib.pyplot as plt
import matplotlib.dates
import matplotlib.patches as mpatches
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

plt.rcParams["figure.figsize"] = (9,6)
plt.rcParams.update({'font.size': 10.5})

google = GoogleTiles()
mercator = google.crs
osm_tiles = OSM()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111,projection=mercator)

ax2.set_extent([8.8850204382049416, 11.092690616402427, 52.907742087109092, 54.219436438416714])
ax2.add_image(osm_tiles, 8, interpolation='bilinear')

ax2.imshow(dwd.nested_data[4,:,:], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=cmap, extent= (8.8850204382049416, 11.092690616402427, 52.907742087109092, 54.219436438416714), transform=mercator)

lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax2.xaxis.set_major_formatter(lon_formatter)
ax2.yaxis.set_major_formatter(lat_formatter)
ax2.set_xticks([8,9,10,11,12], crs=ccrs.PlateCarree())
ax2.set_yticks([53,54,55], crs=ccrs.PlateCarree())

import geopandas as gpd
alpha = 0.7
pathshp = '/scratch/local1/shapefiles/hamburg-latest-free.shp'
borders = gpd.read_file(pathshp + '/Hamburg_AL4.shp')
hh_streets = gpd.read_file(pathshp + '/gis_osm_roads_free_1.shp')
ns_streets = gpd.read_file('/scratch/local1/shapefiles/niedersachsen-latest-free.shp/gis_osm_roads_free_1.shp')
sh_streets = gpd.read_file('/scratch/local1/shapefiles/schleswig-holstein-latest-free.shp/gis_osm_roads_free_1.shp')
rivers = gpd.read_file(pathshp + '/gis_osm_water_a_free_1.shp')
rivers[np.logical_or((rivers['code'] == 8200), (rivers['code'] == 8202))].plot(ax=ax1, color='b', alpha=alpha)
hh_streets[np.logical_or((hh_streets['code'] == 5111), (hh_streets['code'] == 5112))].plot(ax=ax1, color='r',
                                                                                           alpha=alpha)
hh_streets[hh_streets['code'] == 5113].plot(ax=ax1, color=[1, 0.2, 0], alpha=alpha)
sh_streets[np.logical_or((sh_streets['code'] == 5111), (sh_streets['code'] == 5112))].plot(ax=ax1, color='r',
                                                                                           alpha=alpha)
ns_streets[np.logical_or((ns_streets['code'] == 5111), (ns_streets['code'] == 5112))].plot(ax=ax1, color='r',
                                                                                           alpha=alpha)
sh_streets[sh_streets['code'] == 5113].plot(ax=ax1, color=[1, 0.2, 0], alpha=alpha)
ns_streets[ns_streets['code'] == 5113].plot(ax=ax1, color=[1, 0.2, 0], alpha=alpha)


borders.plot(ax=ax1, facecolor='none', edgecolor='k')

pathshp = '/scratch/local1/shapefiles/hamburg-latest-free.shp'
borders = gpd.read_file(pathshp + '/Hamburg_AL4.shp')
extent = ([8.8850204382049416, 11.092690616402427, 52.907742087109092, 54.219436438416714])
plt.rcParams["figure.figsize"] = (9,9)
plt.rcParams.update({'font.size': 16})
fig,ax= plt.subplots(1)
#ax2.set_extent([8.8850204382049416, 11.092690616402427, 52.907742087109092, 54.219436438416714])
im = ax.imshow(np.flipud(test[4,:,:]), norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=newcmap, extent= (8.8850204382049416, 11.092690616402427, 52.907742087109092, 54.219436438416714))
s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter(), fraction=0.046, pad=0.04)
s.set_clim(0,100)
s.set_ticks(contours)
ax.grid(linewidth=0.5)
s.draw_all()
alpha = 0.7
borders.plot(ax=ax, facecolor='none', edgecolor='k')
ax.set_xlim([dwd.Lon.min(),dwd.Lon.max()])
ax.set_ylim([dwd.Lat.min(),dwd.Lat.max()])
time_text = ax.text(9.1,54,datetime.fromtimestamp(dwd.time[0]).strftime("%d/%m/%Y %H:%M:%S"))

fig,ax= plt.subplots(1)
contours = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]

for t in range(len(dwd.nested_data)):
#ax2.set_extent([8.8850204382049416, 11.092690616402427, 52.907742087109092, 54.219436438416714])
    if t==0:
        im = ax.imshow(np.flipud(test[t,:,:]), norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=newcmap, extent= (8.8850204382049416, 11.092690616402427, 52.907742087109092, 54.219436438416714))
        s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter(), fraction=0.046, pad=0.04)
        s.set_clim(0.1,100)
        s.set_ticks(contours)
        s.draw_all()
        s.set_label('Precipitation in mm/h')
        alpha = 0.7
        borders.plot(ax=ax, facecolor='none', edgecolor='k')
        ax.set_xlim([dwd.Lon.min(),dwd.Lon.max()])
        ax.set_ylim([dwd.Lat.min(),dwd.Lat.max()])
        ax.grid(linewidth=0.5)
        plt.tight_layout()
        time_text = ax.text(9.1,54,datetime.fromtimestamp(dwd.time[t]).strftime("%d/%m/%Y %H:%M:%S"))
    else:
        im.set_data(np.flipud(test[t,:,:]))
        time_text.set_text(datetime.fromtimestamp(dwd.time[t]).strftime("%d/%m/%Y %H:%M:%S"))
    plt.savefig('/home/zmaw/u300675/dwd_radar_'+str(t)+'.png')

import geopandas as gpd
alpha = 0.7

pathshp = '/scratch/local1/shapefiles/hamburg-latest-free.shp'
borders = gpd.read_file(pathshp + '/Hamburg_AL4.shp')
import matplotlib.animation as animation
test2 = np.copy(lawr.nested_data)
test2[test2<0.01]=np.nan
fig,ax=plt.subplots(1)
im = ax.imshow(np.flipud(test2[t, :, :]), norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=newcmap,
               extent=(lawr.Lon.min(), lawr.Lon.max(), lawr.Lat.min(), lawr.Lat.max()))
s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter(), fraction=0.046, pad=0.04)
s.set_clim(0.1,100)
contours = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
s.set_ticks(contours)
s.draw_all()
s.set_label('Precipitation in mm/h')
alpha = 0.7
borders.plot(ax=ax, facecolor='none', edgecolor='k')
ax.set_xlim([lawr.Lon.min(),lawr.Lon.max()])
ax.set_ylim([lawr.Lat.min(),lawr.Lat.max()])
ax.grid(linewidth=0.5)
plt.tight_layout()
time_text = ax.text(9.7,53.72,datetime.fromtimestamp(lawr.time[t]).strftime("%d/%m/%Y %H:%M:%S"))

def animate(t):
    im.set_data(np.flipud(test2[t,:,:]))
    time_text.set_text(datetime.fromtimestamp(lawr.time[t]).strftime("%d/%m/%Y %H:%M:%S"))
    return [im]


anim = animation.FuncAnimation(fig, animate,frames=len(lawr.nested_data[0:41]),
                                               interval=200, repeat=1,
                                               blit=True)
writer = animation.FFMpegWriter( fps=5, bitrate=2000,extra_args=['-vcodec', 'h264',
                                      '-pix_fmt', 'yuv420p'])
anim.save('/home/zmaw/u300675/lawr_radar_movie.mp4', writer=writer)



##tracks
col = np.concatenate(
    [np.random.rand(1,np.max(lawr.progField.activeIds))*0.7, np.random.rand(2, np.max(lawr.progField.activeIds))])
plt.rcParams.update({'font.size': 16})
plt.rcParams["figure.figsize"] = (6,6)
fig,ax = plt.subplots(1)
for i, field in enumerate(lawr.progField.activeFields):
    for t in field.histMaxima:
        pnt, = plt.plot(*np.transpose(t[0][2:0:-1]), color=col[:, field.id - 1], marker='o',linestyle='None',markersize=4)
pnt, = plt.plot(-100,-100,color='black',marker='o',linestyle='None',markersize=4)
for i, field in enumerate(lawr.progField.inactiveFields):
    for t in field.histMaxima:
        cross, = plt.plot(*np.transpose(t[0][2:0:-1]), color=(1, 0, 0), marker='x',linestyle='None',markersize=4)
plt.gca().invert_yaxis()
plt.show(block=False)
ax.grid(linewidth=0.5)
ax.set_xlim([20,420])
ax.set_ylim([20,420])
ax.set_xlabel('Extent in km')
ax.set_ylabel('Extent in km')
ax.set_xticks([20,120,220,320,420])
ax.set_yticks([20,120,220,320,420])
ax.set_xticklabels(((ax.get_xticks()-20) * 0.1).astype(int))
ax.set_yticklabels(((ax.get_yticks()-20) * 0.1).astype(int))
ax.legend((pnt,cross),('Accepted Track','Rejected Track'),numpoints=1)
plt.savefig('/home/zmaw/u300675/tracks.png')

i = 0
for t in range(prog - lawr.trainTime, prog):
    if t == prog - lawr.trainTime:
        fig, ax = plt.subplots(1)
        im = ax.imshow(lawr.nested_data[t + lawr.startTime + lawr.trainTime, :, :],
                       norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=newcmap)
        plt.show(block=False)
        o, = ax.plot(*np.transpose(lawr.progField.return_histMaxima(t - prog + lawr.trainTime)[:, 2:0:-1]), 'ko',
                     markersize=10)

        s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter(), fraction=0.046, pad=0.04)
        s.set_clim(0.1, 100)
        s.set_ticks(contours)
        s.set_label('Precipitation in mm/h')
        ax.grid(linewidth=0.5)
        ax.set_xticks([0, 100, 200, 300, 400])
        ax.set_yticks([0, 100, 200, 300, 400])
        ax.set_xlabel('Extent in km')
        ax.set_ylabel('Extent in km')
        ax.set_xticklabels((ax.get_xticks() * 0.1).astype(int))
        ax.set_yticklabels((ax.get_yticks() * 0.1).astype(int))
        s.draw_all()
        ax.invert_yaxis()
        plt.tight_layout()
        time_text = ax.text(30, 400, datetime.fromtimestamp(lawr.time[t + lawr.startTime + lawr.trainTime]).strftime(
            "%d/%m/%Y %H:%M:%S"))
        # s.set_ticklabels(contourLabels)

plt.savefig('/home/zmaw/u300675/analysis_-1.png')

i = 0
for t in range(prog - lawr.trainTime + 1, prog):
    if t == prog - lawr.trainTime + 1:
        fig, ax = plt.subplots(1)
        im = ax.imshow(lawr.nested_data[t + lawr.startTime + lawr.trainTime, :, :],
                       norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=newcmap)
        plt.show(block=False)
        n, = ax.plot(*np.transpose(lawr.progField.return_histMaxima(t - prog + lawr.trainTime - 1)[:, 2:0:-1]), 'ko',
                     markersize=10)
        o, = ax.plot(*np.transpose(lawr.progField.return_histMaxima(t - prog + lawr.trainTime)[:, 2:0:-1]), 'go',
                     alpha=0.8, markersize=10)
        s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter(), fraction=0.046, pad=0.04)
        s.set_clim(0.1, 100)
        s.set_ticks(contours)
        s.set_label('Precipitation in mm/h')
        ax.grid(linewidth=0.5)
        ax.set_xticks([0, 100, 200, 300, 400])
        ax.set_yticks([0, 100, 200, 300, 400])
        ax.set_xlabel('Extent in km')
        ax.set_ylabel('Extent in km')
        ax.set_xticklabels((ax.get_xticks() * 0.1).astype(int))
        ax.set_yticklabels((ax.get_yticks() * 0.1).astype(int))
        s.draw_all()
        time_text = ax.text(30, 400, datetime.fromtimestamp(lawr.time[t + lawr.startTime + lawr.trainTime]).strftime(
            "%d/%m/%Y %H:%M:%S"))
        ax.invert_yaxis()
        plt.tight_layout()
        # s.set_ticklabels(contourLabels)
    else:
        im.set_data(lawr.nested_data[t + lawr.startTime + lawr.trainTime, :, :])
        n.set_data(*np.transpose(lawr.progField.return_histMaxima(t - prog + lawr.trainTime - 1)[:, 2:0:-1]))
        o.set_data(*np.transpose(lawr.progField.return_histMaxima(t - prog + lawr.trainTime)[:, 2:0:-1]))

    time_text.set_text(
        datetime.fromtimestamp(lawr.time[t + lawr.startTime + lawr.trainTime]).strftime("%d/%m/%Y %H:%M:%S"))
    fig.savefig('/home/zmaw/u300675/analyis_' + str(i) + '.png')
    i += 1

i = 0
t = prog - 1
fig, ax = plt.subplots(1)
im = ax.imshow(lawr.nested_data[t + lawr.startTime + lawr.trainTime, :, :],
               norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=newcmap)
plt.show(block=False)
o, = ax.plot(*np.transpose(lawr.progField.return_histMaxima(t - prog + lawr.trainTime)[:, 2:0:-1]), 'go', alpha=1,
             markersize=10)
s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter(), fraction=0.046, pad=0.04)
s.set_clim(0.1, 100)
s.set_ticks(contours)
s.set_label('Precipitation in mm/h')
ax.grid(linewidth=0.5)
ax.set_xticks([0, 100, 200, 300, 400])
ax.set_yticks([0, 100, 200, 300, 400])
ax.set_xlabel('Extent in km')
ax.set_ylabel('Extent in km')
ax.set_xticklabels((ax.get_xticks() * 0.1).astype(int))
ax.set_yticklabels((ax.get_yticks() * 0.1).astype(int))
s.draw_all()
time_text = ax.text(30, 400, datetime.fromtimestamp(lawr.time[t + lawr.startTime + lawr.trainTime]).strftime(
    "%d/%m/%Y %H:%M:%S"))
ax.invert_yaxis()
plt.tight_layout()

plt.savefig('/home/zmaw/u300675/analysis_9.png')