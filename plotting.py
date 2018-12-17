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
ax.set_xticklabels(ax.get_xticks()*0.25)
ax.set_yticklabels(ax.get_yticks()*0.25)
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
ax.set_xticklabels(ax.get_xticks()*0.25)
ax.set_yticklabels(ax.get_yticks()*0.25)
ax.grid(linewidth=0.5)
plt.show()


fig,ax = plt.subplots(1)
corr = ax.imshow(c, cmap=plt.get_cmap('inferno_r'))
plt.colorbar(corr)
plt.plot(24,24,marker='X',color='ghostwhite',markersize=8)
plt.plot(cIdx[1],cIdx[0],marker='X',color='black',markersize=8)
ax.set_xticklabels(ax.get_xticks()*0.25)
ax.set_yticklabels(ax.get_yticks()*0.25)
ax.grid(linewidth=0.5)
plt.show()

import matplotlib.patches as mpatches
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM,GoogleTiles
import matplotlib.pyplot as plt
import matplotlib.dates
import matplotlib.patches as mpatches

osm_tiles = OSM()
plt.rcParams["figure.figsize"] = (9,6)
plt.rcParams.update({'font.size': 22})
ax = plt.axes(projection=ccrs.PlateCarree())
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
xy_pat = [9.9734,53.56833]
xy_pat_T = mercator.transform_point(xy_pat[0],xy_pat[1],ccrs.PlateCarree())
radarCircle_pat = mpatches.Circle(xy=xy_pat_T, radius=20000*1.73, color='k', linewidth=1, fill=0,transform = mercator)
ax2.add_patch(radarCircle_pat)
x_rect = np.array([9.2167,10.7473])
y_rect = np.array([53.11,54.0215])
xy_rect=mercator.transform_points(ccrs.PlateCarree(),x_rect,y_rect)
radarRect = mpatches.Rectangle(xy_rect[0],width=xy_rect[1,0]-xy_rect[0,0],height=xy_rect[1,1]-xy_rect[0,1], color='b',fill=0)
ax2.add_patch(radarRect)

gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.2, linestyle='--')
gl2.xlabels_top = False
gl2.ylabels_left = False

plt.show()



plt.rcParams.update({'font.size': 22})
t = 0
fig, ax = plt.subplots(1, figsize=(10, 8))
im = ax.imshow(dwd.nested_data[0, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=newcmap)
plt.show(block=False)
o, = plt.plot(*np.transpose(dwd.progField.return_maxima(t)[:, 2:0:-1]), 'rX')
s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter())
s.set_clim(0.1, 100)
s.set_ticks(contours)
ax.set_xlim([50,448])
ax.set_ylim([448,50])
ax.grid(linewidth=0.5)
s.draw_all()
ax.set_xticklabels((ax.get_xticks()-50) * 0.25)
ax.set_yticklabels((ax.get_yticks()-50) * 0.25)

t = 0
fig, ax = plt.subplots(1, figsize=(10, 8))
im = ax.imshow(dwd.nested_data[0, :, :], norm=matplotlib.colors.SymLogNorm(vmin=0, linthresh=1), cmap=newcmap)
plt.show(block=False)
o, = plt.plot(*np.transpose(dwd.progField.return_maxima(t)[:, 2:0:-1]), 'rX')
maxima = dwd.progField.return_maxima(t)[:, 2:0:-1]
for u in range(len(maxima)):
    proxCircle = mpatches.Circle((maxima[u,0],maxima[u,1]),3000/dwd.resolution,color='r',linewidth=1,fill=0)
    ax.add_patch(proxCircle)
s = plt.colorbar(im, format=matplotlib.ticker.ScalarFormatter())
s.set_clim(0.1, 100)
s.set_ticks(contours)
ax.set_xlim([100,300])
ax.set_ylim([300,100])
ax.grid(linewidth=0.5)
s.draw_all()
ax.set_xticklabels(ax.get_xticks() * 0.25)
ax.set_yticklabels(ax.get_yticks() * 0.25)
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
ax.set_xlim([100,300])
ax.set_ylim([300,100])
ax.grid(linewidth=0.5)
ax.set_xticklabels(ax.get_xticks() * 0.25)
ax.set_yticklabels(ax.get_yticks() * 0.25)
ax.legend((pnt,cross),('Accepted Track','Rejected Track'),numpoints=1)
