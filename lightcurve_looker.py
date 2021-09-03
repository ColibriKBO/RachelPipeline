# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:14:51 2021

@author: rache
"""
#%% import statements
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import linecache
from astropy.io import fits

#%% load in single light curve
filename = './ColibriArchive/Green-2021-07-16/det_2021-06-23_045216_13_G.txt'

flux = pd.read_csv(filename, delim_whitespace = True, 
                   names = ['filename', 'time', 'flux'], comment = '#')

coords = linecache.getline(filename,6).split(': ')[1].split(' ')
eventfile = linecache.getline(filename,5).split(': ')[1].strip()
index = flux.loc[flux['filename'] == eventfile].index[0]

times = flux['time'] - flux['time'].min()
#flux['time'] = flux['time'] - flux['time'].min()

#x.append(float(coords[0]))
#y.append(float(coords[1]))
starnum = filename.split('_')[3]
med = np.median(flux['flux'])
std = np.std(flux['flux'])
SNR = med/std
eventtime = flux.loc[index, 'time']

#%% plot single light curve

fig, ax1 = plt.subplots()

biassub = './ColibriArchive/biassubtracted/202106023/20210623_00.52.14.567/sub_field1_25ms-E_0000002.fits'
starimage = fits.getdata(biassub)
pad = 25

# Define your sub-array
subarray = starimage[int(float(coords[0]))-pad:int(float(coords[0]))+pad,int(float(coords[1]))-pad:int(float(coords[1]))+pad]

left, bottom, width, height = [0.95, 0.6, 0.2, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])

ax1.plot(flux['time'], flux['flux'], label = 'light curve')
ax1.hlines(med, flux['time'].min(), flux['time'].max(), color = 'black', label = 'median: %i' % med)
ax1.hlines(med + std, flux['time'].min(), flux['time'].max(), linestyle = '--', color = 'black', label = 'stddev: %.3f' %std)
ax1.hlines(med - std, flux['time'].min(), flux['time'].max(), linestyle = '--', color = 'black')

ax1.vlines(eventtime, flux['flux'].min(), flux['flux'].max(), color = 'red', alpha = 0.4, label = 'Event time: %.3f' % eventtime)
ax1.set_xlabel('time (seconds)')
ax1.set_ylabel('Counts/49 pixels')
ax1.set_title('Star #%s [%.1f, %.1f], SNR = %.2f' %(starnum, float(coords[0]), float(coords[1]), SNR))
ax1.legend()

ax2.imshow(subarray, vmin = flux['flux'].min(), vmax = flux['flux'].max())

plt.show()
#plt.savefig('./ColibriArchive/2021-07-15/lightcurve_plots/star82_SNR14.png')
plt.close()


#%% plot whole directory of lightcurves

lightcurves = {}

directory = './ColibriArchive/Green-2021-07-16/'

#get a list of files and sort properly
files = os.listdir(directory)
files = [f for f in files if 'G.txt' in f]
files = sorted(files, key = lambda x: float(x.split('_')[3]))

for filename in files:
    if filename.endswith("G.txt"): 
        
        flux = pd.read_csv(directory+filename, delim_whitespace = True, 
                   names = ['filename', 'time', 'flux'], comment = '#')

        coords = linecache.getline(directory+filename,6).split(': ')[1].split(' ')
        eventfile = linecache.getline(directory+filename,5).split(': ')[1].strip()
        eventindex = flux.loc[flux['filename'] == eventfile].index[0]

        eventtime = flux.loc[index, 'time']

        starnum = filename.split('_')[3]
        med = np.median(flux['flux'])
        std = np.std(flux['flux'])
        SNR = med/std
        
        lightcurves[starnum] = {'flux': flux, 'coords': coords, 'event time': eventtime,
                               'median': med, 'std': std, 'SNR': SNR, 'eventfile':eventfile}
    else:
        continue
    
#%%plot all light curves

for key, info in lightcurves.items():
    med = info['median']
    std = info['std']
    flux = info['flux']
    eventtime = info['event time']
    coords = info['coords']
    SNR = info['SNR']
    eventfile = info['eventfile']
    
    fig, ax1 = plt.subplots()

    biassub = './ColibriArchive/biassubtracted/202106023/20210623_00.52.14.567/sub_field1_25ms-E_0000002.fits'
    #starimage = fits.getdata(eventfile)
    starimage = fits.getdata(biassub)
    pad = 100
    
    # Define your sub-array
    subarray = starimage[int(float(coords[0]))-pad:int(float(coords[0]))+pad,int(float(coords[1]))-pad:int(float(coords[1]))+pad]

    left, bottom, width, height = [0.95, 0.6, 0.2, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.imshow(subarray, vmin = 0, vmax = med)
    
    ax1.plot(flux['time'], flux['flux'])
    ax1.hlines(med, flux['time'].min(), flux['time'].max(), color = 'black', label = 'median: %i' % med)                                                                                                 
    ax1.hlines(med + std, flux['time'].min(), flux['time'].max(), linestyle = '--', color = 'black', label = 'stddev: %.3f' % std)
    ax1.hlines(med - std, flux['time'].min(), flux['time'].max(), linestyle = '--', color = 'black')
    
    ax1.vlines(eventtime, flux['flux'].min(), flux['flux'].max(), color = 'red', alpha = 0.4, label = 'Event time: %.3f' % eventtime)
    ax1.set_xlabel('time (seconds)')
    ax1.set_ylabel('Counts/49 pixels')
    ax1.set_title('Star #%s [%.1f, %.1f], SNR = %.2f' %(key, float(coords[0]), float(coords[1]), SNR))
    ax1.legend()

    #plt.show()
    plt.savefig(directory + 'lightcurves/star' + key + '.png', bbox_inches = 'tight')
    plt.close()


#%% load in square/circle aperture data 

square_flux = pd.read_csv('./ColibriArchive/2021-05-28/det_2020-08-20_034734_58_R.txt', delim_whitespace = True, 
                   names = ['filename', 'time', 'flux'], comment = '#')

circle_flux = pd.read_csv('./ColibriArchive/2021-05-28/circle/det_2020-08-20_034734_58_R.txt', delim_whitespace = True, 
                   names = ['filename', 'time', 'flux'], comment = '#')


#%% load in light curve data

ap1 = square_flux = pd.read_csv('./ColibriArchive/2021-06-07/ap_r1/det_2020-08-20_034736_5_R.txt', delim_whitespace = True, 
                   names = ['filename', 'time', 'flux'], comment = '#')
ap2 = square_flux = pd.read_csv('./ColibriArchive/2021-06-07/ap_r2/det_2020-08-20_034736_5_R.txt', delim_whitespace = True, 
                   names = ['filename', 'time', 'flux'], comment = '#')
ap3 = square_flux = pd.read_csv('./ColibriArchive/2021-06-07/ap_r3/det_2020-08-20_034736_5_R.txt', delim_whitespace = True, 
                   names = ['filename', 'time', 'flux'], comment = '#')

#%%subtract off flux additions and calculate median and stddev
ap1['flux'] = ap1['flux'] - (100*2*1)
ap1_med = np.median(ap1['flux'])
ap1_std = np.std(ap1['flux'])
ap2['flux'] = ap2['flux'] - (100*2*2)
ap2_med = np.median(ap2['flux'])
ap2_std = np.std(ap2['flux'])
ap3['flux'] = ap3['flux'] - (100*2*3)
ap3_med = np.median(ap3['flux'])
ap3_std = np.std(ap3['flux'])
#%%compare aperture sizes

plt.plot(ap1['time'], ap1['flux'], label = '3 x 3')
plt.plot(ap2['time'], ap2['flux'], label = '5 x 5')
plt.plot(ap3['time'], ap3['flux'], label = '7 x 7')

plt.xlabel('time (s)')
plt.ylabel('Flux')
plt.title('Star #5')
plt.legend()

plt.show()
plt.close()

plt.plot(ap1['time'], ap1['flux'], label = '3 x 3')
plt.hlines(ap1_med, ap1['time'].min(), ap1['time'].max(), label = ap1_med)
plt.hlines(ap1_med + ap1_std, ap1['time'].min(), ap1['time'].max(), linestyle = '--', label = ap1_std)
plt.hlines(ap1_med - ap1_std, ap1['time'].min(), ap1['time'].max(), linestyle = '--')

plt.xlabel('time (s)')
plt.ylabel('Flux')
plt.title('Star #5: 3x3')
plt.legend()

plt.show()
plt.close()

plt.plot(ap2['time'], ap2['flux'], label = '5 x 5')
plt.hlines(ap2_med, ap2['time'].min(), ap2['time'].max(), label = ap2_med)
plt.hlines(ap2_med + ap2_std, ap2['time'].min(), ap2['time'].max(), linestyle = '--', label = ap2_std)
plt.hlines(ap2_med - ap2_std, ap2['time'].min(), ap2['time'].max(), linestyle = '--')

plt.xlabel('time (s)')
plt.ylabel('Flux')
plt.title('Star #5: 5x5')
plt.legend()

plt.show()
plt.close()

plt.plot(ap3['time'], ap3['flux'], label = '7 x 7')
plt.hlines(ap3_med, ap3['time'].min(), ap3['time'].max(), label = ap3_med)
plt.hlines(ap3_med + ap3_std, ap3['time'].min(), ap3['time'].max(), linestyle = '--', label = ap3_std)
plt.hlines(ap3_med - ap3_std, ap3['time'].min(), ap3['time'].max(), linestyle = '--')

plt.xlabel('time (s)')
plt.ylabel('Flux')
plt.title('Star #5: 9x9')
plt.legend()

plt.show()
plt.close()
#%% make plot
plt.plot(square_flux['time'], square_flux['flux'], label = 'square ap')
plt.plot(circle_flux['time'], circle_flux['flux'], label = 'circle ap')

plt.xlabel('Time (s)')
plt.ylabel('Flux')
plt.legend()

plt.show()
plt.close()

plt.scatter(square_flux['flux'], circle_flux['flux'])
plt.plot(range(0, 300), range(0, 300))
plt.xlabel('square flux')
plt.ylabel('circle flux')

plt.show()
plt.close()

#%%new data
star0 = pd.read_csv('./ColibriArchive/2021-06-08/det_2021-06-02_083159_0_R.txt', delim_whitespace = True, 
                   names = ['filename', 'time', 'flux'], comment = '#')
star35 = pd.read_csv('./ColibriArchive/2021-06-08/det_2021-06-02_083200_35_R.txt', delim_whitespace = True, 
                   names = ['filename', 'time', 'flux'], comment = '#')
star77 = pd.read_csv('./ColibriArchive/2021-06-08/det_2021-06-02_083201_77_R.txt', delim_whitespace = True, 
                   names = ['filename', 'time', 'flux'], comment = '#')

#%%fix times

star0.loc[star0['time'] < 50., 'time'] = 60. + star0['time']

star35.loc[star35['time'] < 50., 'time'] = 60. + star35['time']

star77.loc[star77['time'] < 50., 'time'] = 60. + star77['time']
#%%get stats
star0['flux'] = star0['flux'] #- (100*2*1)
star0_med = np.median(star0['flux'])
star0_std = np.std(star0['flux'])
star35['flux'] = star35['flux'] #- (100*2*2)
star35_med = np.median(star35['flux'])
star35_std = np.std(star35['flux'])
star77['flux'] = star77['flux'] #- (100*2*3)
star77_med = np.median(star77['flux'])
star77_std = np.std(star77['flux'])

#%%plots
#plt.plot(star0['time'], star0['flux'], label = 'light curve')
plt.plot(star0['flux'], label = 'light curve')
#plt.xticks(range(len(star0['flux'])), star0['time'], rotation=50, horizontalalignment='right', weight='bold', size='large')
plt.hlines(star0_med, 0, 50, label = star0_med)
plt.hlines(star0_med + star0_std, 0, 50, linestyle = '--', label = star0_std)
plt.hlines(star0_med - star0_std, 0, 50, linestyle = '--')

plt.xlabel('time (s)')
plt.ylabel('Flux')
plt.title('Star #0: 3x3')
plt.legend()

plt.show()
plt.close()

#plt.plot(star35['time'], star35['flux'], label = 'light curve')
plt.plot(star35['flux'], label = 'light curve')
plt.hlines(star35_med, 0, 80, label = star35_med)
plt.hlines(star35_med + star35_std, 0, 80, linestyle = '--', label = star35_std)
plt.hlines(star35_med - star35_std, 0, 80, linestyle = '--')

plt.xlabel('time (s)')
plt.ylabel('Flux')
plt.title('Star #35: 3x3')
plt.legend()

plt.show()
plt.close()

#plt.plot(star77['time'], star77['flux'], label = 'Light curve')
plt.plot(star77['flux'], label = 'Light curve')
plt.hlines(star77_med,0, 50, label = star77_med)
plt.hlines(star77_med + star77_std, 0, 50, linestyle = '--', label = star77_std)
plt.hlines(star77_med - star77_std, 0, 50, linestyle = '--')

plt.xlabel('time (s)')
plt.ylabel('Flux')
plt.title('Star #77: 3x3')
plt.legend()

plt.show()
plt.close()