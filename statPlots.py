#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 11:11:52 2021
'''make plots of mean, median, mode values for a night
@author: rbrown
"""

import numpy as np
import pandas as pd
from astropy.time import Time
import matplotlib.pyplot as plt

#%%
#one night version

scope = 'red'

data = pd.read_csv('./20210804_bias_stats.txt', delim_whitespace = True)
data[['day','hour']] = data['time'].str.split('T', expand = True)

#find time breaks in data (when a new folder was created)

#get list of different bias folders, the indices of where these start in the data frame
folders = data['filename'].str.split('\\', expand = True)[1]
folders, index = np.unique(folders, return_index = True)
labels = data['hour'][index]
labels = labels.str.split('.', expand = True)[0]

#minimum and maximum value in one night
lower = np.min(data.min(axis = 1))
lower = 96
upper = np.max(data.max(axis = 1))
upper = 99

#%%
#plot with single nights data with temperature
fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (12,4), gridspec_kw=dict(hspace = 0.05))


ax1.scatter(data['hour'], data['med'], label = 'median', s = 2)
ax1.scatter(data['hour'], data['mean'], label = 'mean', s = 2)
ax1.scatter(data['hour'], data['mode'], label = 'mode', s = 2)

ax1.set_title(scope + ' biases - ' + data.loc[0]['day'])
ax1.set_ylabel('image pixel value')
ax1.vlines(index, lower-0.2, upper+0.2, color = 'black', linewidth = 1)
ax1.set_xticks(index)
ax1.set_xticklabels(labels, rotation=20)
ax1.set_ylim(lower-0.2, upper+0.2)

ax1.legend()

ax2.scatter(data['hour'], data['baseTemp'], label = 'Base temp', s = 2)
ax2.scatter(data['hour'], data['FPGAtemp'], label = 'FGPA temp', s = 2)

ax2.vlines(index, 30, 60, color = 'black', linewidth = 1)
ax2.set_xlabel('time')
ax2.set_ylabel('Temperature (C)')
ax2.set_xticks(index)
ax2.set_xticklabels(labels,rotation=20)
#ax2.set_ylim(lower_13-0.2, upper_13+0.2)

ax2.legend()

#plt.savefig('./imageStats/both_bias_stats_' + scope + '.png')
plt.show()
plt.close()
#%%
#get data for each night
scope = 'red'

data_04 = pd.read_csv('./imageStats/20210804_bias_stats_' + scope + '.txt', delim_whitespace = True)
data_13 = pd.read_csv('./imageStats/202108013_bias_stats_' + scope + '.txt', delim_whitespace = True)

data_04[['day','hour']] = data_04['time'].str.split('T', expand = True)  
data_13[['day','hour']] = data_13['time'].str.split('T', expand = True)  

#%%
#find time breaks in data (when a new folder was created)

#get list of different bias folders, the indices of where these start in the data frame
folders_04 = data_04['filename'].str.split('\\', expand = True)[1]
folders_04, index_04 = np.unique(folders_04, return_index = True)
labels_04 = data_04['hour'][index_04]
labels_04 = labels_04.str.split('.', expand = True)[0]


folders_13 = data_13['filename'].str.split('\\', expand = True)[1]
folders_13, index_13 = np.unique(folders_13, return_index = True)
labels_13 = data_13['hour'][index_13]
labels_13 = labels_13.str.split('.', expand = True)[0]

#minimum and maximum value in one night
lower_04 = np.min(data_04.min(axis = 1))
upper_04 = np.max(data_04.max(axis = 1))

#minimum and max value in other night
lower_13 = np.min(data_13.min(axis = 1))
upper_13 = np.max(data_13.max(axis = 1))

#min and max value for both nights
lower = np.min([lower_13, lower_04])
upper = np.max([upper_13, upper_13])


#%%
#make plot with 2 nights data in side by side panels

fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True, figsize = (12,4), gridspec_kw=dict(wspace = 0.05))


ax1.scatter(data_04['hour'], data_04['med'], label = 'median', s = 2)
ax1.scatter(data_04['hour'], data_04['mean'], label = 'mean', s = 2)
ax1.scatter(data_04['hour'], data_04['mode'], label = 'mode', s = 2)

ax1.set_title(scope + ' biases - ' + data_04.loc[0]['day'])
ax1.set_ylabel('image pixel value')
ax1.vlines(index_04, lower-0.2, upper+0.2, color = 'black', linewidth = 1)
ax1.set_xlabel('time')
ax1.set_xticks(index_04)
ax1.set_xticklabels(labels_04,rotation=20)
ax1.set_ylim(lower-0.2, upper+0.2)


ax2.scatter(data_13['hour'], data_13['med'], label = 'median', s = 2)
ax2.scatter(data_13['hour'], data_13['mean'], label = 'mean', s = 2)
ax2.scatter(data_13['hour'], data_13['mode'], label = 'mode', s = 2)

ax2.set_title(scope + ' biases - ' + data_13.loc[0]['day'])
ax2.vlines(index_13, lower-0.2, upper+0.2, color = 'black', linewidth = 1)
ax2.set_xlabel('time')
ax2.set_xticks(index_13)
ax2.set_xticklabels(labels_13,rotation=20)
#ax2.set_ylim(lower_13-0.2, upper_13+0.2)

ax2.legend()

plt.savefig('./imageStats/both_bias_stats_' + scope + '.png')
plt.show()
plt.close()
#%%
#plot with single nights data
plt.scatter(data_04['hour'], data_04['med'], label = 'median', s = 2)
plt.scatter(data_04['hour'], data_04['mean'], label = 'mean', s = 2)
plt.scatter(data_04['hour'], data_04['mode'], label = 'mode', s = 2)



plt.title(scope + ' biases - ' + data_04.loc[0]['day'])
plt.ylabel('image pixel value')
plt.vlines(index_04, lower_04-0.2, upper_04+0.2, color = 'black', linewidth = 1)
plt.xlabel('time')
plt.xticks(index_04, labels_04,rotation=20)
plt.ylim(lower_04-0.2, upper_13+0.2)

plt.legend()


plt.savefig('./imageStats/20210804_bias_stats_' + scope + '.png')
plt.show()
plt.close()

#%%

plt.scatter(data_13['hour'], data_13['med'], label = 'median', s = 2)
plt.scatter(data_13['hour'], data_13['mean'], label = 'mean', s = 2)
plt.scatter(data_13['hour'], data_13['mode'], label = 'mode', s = 2)

plt.vlines(index_13, lower_13-0.2, upper_13+0.2, color = 'black', linewidth = 1)

plt.title(scope + ' biases - ' + data_13.loc[0]['day'])
plt.ylabel('image pixel value')

plt.xlabel('time')
plt.xticks(index_13, labels_13, rotation=20)
plt.ylim(lower_13-0.2, upper_13+0.2)

plt.legend()


plt.savefig('./imageStats/202108013_bias_stats_' + scope + '.png')
plt.show()
plt.close()






