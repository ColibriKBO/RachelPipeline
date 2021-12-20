#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Created on Mon Nov 22 11:05:06 2021

@author: rbrown
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime 
import astropy
from astropy.io import fits
from astropy import wcs
import getRAdec
import pathlib
import snplots
import lightcurve_maker
import lightcurve_looker


def match_RADec(data, gdata, SR):
    '''matches list of found stars with Gaia catalog by RA/dec to get magnitudes
    input: pandas Dataframe of our star data {x, y, RA, dec}, dataframe of Gaia data {ra, dec, magnitudes....}
    returns: original star data frame with gaia magnitude columns added where a match was found'''
    
    #from Mike Mazur's 'MagLimitChecker.py' script--------------------
    
    match_counter = 0
    #TODO: come back to this, check astrometry.net solution
    #SR = 0.015 # Search distance in degrees
    SR = 0.004   #15 arcsec
    for i in range(len(data.index)):
         RA = data.loc[i,'ra']
         DEC = data.loc[i,'dec']

         df = gdata[(gdata.ra <= (RA+SR))]
         df = df[(df.ra >= (RA-SR))]
         df = df[(df.dec <= (DEC+SR))]
         df = df[(df.dec >= (DEC-SR))]
         #TODO: should take closest match not brightest
         df['diff'] = np.sqrt((df.ra - RA)**2 + (df.dec - DEC)**2)
        # df.sort_values(by=["phot_g_mean_mag"], ascending=True, inplace=True)
         df.sort_values(by=['diff'], ascending=True, inplace=True)
         df.reset_index(drop=True, inplace=True)

         if len(df.index)>=1:   #RAB - added >= sign
             
             data.loc[i,'BMAG'] = df.loc[0]['phot_bp_mean_mag']
             data.loc[i,'GMAG'] = df.loc[0]['phot_g_mean_mag']
             data.loc[i,'RMAG'] = df.loc[0]['phot_rp_mean_mag']
             
       #end of Mike's section -------------------------------------------
             match_counter +=1 
    
    print('Number of Gaia matches: ', match_counter)
    return data
             

def match_XY(mags, snr, SR):
    '''matches two data frames based on X, Y 
    input: pandas dataframe of star data with Gaia magnitudes {x, y, RA, dec, 3 magnitudes} 
    dataframe of star data with SNR {x, y, med, std, snr}
    returns: original star data frame with SNR data added where a match was found'''
    
    match_counter = 0
    
    #modified from Mike Mazur's code in 'MagLimitChecker.py'------------------
    SR = 0.1 # Search distance in degrees
    for i in range(len(mags.index)):
         X = mags.loc[i,'X']
         Y = mags.loc[i,'Y']

         df = snr[(snr.X < (X+SR))]
         df = df[(df.X > (X-SR))]
         df = df[(df.Y < (Y+SR))]
         df = df[(df.Y > (Y-SR))]
         df.sort_values(by=["SNR"], ascending=True, inplace=True)
         df.reset_index(drop=True, inplace=True)

         if len(df.index)>=1:
             mags.loc[i,'med'] = df.loc[0]['med']
             mags.loc[i,'std'] = df.loc[0]['std']
             mags.loc[i,'SNR'] = df.loc[0]['SNR']
             
       #end of Mike's section -------------------------------------------
             match_counter +=1 
    
    print('Number of SNR matches: ', match_counter)
    return mags
    
'''---------------------------------SCRIPT STARTS HERE--------------------------------------------'''

'''------------set up--------------------'''
print('setting up')
#time and date of observations/processing
obs_date = datetime.date(2021, 8, 4)            #date of observation
obs_time = datetime.time(4, 50, 10)             #time of observation (to the second)
process_date = datetime.date(2021, 11, 24)      #date of initial pipeline
image_index = '0000002'                         #index of image to use
polynom_order = '3rd'                               #order of astrometry.net plate solution polynomial
ap_r = 3                                        #radius of aperture for photometry

telescope = 'Red'                               #telescope identifier
field_name = 'field1'                           #name of field observed

#paths to needed files
base_path = pathlib.Path('/', 'home', 'rbrown', 'Documents', 'Colibri')
data_path = base_path.joinpath('ColibriData', 'RedData', str(obs_date).replace('-', ''))    #path that holds data

#get exact name of desired minute directory
subdirs = [f.name for f in data_path.iterdir() if f.is_dir()]                   #all minutes in night directory
minute_dir = [f for f in subdirs if str(obs_time).replace(':', '.') in f][0]    #minute we're interested in

#path to output files
save_path = base_path.joinpath('ColibriArchive', telescope, str(obs_date), minute_dir)    #path to save outputs in
lightcurve_path = save_path.joinpath('lightcurves')                          #path that holds light curves

#make directory to hold results in if doesn't already exist
save_path.mkdir(parents=True, exist_ok=True)


'''-------------make light curves of data----------------------'''
print('making light curve .txt files')
lightcurve_maker.getLightcurves(data_path.joinpath(minute_dir), save_path, ap_r)   #save .txt files with times|fluxes
#save .png plots of lightcurves
#print('saving plots of light curves')
#lightcurve_looker.plot_wholecurves(lightcurve_path)


'''-----------get [x,y] to [RA, Dec] transformation------------'''
print('RA to dec transformation')

#get filepaths
transform_file = sorted(save_path.glob('*' + image_index + '_new-image_' + polynom_order + '.fits'))[0]   #output from astrometry.net
#transform_file = save_path.joinpath('20210804_04.50_0000002_new-image.fits')    #output from astrometry.net
star_pos_file = sorted(save_path.glob('*.npy'))[0]                              #star position file from lightcurve_maker
RADec_file = save_path.joinpath(minute_dir + '_xy_RD.txt')                      #file to save XY-RD in


#get dataframe of {x, y, RA, dec}
coords = getRAdec.getRAdec(transform_file, star_pos_file, RADec_file)
coords_df = pd.DataFrame(coords, columns = ['X', 'Y', 'ra', 'dec'])         #pandas dataframe containing coordinate info


'''---------read in Gaia coord file to get magnitudes----------'''
print('getting Gaia data')
gaia = pd.read_csv(base_path.joinpath('gaia_edr3_Field1.psv'), header = 2, sep = '|', 
    names = ['source_id', 'ra', 'dec', 'parallax', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'])
#gaia.dropna(inplace = True)  #TODO: check more carefully (are mags nan?)


'''----------------------get star SNRs------------------------'''
print('calculating SNRS')
#dataframe of star info {x, y, median, stddev, median/stddev}
stars = pd.DataFrame(snplots.snr_single(lightcurve_path), columns = ['X', 'Y', 'med', 'std', 'SNR'])


'''----------------matching tables----------------------------'''
print('matching tables')
# 1: match (RA, dec) from light curves with (RA, dec) from Gaia to get magnitudes
rd_mag = match_RADec(coords_df, gaia, 0.015)

# 2: match (X,Y) from light curves with (X,Y) from position file to get (RA, dec)
final = match_XY(rd_mag, stars, 0.015)


'''---------------------make plot of mag vs snr-----------------'''
print('making mag vs snr plot')
plt.scatter(final['GMAG'], final['SNR'], s = 10)

plt.title('ap r = ' + str(ap_r) + ', ' + field_name + ', ' + telescope + ', ' + str(obs_date) + ', ' + str(obs_time))
plt.xlabel('Gaia g-band magnitude')
plt.ylabel('SNR (med/std)')


plt.savefig(save_path.joinpath(minute_dir + '_magvSNR.png'))
plt.show()
plt.close()

'''----------------------make plot of mag vs mean value-----------'''
print('making mag vs mean plot')
plt.scatter(final['GMAG'], -2.5*np.log(final['med']), s = 10)

plt.title('ap r = ' + str(ap_r) + ', ' + field_name + ', ' + telescope + ', ' + str(obs_date) + ', ' + str(obs_time))
plt.xlabel('Gaia g-band magnitude')
plt.ylabel('-2.5*log(median)')

plt.savefig(save_path.joinpath(minute_dir + '_magvmed.png'))
plt.show()
plt.close()

#%%

'''---------get list of stars that are outliers in above plot-----------'''
#outliers = final.loc(final['GMAG'] < 9)
outliers = final[final['GMAG'] < 8.5]
outliers = outliers[-2.5*np.log(outliers['med']) > -13]
outliers_sorted = outliers.sort_values(by=['GMAG'])
#outliers.loc(-2.5*np.log(outliers['med']) > -14, inplace = True)

plt.scatter(outliers['GMAG'], -2.5*np.log(outliers['med']), s = 10)

plt.title('ap r = ' + str(ap_r) + ', ' + field_name + ', ' + telescope + ', ' + str(obs_date) + ', ' + str(obs_time))
plt.xlabel('Gaia g-band magnitude')
plt.ylabel('-2.5*log(median)')

#plt.savefig(save_path.joinpath(minute_dir + '_magvmed.png'))
plt.show()
plt.close()


#%% examine final dataframe
final.sort_values(by=["GMAG"], ascending=True, inplace=True)
gaia.sort_values(by=["ra"], ascending=True, inplace=True)



