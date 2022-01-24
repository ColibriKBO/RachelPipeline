#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Created on Mon Nov 22 11:05:06 2021
Update: Jan. 24, 2022, 11:20

@author: Rachel A Brown
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
import read_npy


def match_RADec(data, gdata, SR):
    '''matches list of found stars with Gaia catalog by RA/dec to get magnitudes
    input: pandas Dataframe of detected star data {x, y, RA, dec}, dataframe of Gaia data {ra, dec, magnitudes....}, search radius [deg]
    returns: original star data frame with gaia magnitude columns added where a match was found'''
    
    #from Mike Mazur's 'MagLimitChecker.py' script--------------------
    
    match_counter = 0
    SR = 0.004   #Search distance in degrees (15 arcsec)
    
    for i in range(len(data.index)):            #loop through each detected star
         RA = data.loc[i,'ra']                  #RA coord for matching
         DEC = data.loc[i,'dec']                #dec coord for matching

         df = gdata[(gdata.ra <= (RA+SR))]      #make new data frame with rows from Gaia df that are within upper RA limit
         df = df[(df.ra >= (RA-SR))]            #only include rows from this new df that are within lower RA limit
         df = df[(df.dec <= (DEC+SR))]          #only include rows from this new df that are within upper dec limit
         df = df[(df.dec >= (DEC-SR))]          #only include rows from this new df that are withing lower dec limit
         
         #RAB - matching based on smallest distance instead of brightest magnitude
       #  df['diff'] = np.sqrt(((df.ra - RA)**2*np.cos(np.radians(df.dec))) + (df.dec - DEC)**2)
       #  df.sort_values(by=['diff'], ascending=True, inplace = True)
       
         df.sort_values(by=["phot_g_mean_mag"], ascending=True, inplace=True)       #sort matches by brightness (brightest at tope)
         df.reset_index(drop=True, inplace=True)

        #if matches are found, add corresponsing magnitude columns from brightest Gaia match to original star dataframe
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
    dataframe of star data with SNR {x, y, med, std, snr}, search radius [px]
    returns: original star data frame with SNR data added where a match was found'''
    
    match_counter = 0
    
    #modified from Mike Mazur's code in 'MagLimitChecker.py'------------------
    SR = 0.1 # Search distance in pixels
    for i in range(len(mags.index)):                    #loop through each detected star
         X = mags.loc[i,'X']                            #X coord from star magnitude table
         Y = mags.loc[i,'Y']                            #Y coord from star magnitude table

         df = snr[(snr.X < (X+SR))]                     #make new data frame with rows from SNR df that are within upper X limit
         df = df[(df.X > (X-SR))]                       #only include rows from this new df that are within lower X limit
         df = df[(df.Y < (Y+SR))]                       #only include rows from this new df that are within lower Y limit
         df = df[(df.Y > (Y-SR))]                       #only include rows from this new df that are within lower Y limit
         df.sort_values(by=["SNR"], ascending=True, inplace=True)       #sort matches by SNR
         df.reset_index(drop=True, inplace=True)

         #if matches are found, add corresponsing med, std, SNR columns from match to original star magnitude dataframe
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
obs_time = datetime.time(4, 49, 6)              #time of observation (to the second)
process_date = datetime.date(2021, 11, 24)      #date of initial pipeline
image_index = '0000002'                         #index of image to use
polynom_order = '3rd'                           #order of astrometry.net plate solution polynomial
ap_r = 3                                        #radius of aperture for photometry
gain = 'low'                                    #which gain to take from rcd files ('low' or 'high')
telescope = 'Red'                               #telescope identifier
field_name = 'field1'                           #name of field observed

#paths to needed files
base_path = pathlib.Path('/', 'home', 'rbrown', 'Documents', 'Colibri')                              #path to main directory
data_path = base_path.joinpath('ColibriData', telescope + 'Data', str(obs_date).replace('-', ''))    #path that holds data

#get exact name of desired minute directory
subdirs = [f.name for f in data_path.iterdir() if f.is_dir()]                   #all minutes in night directory
minute_dir = [f for f in subdirs if str(obs_time).replace(':', '.') in f][0]    #minute we're interested in

#path to output files
save_path = base_path.joinpath('ColibriArchive', telescope, str(obs_date), minute_dir)      #path to save outputs in
lightcurve_path = save_path.joinpath(gain + '_lightcurves')                                 #path that holds light curves

#make directory to hold results in if doesn't already exist
save_path.mkdir(parents=True, exist_ok=True)


'''-------------make light curves of data----------------------'''
print('making light curve .txt files')
lightcurve_maker.getLightcurves(data_path.joinpath(minute_dir), save_path, ap_r, gain, telescope)   #save .txt files with times|fluxes

#save .png plots of lightcurves
print('saving plots of light curves')
lightcurve_looker.plot_wholecurves(lightcurve_path)


'''-----------get [x,y] to [RA, Dec] transformation------------'''
print('RA to dec transformation')

#get filepaths
transform_file = sorted(save_path.glob('*' + image_index + '_new-image_' + polynom_order + '.fits'))[0]   #output from astrometry.net
star_pos_file = sorted(save_path.glob('*' + gain + '*.npy'))[0]                                         #star position file from lightcurve_maker
star_pos_ds9 = save_path.joinpath(star_pos_file.name.replace('.npy', '_ds9.txt'))                       #star positions in ds9 format (for marking regions)
RADec_file = save_path.joinpath('XY_RD_' + gain + '_' + minute_dir + '.txt')                             #file to save XY-RD in


#get dataframe of {x, y, RA, dec}
coords = getRAdec.getRAdec(transform_file, star_pos_file, RADec_file)
coords_df = pd.DataFrame(coords, columns = ['X', 'Y', 'ra', 'dec'])         #pandas dataframe containing coordinate info

#save star coords in .txt file using ds9 format (can be marked on ds9 image using 'regions' tool)
read_npy.to_ds9(star_pos_file, star_pos_ds9)

'''---------read in Gaia coord file to get magnitudes----------'''
print('getting Gaia data')
gaia = pd.read_csv(base_path.joinpath('gaia_edr3_Field1.psv'), header = 2, sep = '|', 
    names = ['source_id', 'ra', 'dec', 'parallax', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'])
gaia.dropna(inplace = True)  


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


#for Blue: filter out stars that are in striped regions:
#final = final.drop(final[(final.X < 450) | (final.X > 1750)].index)

#save text file version of this table for later reference
final.to_csv(save_path.joinpath('starTable_' + minute_dir + '_' + gain + '_' + polynom_order + '_' + telescope + '.txt'), sep = ' ', na_rep = 'nan')

'''---------------------make plot of mag vs snr-----------------'''
print('making mag vs snr plot')
plt.scatter(final['GMAG'], final['SNR'], s = 10)

plt.title('ap r = ' + str(ap_r) + ', ' + field_name + ', ' + telescope + ', ' + str(obs_date) + ', ' + str(obs_time) + ', ' + gain)
plt.xlabel('Gaia g-band magnitude')
plt.ylabel('SNR (med/std)')


plt.savefig(save_path.joinpath('magvSNR_' + gain  + '_' + minute_dir + '.png'))
plt.show()
plt.close()

'''----------------------make plot of mag vs mean value-----------'''
print('making mag vs mean plot')
plt.scatter(final['GMAG'], -2.5*np.log(final['med']), s = 10)

plt.title('ap r = ' + str(ap_r) + ', ' + field_name + ', ' + telescope + ', ' + str(obs_date) + ', ' + str(obs_time) + ', ' + gain)
plt.xlabel('Gaia g-band magnitude')
plt.ylabel('-2.5*log(median)')

plt.savefig(save_path.joinpath('magvmed_' + gain + '_' + minute_dir + '.png'))
plt.show()
plt.close()

