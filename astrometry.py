#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 14:14:16 2021
look at the output files from astrometry.net
@author: rbrown
"""
#%%imports
import pathlib
from astropy.io import fits
import astropy.stats
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import getRAdec


#%%


def get_results(filename_base):
    '''read in files form astrometry.net output and put data into variables, also read in xy-radec file
    input: base name of the file that was analyzed
    returns: headers and data for astrometry.net files'''
    

    axy = fits.open(save_path.joinpath(filename_base + '_axy_' + order + '.fits'))
    corr = fits.open(save_path.joinpath(filename_base + '_corr_' + order + '.fits'))
    new_image = fits.open(save_path.joinpath(filename_base + '_new-image_' + order + '.fits'))
    wcs = fits.open(save_path.joinpath(filename_base + '_wcs_' + order + '.fits'))
    rdls = fits.open(save_path.joinpath(filename_base + '_rdls_' + order + '.fits'))

    #axy.info()
    #corr.info()
    #new_image.info()
    #corr.info()
    #rdls.info()

    #WCS solution
    wcs_header = wcs[0].header
    
    #new image (with RA, dec info)
    newimage_header = new_image[0].header
    newimage_data = new_image[0].data

    #X, Y of extracted sources in image
    axy_header = axy[1].header
    axy_data = axy[1].data

    #RA, dec of extracted sources in image
    rdls_header = rdls[1].header
    rdls_data = rdls[1].data

    corr_header = corr[1].header
    corr_data = corr[1].data
    

    return wcs_header, newimage_header, newimage_data, axy_header, axy_data, rdls_header, rdls_data, corr_header, corr_data


def match_XY(ast, colibri, SR):
    '''matches two data frames based on X, Y 
    input: pandas dataframe of star data with Gaia magnitudes {x, y, RA, dec, 3 magnitudes} 
    dataframe of star data with SNR {x, y, med, std, snr}
    returns: original star data frame with SNR data added where a match was found'''
    
    match_counter = 0
    
    #modified from Mike Mazur's code in 'MagLimitChecker.py'------------------
    SR = 2 # Search distance in degrees
    for i in range(ast.shape[0]):
         X = ast.loc[i,'field_x']
         Y = ast.loc[i,'field_y']

         df = colibri[(colibri['#X'] < (X+SR))]
         df = df[(df['#X'] > (X-SR))]
         df = df[(df['Y'] < (Y+SR))]
         df = df[(df['Y'] > (Y-SR))]
         df['diff_x'] = df['#X'] - X
         df['diff_y'] = df['Y'] - Y
         df.sort_values(by=["diff_x"], ascending=True, inplace=True)
         df.reset_index(drop=True, inplace=True)

         if len(df.index)>=1:
             ast.loc[i,'colibri_X'] = df.loc[0]['#X']
             ast.loc[i,'colibri_Y'] = df.loc[0]['Y']
             ast.loc[i,'colibri_RA'] = df.loc[0]['RA']
             ast.loc[i, 'colibri_dec'] = df.loc[0]['Dec']

       #end of Mike's section -------------------------------------------
             match_counter +=1 
    
    ast['ra_diff'] = ast['colibri_RA'] - ast['index_ra']
    ast['dec_diff'] = ast['colibri_dec'] - ast['index_dec']
    ast['x_diff'] = ast['colibri_X'] - ast['index_x']
    ast['y_diff'] = ast['colibri_Y'] - ast['index_y']
             
    print('Number of SNR matches: ', match_counter)
    return ast
    

def XY_diffPlot(matched):
    '''makes plot of X difference vs Y differences, calculates mean and stddev
    input: dataframe containing columns 'x_diff' and 'y_diff'
    returns: displays and saves plot of xdifferences and y differences, prints out mean and stddev'''
    
   # x_mean = np.mean(matched['x_diff'])
    clip_sigma = 2
    clipped_xdiff = astropy.stats.sigma_clip(matched['x_diff'], sigma = clip_sigma, cenfunc = 'mean')
    clipped_ydiff = astropy.stats.sigma_clip(matched['y_diff'], sigma = clip_sigma, cenfunc = 'mean')
    x_mean = np.mean(clipped_xdiff)
    x_std = np.std(clipped_xdiff)
    y_mean = np.mean(clipped_ydiff)
    y_std = np.std(clipped_ydiff)
    
    #print('%f sigma clipped X mean: %.3f' %(clip_sigma x_mean)
    #print('2 sigma clipped X stddev: ', x_std)
    #print('Y mean:  ', y_mean)
    #print('Y stddev: ', y_std)
    
    plt.figure(figsize=(10, 10), dpi=100)
    
    plt.scatter(matched['x_diff'], matched['y_diff'], s = 8)
    plt.scatter(0, 0, color = 'black', marker = '+', s = 50)
    
    plt.text(-6, 6, 'X mean: %.3f +/- %.3f px' %(x_mean, x_std), fontsize = 18)
    plt.text(-6, 5, 'Y mean: %.3f +/- %.3f px' %(y_mean, y_std), fontsize = 18)
    plt.text(-6, 4, '(%.0f$\sigma$ clipped)' %(clip_sigma), fontsize = 18)

    plt.title(order + ' order, ' + telescope + ', ' + field_name + ', ' + str(obs_date) + ', ' + str(obs_time), fontsize = 20)

    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    
    plt.xlabel('(Colibri X - Index X) [px]', fontsize = 18)
    plt.ylabel('(Colibri Y - Index Y) [px]', fontsize = 18)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(save_path.joinpath('X-Y_diff_' + order + '_' + telescope + '_' + field_name + '_' + str(obs_date) + '_' + str(obs_time) + '.png'))
    plt.show()
    plt.close()

def RAdec_diffPlot(matched):
    '''makes plot of X difference vs Y differences, calculates mean and stddev
    input: dataframe containing columns 'x_diff' and 'y_diff'
    returns: displays and saves plot of xdifferences and y differences, prints out mean and stddev'''
    
    clip_sigma = 2
    clipped_radiff = astropy.stats.sigma_clip(matched['ra_diff'], sigma = clip_sigma, cenfunc = 'mean')
    clipped_decdiff = astropy.stats.sigma_clip(matched['dec_diff'], sigma = clip_sigma, cenfunc = 'mean')
    
    ra_mean = np.mean(clipped_radiff)*3600
    ra_std = np.std(clipped_radiff)*3600
    dec_mean = np.mean(clipped_decdiff)*3600
    dec_std = np.std(clipped_decdiff)*3600
    
    #print('RA mean: ', ra_mean)
    #print('RA stddev: ', ra_std)
    #print('dec mean:  ', dec_mean)
    #print('dec stddev: ', dec_std)
    
    plt.figure(figsize=(10, 10), dpi=100)
    
    plt.scatter(matched['ra_diff'].multiply(3600), matched['dec_diff'].multiply(3600), s = 8)
    plt.scatter(0, 0, color = 'black', marker = '+', s = 50)
    
    plt.text(-12, 12, 'RA mean: %.3f +/- %.3f px' %(ra_mean, ra_std), fontsize = 18)
    plt.text(-12, 10, 'Dec mean: %.3f +/- %.3f px' %(dec_mean, dec_std), fontsize = 18)
    plt.text(-12, 8, '(%.0f$\sigma$ clipped)' %(clip_sigma), fontsize = 18)

    plt.xlim(-0.004*3600, 0.004*3600)
    plt.ylim(-0.004*3600, 0.004*3600)

    plt.title(order + ' order, ' + telescope + ', ' + field_name + ', ' + str(obs_date) + ', ' + str(obs_time), fontsize = 20)
    plt.xlabel('(Colibri RA - Index RA) [arcsec]', fontsize = 18)
    plt.ylabel('(Colibri Dec - Index Dec) [arcsec]', fontsize = 18)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(save_path.joinpath('RA-dec_diff_' + order + '_' + telescope + '_' + field_name + '_' + str(obs_date) + '_' + str(obs_time) + '.png'))
    plt.show()
    plt.close()


def XY_arrowPlot(matched):
    '''makes plot of X, Y differences as arrows on XY plane
    input: dataframe containing columns of X, Y corrds, X, Y differences
    returns: displays and saves arrow plot'''
    
    arrow_scale = 0.04
    plt.figure(figsize=(10, 10), dpi=100)
    
    plt.quiver(matched['colibri_X'], matched['colibri_Y'], matched['x_diff'], matched['y_diff'], 
               units = 'xy', scale = arrow_scale, scale_units = 'xy')

    plt.xlim(-100, 2100)
    plt.ylim(-100, 2100)
    
    plt.text(-50, 2000, 'Arrow scale: %.2f ' %(arrow_scale), fontsize = 18)

    plt.title(order + ' order, ' + telescope + ', ' + field_name + ', ' + str(obs_date) + ', ' + str(obs_time), fontsize = 20)

    plt.xlabel('Colibri X [px]', fontsize = 18)
    plt.ylabel('Colibri Y [px]', fontsize = 18)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(save_path.joinpath('X-Y_diff_arrows_' + order + '_' + telescope + '_' + field_name + '_' + str(obs_date) + '_' + str(obs_time) + '.png'))
    plt.show()
    plt.close()


def RAdec_arrowPlot(matched):
    '''makes plot of X, Y differences as arrows on XY plane
    input: dataframe containing columns of X, Y corrds, X, Y differences
    returns: displays and saves arrow plot'''
    
    arrow_scale = 0.04
    plt.figure(figsize=(10, 10), dpi=100)
        
    plt.quiver(matched['colibri_RA'], matched['colibri_dec'], matched['ra_diff'], matched['dec_diff'], 
               units = 'xy', scale = arrow_scale, scale_units = 'xy')
    
    plt.text(273.08, -19.25, 'Arrow scale: %.2f ' %(arrow_scale), fontsize = 18)

    plt.xlim(273.05, 274.115)
    plt.ylim(-18.221, -19.286)

    plt.title(order + ' order, ' + telescope + ', ' + field_name + ', ' + str(obs_date) + ', ' + str(obs_time), fontsize = 20)
    plt.xlabel('Colibri RA [degrees]', fontsize = 18)
    plt.ylabel('Colibri Dec [degrees]', fontsize = 18)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(save_path.joinpath('RA-dec_diff_arrows_' + order + '_' + telescope + '_' + field_name + '_' + str(obs_date) + '_' + str(obs_time) + '.png'))
    plt.show()
    plt.close()
    
def plot_image(image_data):
    '''make plot of colibri image data in grayscale with colorbar
    input: filename
    returns: saves plot to disk'''
        
    arrow_scale = 0.04
        
    plt.figure(figsize = (10, 10), dpi = 100)
    plt.imshow(image_data, vmin = 0, vmax = 20)
        
    plt.gca().invert_yaxis()
    plt.colorbar().set_label('Pixel Value', fontsize = 18)
        
    plt.quiver(matched['colibri_X'], matched['colibri_Y'], matched['x_diff'], matched['y_diff'], 
                   units = 'xy', scale = arrow_scale, scale_units = 'xy')
        
    plt.text(15, 20, 'Arrow scale: %.2f ' %(arrow_scale), fontsize = 18)

    plt.title(order + ' order, ' + telescope + ', ' + field_name + ', ' + str(obs_date) + ', ' + str(obs_time), fontsize = 20)

    plt.xlabel('Colibri X [px]', fontsize = 18)
    plt.ylabel('Colibri Y [px]', fontsize = 18)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')
        
    plt.savefig(save_path.joinpath('XY_arrows-image_' + order + '_' + telescope + '_' + field_name + '_' + str(obs_date) + '_' + str(obs_time) + '.png'))
    plt.show()
    plt.close()    


#%%Execute script

obs_date = datetime.date(2021, 8, 4)            #date of observation
obs_time = datetime.time(4, 50, 10)             #time of observation (to the second)
process_date = datetime.date(2021, 11, 24)      #date of initial pipeline
image_index = '0000002'                         #index of image to use
order = '2nd'

telescope = 'Red'                               #telescope identifier
field_name = 'field1'                           #name of field observed

#paths to needed files
base_path = pathlib.Path('/', 'home', 'rbrown', 'Documents', 'Colibri')
data_path = base_path.joinpath('ColibriData', 'RedData', str(obs_date).replace('-', ''))    #path that holds data

#get exact name of desired minute directory
subdirs = [f.name for f in data_path.iterdir() if f.is_dir()]                   #all minutes in night directory
minute_dir = [f for f in subdirs if str(obs_time).replace(':', '.') in f][0]    #minute we're interested in

save_path = base_path.joinpath('ColibriArchive', telescope, str(obs_date), minute_dir)    #path to save outputs in
filename_base = '20210804_04.50_0000002'

#make star transform file
coords = getRAdec.getRAdec(save_path.joinpath(filename_base + '_new-image_' + order + '.fits'), 
                           sorted(save_path.glob('*.npy'))[0], 
                           save_path.joinpath(minute_dir + '_' + order + '_xy_RD.txt'))

#read in this file, get astrometry.net results
colibri_stars = pd.read_csv(save_path.joinpath(minute_dir + '_' + order + '_xy_RD.txt'), delim_whitespace = True, header = 7)
wcs_header, newimage_header, newimage_data, axy_header, axy_data, rdls_header, rdls_data, corr_header, corr_data = get_results(filename_base)

#make dataframe of correlation stars
corr_df = pd.DataFrame(np.array(corr_data).byteswap().newbyteorder())

#match astrometry stars with colibri stars
matched = match_XY(corr_df, colibri_stars, 0.1)

#make plots
XY_diffPlot(matched)
RAdec_diffPlot(matched)
XY_arrowPlot(matched)
RAdec_arrowPlot(matched)
plot_image(newimage_data)

#%%translate .npy file into .txt for plotting in ds9
def npy2text():
    col_savefile = save_path.joinpath('colibri_stars.txt')
    with open(col_savefile, 'w') as filehandle:
    
        for index, line in colibri_stars.iterrows():
            filehandle.write('%s %f %f %f\n' %('circle', line[0], line[1], 6.))
            #filehandle.write('%f %f\n' %(line[0], line[1]))
        
        
        corr_savefile = save_path.joinpath('corr_stars.txt')
        with open(corr_savefile, 'w') as filehandle:
    
            for i in range(len(corr_data)):
                filehandle.write('%s %f %f %f\n' %('circle', corr_data['index_x'][i], corr_data['index_y'][i], 6.))
                #filehandle.write('%f %f\n' %(line[0], line[1]))
        
        axy_savefile = save_path.joinpath('axy_stars.txt')
        with open(axy_savefile, 'w') as filehandle:
    
            for i in range(len(axy_data)):
                filehandle.write('%s %f %f %f\n' %('circle', axy_data['X'][i], axy_data['Y'][i], 6.))
                #filehandle.write('%f %f\n' %(line[0], line[1]))
        
        
        
        
        
        
        
        
        
        
        
        
        