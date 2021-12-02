# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:51:30 2021

@author: rache
"""

import astropy
from astropy.io import fits
import numpy as np
import pathlib
import datetime
from astropy import wcs

def getRAdec(transform_file, star_pos_file, savefile):
    '''get WCS transform from astrometry.net header
    input: astrometry.net output file, star position file (.npy)
    returns: coordinate transform'''
    
    #load in transformation information
    transform_im = fits.open(transform_file)
    transform = wcs.WCS(transform_im[0].header)
    
    #get star coordinates from observation image (.npy file)
    star_pos = np.load(star_pos_file)
    
    #get transformation
    world = transform.wcs_pix2world(star_pos, 0)
   # print(world)
    px = transform.wcs_world2pix(world, 0)
   # print(px)
    
    #optional: save text file with transformation
    with open(savefile, 'w') as filehandle:
        filehandle.write('#Date: ' + '\n')
        filehandle.write('#Time: ' + '\n')
        filehandle.write('#Image name: ' + '\n')
        filehandle.write('#\n#\n#\n#\n#X  Y  RA  Dec\n')
    
        for i in range(0, len(star_pos)):
            #output table: | x | y | RA | Dec | 
            filehandle.write('%f %f %f %f\n' %(star_pos[i][0], star_pos[i][1], world[i][0], world[i][1]))
      
    coords = np.array([star_pos[:,0], star_pos[:,1], world[:,0], world[:,1]]).transpose()
    return coords
    
    
'''call function'''
#telescope = 'Red'
#obs_date = datetime.date(2021, 8, 4)

#dir_path = pathlib.Path('../ColibriArchive/' + telescope +'/' + str(obs_date) + '/20210804_04.50.10.938/') 
#transform_file = pathlib.Path.joinpath(dir_path, '20210804_04.50_0000001_new-image.fits')
##star_pos = pathlib.Path.joinpath(dir_path, 'RedData_04.50.10.938_2sig_pos.npy')
#savefile = pathlib.Path.joinpath(dir_path, 'xy_rd_04.50.txt')

#coords = getRAdec(transform_file, star_pos, savefile)

#print(coords)


