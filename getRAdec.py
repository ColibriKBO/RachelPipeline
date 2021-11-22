# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:51:30 2021

@author: rache
"""

import astropy
from astropy.io import fits
import numpy as np
from astropy import wcs

#%%Load in transformation information
transform_im = fits.open('ColibriArchive/2021-07-08/astrometry/23field1_new-image.fits')
transform = wcs.WCS(transform_im[0].header)
#%%load in coords to be transformed

star_pos = np.load('ColibriArchive/2021-07-08/field1_pos_23.npy')
xy_coords = star_pos[:,:-1]

#%%

world = transform.wcs_pix2world(xy_coords, 0)
print(world)
px = transform.wcs_world2pix(world, 0)
print(px)

#%%save x,y coords & RA Dec as .txt file
savefile = './ColibriArchive/2021-07-08/xy_rd_23.txt'
with open(savefile, 'w') as filehandle:
    
    for i in range(0, len(star_pos)):
      #  print(xy_coords[i][0], xy_coords[i][1], world[i][0], world[i][1])
        #print(xy_coords[i], world[i])
        filehandle.write('%f %f %f %f\n' %(xy_coords[i][0], xy_coords[i][1], world[i][0], world[i][1]))
