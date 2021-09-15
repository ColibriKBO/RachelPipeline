#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 11:53:28 2021
Read in all images from a single night, get the median value, save a text file with 
filename|time|median|bias-subtracted median|mean|bias-subtracted mean|mode|bias-subtracted mode

@author: rbrown
"""
import numpy as np
import numba as nb
import scipy.stats 
import binascii, os, sys, glob
from astropy.io import fits

# Function for reading specified number of bytes
def readxbytes(fid, numbytes):
    for i in range(1):
        data = fid.read(numbytes)
        if not data:
            break
    return data

# Function to read 12-bit data with Numba to speed things up
@nb.njit(nb.uint16[::1](nb.uint8[::1]),fastmath=True,parallel=True)
def nb_read_data(data_chunk):
    """data_chunk is a contigous 1D array of uint8 data)
    eg.data_chunk = np.frombuffer(data_chunk, dtype=np.uint8)"""
    #ensure that the data_chunk has the right length

    assert np.mod(data_chunk.shape[0],3)==0

    out=np.empty(data_chunk.shape[0]//3*2,dtype=np.uint16)
    image1 = np.empty((2048,2048),dtype=np.uint16)
    image2 = np.empty((2048,2048),dtype=np.uint16)

    for i in nb.prange(data_chunk.shape[0]//3):
        fst_uint8=np.uint16(data_chunk[i*3])
        mid_uint8=np.uint16(data_chunk[i*3+1])
        lst_uint8=np.uint16(data_chunk[i*3+2])

        out[i*2] =   (fst_uint8 << 4) + (mid_uint8 >> 4)
        out[i*2+1] = ((mid_uint8 % 16) << 8) + lst_uint8

    return out

def getSizeRCD(filenames):
    """ MJM - Get the size of the images and number of frames """
    filename_first = filenames[0]
    frames = len(filenames)

    width = 2048
    height = 2048

    # You could also get this from the RCD header by uncommenting the following code
    # with open(filename_first, 'rb') as fid:
    #     fid.seek(81,0)
    #     hpixels = readxbytes(fid, 2) # Number of horizontal pixels
    #     fid.seek(83,0)
    #     vpixels = readxbytes(fid, 2) # Number of vertical pixels

    #     fid.seek(100,0)
    #     binning = readxbytes(fid, 1)

    #     bins = int(binascii.hexlify(binning),16)
    #     hpix = int(binascii.hexlify(hpixels),16)
    #     vpix = int(binascii.hexlify(vpixels),16)
    #     width = int(hpix / bins)
    #     height = int(vpix / bins)

    return width, height, frames

# Function to split high and low gain images
def split_images(data,pix_h,pix_v,gain):
    interimg = np.reshape(data, [2*pix_v,pix_h])

    if gain == 'low':
        image = interimg[::2]
    else:
        image = interimg[1::2]

    return image

# Function to read RCD file data
def readRCD(filename):

    hdict = {}

    with open(filename, 'rb') as fid:

        # Go to start of file
        fid.seek(0,0)

        # Serial number of camera
        fid.seek(63,0)
        hdict['serialnum'] = readxbytes(fid, 9)

        # Timestamp
        fid.seek(152,0)
        hdict['timestamp'] = readxbytes(fid, 29).decode('utf-8')

        # Load data portion of file
        fid.seek(384,0)

        table = np.fromfile(fid, dtype=np.uint8, count=12582912)

    return table, hdict

def importFramesRCD(parentdir, filenames, start_frame, num_frames, bias):
    """ reads in frames from .rcd files starting at frame_num
    input: parent directory (minute), list of filenames to read in, starting frame number, how many frames to read in, 
    bias image (2D array of fluxes)
    returns: array of image data arrays, array of header times of these images"""
    
    imagesData = []    #array to hold image data
    images_subData = [] #array to hold bias subtracted image data
    imagesTimes = []   #array to hold image times
    
    hnumpix = 2048
    vnumpix = 2048
    imgain = 'low'
    
    '''list of filenames to read between starting and ending points'''
    files_to_read = [filename for i, filename in enumerate(filenames) if i >= start_frame and i < start_frame + num_frames]

    for filename in files_to_read:


        data, header = readRCD(filename)
        headerTime = header['timestamp']

        images = nb_read_data(data)
        image = split_images(images, hnumpix, vnumpix, imgain)
        

        #change time if time is wrong (29 hours)
        hour = str(headerTime).split('T')[1].split(':')[0]
        fileMinute = str(headerTime).split(':')[1]
        dirMinute = parentdir.split('_')[1].split('.')[1]
        
        #check if hour is bad, if so take hour from directory name and change header
        if int(hour) > 23:
            
            #directory name has local hour, header has UTC hour, need to convert (+4)
            newLocalHour = int(parentdir.split('_')[1].split('.')[0])
        
            if int(fileMinute) < int(dirMinute):
                newUTCHour = newLocalHour + 4 + 1     #add 1 if hour changed over during minute
            else:
                newUTCHour = newLocalHour + 4
        
            #replace bad hour in timestamp string with correct hour
            newUTCHour = str(newUTCHour)
            newUTCHour = newUTCHour.zfill(2)
        
            replaced = str(headerTime).replace('T' + hour, 'T' + newUTCHour).strip('b').strip(' \' ')
        
            #encode into bytes
            headerTime = replaced

        image_sub = np.subtract(image,bias)      #subtract bias
        imagesData.append(image)
        images_subData.append(image_sub)
        imagesTimes.append(headerTime)

    '''make into array'''
    imagesData = np.array(imagesData, dtype='float64')
    images_subData = np.array(images_subData, dtype='float64')
    
    '''reshape, make data type into floats'''
    if imagesData.shape[0] == 1:
        imagesData = imagesData[0]
        imagesData = imagesData.astype('float64')
        
    if images_subData.shape[0] == 1:
        images_subData = images_subData[0]
        images_subData = images_subData.astype('float64')
        
    return imagesData, images_subData, imagesTimes

def getBias(filepath, numOfBiases):
    """ get median bias image from a set of biases (length =  numOfBiases) from filepath
    input: filepath to bias image directory, number of bias  images to take median from
    return: median bias image"""
    
    print('Calculating median bias...')
    
    # Added a check to see if the fits conversion has been done.
    # Comment out if you only want to check for presence of fits files.
    # If commented out, be sure to uncomment the 'if not glob(...)' below
    if os.path.isfile(filepath[0] + 'converted.txt') == False:
        with open(filepath[0] + 'converted.txt', 'a'):
            os.utime(filepath[0] + 'converted.txt')
            os.system("python .\\RCDtoFTS.py "+filepath[0])
    else:
        print('Already converted raw files to fits format.')
        print('Remove file converted.txt if you want to overwrite.')

    # if not glob(filepath[0]+'/*.fits'):
    #     print('converting biases to .fits')
    #     os.system("python ..\\..\\RCDtoFTS2.py "+filepath[0])
    
    '''get list of bias images to combine'''
    biasFileList = glob.glob(filepath[0] + '*.fits')
    biases = []   #list to hold bias data
    
    '''append data from each bias image to list of biases'''
    for i in range(0, numOfBiases):
        biases.append(fits.getdata(biasFileList[i]))
    
    '''take median of bias images'''
    biasMed = np.median(biases, axis=0)
    
    return biasMed
    

#get night directory
if __name__ == '__main__':
   # if len(sys.argv) > 1:
   #     nightdir = sys.argv[1]   #night directory
    
    nightdir = './ColibriData/202106023/'
    savefile = './'+nightdir.split('/')[-2] + '_stats.txt'
    
    with open(savefile, 'a') as filehandle:
                filehandle.write('filename    time    med    bias_sub_med    mean    bias_sub_mean    mode    bias_sub_mode\n')          
    
    
        
    minutedirs = glob.glob(nightdir+'*/')  #list of minute directories
    minutedirs.sort()
    
    minutedirs = [f for f in minutedirs if 'Bias' not in f]  #don't run pipeline on bias images
         
    '''get median bias image to subtract from all frames'''
    biasFilepath = glob.glob(nightdir + '/Bias/'+ '*/')
    NumBiasImages = 9                             #number of bias images to combine in median bias image
    bias = getBias(biasFilepath, NumBiasImages)    #take median of NumBiasImages to use as bias
    print('number of subdirectories: ', len(minutedirs))
    
    #loop through each minute directory in night directory
    for minute in minutedirs:
        images = glob.glob(minute + '*.rcd')      #list of images
        images.sort()
        print('working on: ',minute)
        
#        images, Times = importFramesRCD(minute, images)#, 0, len(images) - 1)
        
        #loop through each image in minute directory
        for image in images:
#            filenames.append(image)
            data, datasub, time = importFramesRCD(minute, [image], 0, 1, bias)
            med = np.median(data)     #median value for image
            mean = np.mean(data)
            mode = scipy.stats.mode(data, axis = None)[0][0]
            
            med_sub = np.median(datasub)   #median value for bias subtracted image
            mean_sub = np.mean(datasub)
            mode_sub = scipy.stats.mode(datasub, axis = None)[0][0]
            
            #append image info to file
            with open(savefile, 'a') as filehandle:
                filehandle.write('%s %s %f %f %f %f %f %f\n' %(image, time[0], med, med_sub, mean, mean_sub, mode, mode_sub))
            

            
    #write lists to .txt file
    
            
    