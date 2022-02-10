"""
Created 2018 by Emily Pass

Update: Feb. 10, 2022 - Rachel Brown

-initial Colibri data processing pipeline for flagging candidate
KBO occultation events
"""

import sep
import numpy as np
import numba as nb
from glob import glob
from astropy.io import fits
from astropy.convolution import convolve_fft, RickerWavelet1DKernel
from astropy.time import Time
from joblib import delayed, Parallel
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path
import multiprocessing
import datetime
import matplotlib.pyplot as plt
import os
import gc
import time as timer
import binascii, array
#from datetime import datetime, date, time, timezone


def initialFindFITS(data):
#TODO: set threshold as variable
    """ Locates the stars in the initial time slice 
    input: flux data in 2D array for a fits image
    returns: [x, y, half light radius] of all stars in pixels"""

    ''' Background extraction for initial time slice'''
    data_new = deepcopy(data)
    bkg = sep.Background(data_new)
    bkg.subfrom(data_new)
    thresh = 2. * bkg.globalrms  # set detection threshold to mean + 3 sigma

    
    ''' Identify stars in initial time slice '''
    objects = sep.extract(data_new, thresh)


    ''' Characterize light profile of each star '''
    halfLightRad = np.sqrt(objects['npix'] / np.pi) / 2.  # approximate half light radius as half of radius

    
    ''' Generate tuple of (x,y,r) positions for each star'''
    positions = zip(objects['x'], objects['y'], halfLightRad)
    

    return positions


def refineCentroid(data, time, coords, sigma):
    """ Refines the centroid for each star for an image based on previous coords 
    input: flux data in 2D array for single fits image, header time of image, 
    coord of stars in previous image, weighting (Gauss sigma)
    returns: new [x, y] positions, header time of image """

    '''initial x, y positions'''
    x_initial = [pos[0] for pos in coords]
    y_initial = [pos[1] for pos in coords]
    
    '''use an iterative 'windowed' method from sep to get new position'''
    new_pos = np.array(sep.winpos(data, x_initial, y_initial, sigma, subpix=5))[0:2, :]
    x = new_pos[:][0].tolist()
    y = new_pos[:][1].tolist()
    
    '''returns tuple x, y (python 3: zip(x, y) -> tuple(zip(x,y))) and time'''
    return tuple(zip(x, y)), time

def averageDrift(positions, times):
    """ Determines the median x/y drift rates of all stars in a minute (first to last image)
    input: array of [x,y] star positions, times of each position
    returns: median x, y drift rate [px/s] taken over all stars"""
    
    times = Time(times, precision=9).unix     #convert position times to unix (float)
    
    '''determine how much drift has occured betweeen first and last frame (pixels)'''
    x_drifts = np.subtract(positions[1,:,0], positions[0,:,0])
    y_drifts = np.subtract(positions[1,:,1], positions[0,:,1])

    time_interval = np.subtract(times[1], times[0])  #time between first and last frame (s)
        
    '''get median drift rate across all stars [px/s] '''
    x_drift_rate = np.median(x_drifts/time_interval)   
    y_drift_rate = np.median(y_drifts/time_interval)
     
    return x_drift_rate, y_drift_rate

def timeEvolveFITS(data, t, coords, x_drift, y_drift, r, stars, x_length, y_length):
    """ Adjusts aperture based on star drift and calculates flux in aperture 
    input: image data (flux in 2d array), image header times, star coords, 
    x per frame drift rate, y per frame drift rate, aperture length to sum flux in, 
    number of stars, x image length, y image length
    returns: new star coords [x,y], image flux, times as tuple"""

    '''get proper frame times to apply drift'''
    frame_time = Time(t, precision=9).unix   #current frame time from file header (unix)
    drift_time = frame_time - coords[1,3]    #time since previous frame [s]
    
    '''add drift to each star's coordinates based on time since last frame'''
    x = [coords[ind, 0] + x_drift*drift_time for ind in range(0, stars)]
    y = [coords[ind, 1] + y_drift*drift_time for ind in range(0, stars)]
    
    '''get list of indices near edge of frame'''
    EdgeInds = clipCutStars(x, y, x_length, y_length)
    EdgeInds = list(set(EdgeInds))
    EdgeInds.sort()
    
    '''remove stars near edge of frame'''
    xClip = np.delete(np.array(x), EdgeInds)
    yClip = np.delete(np.array(y), EdgeInds)
    
    '''add up all flux within aperture'''
#TODO: background annulus
    l = r    #square aperture size -> centre +/- l pixels (total area: (2l+1)^2 )
    sepfluxes = (sep.sum_circle(data, xClip, yClip, r)[0]).tolist()
  #  fluxSumTimeStart = timer.process_time()
    #fluxes = sum_flux(data, xClip, yClip, l)
  #  fluxSumTimeEnd = timer.process_time()
    
  #  print('time to sum fluxes: ', fluxSumTimeEnd - fluxSumTimeStart)
    
    '''set fluxes at edge to 0'''
    for i in EdgeInds:
     #   fluxes.insert(i, 0)
        sepfluxes.insert(i,0)
        
    '''returns x, y star positions, fluxes at those positions, times'''
  #  star_data = tuple(zip(x, y, fluxes, np.full(len(fluxes), frame_time)))
    star_data = tuple(zip(x, y, sepfluxes, np.full(len(sepfluxes), frame_time)))
    return star_data

def timeEvolveFITSNoDrift(data, t, coords, r, stars, x_length, y_length):
    """ Adjusts aperture based on star drift and calculates flux in aperture 
    input: image data (flux in 2d array), image header times, star coords, 
    aperture length to sum flux in, number of stars, x image length, y image length
    returns: new star coords [x,y], image flux, times as tuple"""
    
    frame_time = Time(t, precision=9).unix  #current frame time from file header (unix)
     
    ''''get each star's coordinates (not accounting for drift)'''
    x = [coords[ind, 0] for ind in range(0, stars)]
    y = [coords[ind, 1] for ind in range(0, stars)]
    
    '''get list of indices near edge of frame'''
    EdgeInds = clipCutStars(x, y, x_length, y_length)
    EdgeInds = list(set(EdgeInds))
    EdgeInds.sort()
    
    '''remove stars near edge of frame'''
    xClip = np.delete(np.array(x), EdgeInds)
    yClip = np.delete(np.array(y), EdgeInds)
    
    '''add up all flux within aperture'''
#TODO: background annulus
    l = r #square aperture size -> centre +/- l pixels (total area: (2l+1)^2 )
    sepfluxes = (sep.sum_circle(data, xClip, yClip, r)[0]).tolist()
   # fluxes = (sum_flux(data, xClip, yClip, l))

    '''set fluxes at edge to 0'''
    for i in EdgeInds:
       # fluxes.insert(i, 0)
        sepfluxes.insert(i,0)
    
    '''returns x, y star positions, fluxes at those positions, times'''
    #star_data = tuple(zip(x, y, fluxes, np.full(len(fluxes), frame_time)))
    star_data = tuple(zip(x, y, sepfluxes, frame_time))
    return star_data

def clipCutStars(x, y, x_length, y_length):
    """ When the aperture is near the edge of the field of view sets flux to zero to prevent 
    fadeout
    input: x coords of stars, y coords of stars, length of image in x-direction, 
    length of image in y-direction
    returns: indices of stars to remove"""

    edgeThresh = 20.          #number of pixels near edge of image to ignore
    
    '''make arrays of x, y coords'''
    xeff = np.array(x)
    yeff = np.array(y) 
    
    '''get list of indices where stars too near to edge'''
    ind = np.where(edgeThresh > xeff)
    ind = np.append(ind, np.where(xeff >= (x_length - edgeThresh)))
    ind = np.append(ind, np.where(edgeThresh > yeff))
    ind = np.append(ind, np.where(yeff >= (y_length - edgeThresh)))
    
    return ind

def sum_flux(data, x_coords, y_coords, l):
    '''function to sum up flux in square aperture of size: centre +/- l pixels
    input: image data [2D array], stars x coords, stars y coords, 'radius' of square side [px]
    returns: list of fluxes for each star'''
    
    '''loop through each x and y coordinate, adding up flux in square (2l+1)^2'''
 
    star_flux_lists = [[data[y][x]
                        for x in range(int(x_coords[star] - l), int(x_coords[star] + l) + 1) 
                        for y in range(int(y_coords[star] - l), int(y_coords[star] + l) + 1)]
                           for star in range(0, len(x_coords))]
    
    star_fluxes = [sum(fluxlist) for fluxlist in star_flux_lists]
    
    return star_fluxes


def dipDetection(fluxProfile, kernel, num):
    """ Checks for geometric dip, and detects dimming using Ricker Wavelet (Mexican Hat) kernel
    input: light curve of star (array of fluxes in each image), Ricker wavelet kernel, 
    current star number
    returns: -1 for no detection or -2 if data unusable,
    if event detected returns frame number and star's light curve"""

    '''' Prunes profiles'''
    light_curve = np.trim_zeros(fluxProfile)
    
    if len(light_curve) == 0:
        print('empty profile: ', num)
        return -2, []  # reject empty profiles
      
    FramesperMin = 2400      #ideal number of frames in a directory (1 minute)
    minSNR = 5             #median/stddev limit

    '''perform checks on data before proceeding'''
   # if len(light_curve) < 10:
   #     print('Light curve too short: star', num)
   #     return -2, []  # reject stars that go out of frame to rapidly
    
    if abs(np.mean(light_curve[:FramesperMin]) - np.mean(light_curve[-FramesperMin:])) > np.std(light_curve[:FramesperMin]):
        print('Tracking failure: star ', num)
        return -2, []  # reject tracking failures
    
    if np.median(light_curve)/np.std(light_curve) < minSNR:
        print('Signal to Noise too low: star', num)
        return -2, []  # reject stars that are very dim, as SNR is too poor

#TODO: remove this, just to save light curve of each star (doesn't look for dips)
   # if num == 0:
   # print('returning star ', num)
    return 25, light_curve
    
    '''convolve light curve with ricker wavelet kernel'''
    #will throw error if try to normalize (sum of kernel too close to 0)
    conv = convolve_fft(light_curve, kernel, normalize_kernel=False)
    minLoc = np.argmin(conv)   #index of minimum value of convolution
    minVal = min(conv)    #minimum of convolution
    
    '''geometric dip detection (greater than 40%)'''
    geoDip = 0.6    #threshold for geometric dip
    norm_trunc_profile = light_curve/np.median(light_curve)  #normalize light curve
    
    if norm_trunc_profile[minLoc] < geoDip:
        
        #frame number of dip
        critFrame = np.where(fluxProfile == light_curve[minLoc])[0]
        print (datetime.datetime.now(), "Detected >40% dip: frame", str(critFrame) + ", star", num)
        
        return critFrame[0], light_curve

    '''if no geometric dip, look for smaller diffraction dip'''
    KernelLength = len(kernel.array)
    
    #check if dip is at least one kernel length from edge
    if KernelLength <= minLoc < len(light_curve) - KernelLength:
        
        edgeBuffer = 10     #how many elements from beginning/end of the convolution to exclude
        bkgZone = conv[edgeBuffer: -edgeBuffer]        #background
    
        dipdetection = 3.75  #dip detection threshold 
        
    else:
        print('event cutoff star: ', num)
        return -2, []  # reject events that are cut off at the start/end of time series

    #if minimum < background - 3.75*sigma
    if minVal < np.mean(bkgZone) - dipdetection * np.std(bkgZone):  

        critFrame = np.where(fluxProfile == light_curve[minLoc])[0]
        print('found significant dip in star: ', num, ' at frame: ', critFrame[0])
        
        return critFrame[0], light_curve
        
    else:
        return -1, []  # reject events that do not pass dip detection


def getSizeFITS(filenames):
    """ gets dimensions of fits 'video' """
    """FITS images are in directories of 1 minute of data with ~2400 images
    input: list of filenames in directory
    returns: width, height of fits image, number of images in directory, 
    list of header times for each image"""
    
    '''get names of first and last image in directory'''
    filename_first = filenames[0]
    frames = len(filenames)     #number of images in directory

    '''get width/height of images from first image'''
    file = fits.open(filename_first)
    header = file[0].header
    width = header['NAXIS1']
    height = header['NAXIS1']

    return width, height, frames


def importFramesFITS(parentdir, filenames, start_frame, num_frames, bias):
    """ reads in frames from fits files starting at frame_num
    input: parent directory (minute), list of filenames to read in, starting frame number, how many frames to read in, 
    bias image (2D array of fluxes)
    returns: array of image data arrays, array of header times of these images"""

    imagesData = []    #array to hold image data
    imagesTimes = []   #array to hold image times
    
    '''list of filenames to read between starting and ending points'''
    files_to_read = [filename for i, filename in enumerate(filenames) if i >= start_frame and i < start_frame + num_frames]

    '''get data from each file in list of files to read, subtract bias frame (add 100 first, don't go neg)'''
    for filename in files_to_read:

        file = fits.open(filename)
        
        header = file[0].header
        
        data = file[0].data - bias
        headerTime = header['DATE-OBS']
        
        #change time if time is wrong (29 hours)
#TODO: path object name change
        hour = str(headerTime).split('T')[1].split(':')[0]
        fileMinute = str(headerTime).split(':')[1]
        dirMinute = parentdir.split('_')[1].split('.')[1]
        
        #check if hour is bad, if so take hour from directory name and change header
        if int(hour) > 23:
            
            #directory name has local hour, header has UTC hour, need to convert (+4)
            #on red don't need to convert between UTC and local (local is UTC)
#TODO: path object name change
            newLocalHour = int(parentdir.split('_')[1].split('.')[0])
        
            if int(fileMinute) < int(dirMinute):
               # newUTCHour = newLocalHour + 4 + 1     #add 1 if hour changed over during minute
                newUTCHour = newLocalHour + 1         #add 1 if hour changed over during minute
            else:
                #newUTCHour = newLocalHour + 4
                newUTCHour  = newLocalHour  
        
            #replace bad hour in timestamp string with correct hour
            newUTCHour = str(newUTCHour)
            newUTCHour = newUTCHour.zfill(2)
        
            replaced = str(headerTime).replace('T' + hour, 'T' + newUTCHour).strip('b').strip(' \' ')
        
            #encode into bytes
            #newTimestamp = replaced.encode('utf-8')
            headerTime = replaced
            
        file.close()

        
      #  print('File read in time: ', fileReadinEnd - fileReadinStart)

        imagesData.append(data)
        imagesTimes.append(headerTime)
         
    '''make into array'''
    imagesData = np.array(imagesData, dtype='float64')
    
    '''reshape, make data type into floats'''
    if imagesData.shape[0] == 1:
        imagesData = imagesData[0]
        imagesData = imagesData.astype('float64')
        
    return imagesData, imagesTimes

#############
# RCD reading section - MJM 20210827
#############

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

#TODO: add Gain keyword
def importFramesRCD(parentdir, filenames, start_frame, num_frames, bias):
    """ reads in frames from .rcd files starting at frame_num
    input: parent directory (minute), list of filenames to read in, starting frame number, how many frames to read in, 
    bias image (2D array of fluxes)
    returns: array of image data arrays, array of header times of these images"""
    
    imagesData = []    #array to hold image data
    imagesTimes = []   #array to hold image times
    
    hnumpix = 2048
    vnumpix = 2048
    imgain = 'low'
#TODO: add gain keyword
    
    '''list of filenames to read between starting and ending points'''
    files_to_read = [filename for i, filename in enumerate(filenames) if i >= start_frame and i < start_frame + num_frames]
    
    for filename in files_to_read:


        data, header = readRCD(filename)
        headerTime = header['timestamp']

        images = nb_read_data(data)
        image = split_images(images, hnumpix, vnumpix, imgain)
        image = np.subtract(image,bias)

        #change time if time is wrong (29 hours)
        hour = str(headerTime).split('T')[1].split(':')[0]
        fileMinute = str(headerTime).split(':')[1]
        dirMinute = parentdir.split('_')[1].split('.')[1]
        
        #check if hour is bad, if so take hour from directory name and change header
        if int(hour) > 23:
            
            #directory name has local hour, header has UTC hour, need to convert (+4)
            #for red: local time is UTC time (don't need +4)
#TODO: pathlib name
            newLocalHour = int(parentdir.split('_')[1].split('.')[0])
        
            if int(fileMinute) < int(dirMinute):
                #newUTCHour = newLocalHour + 4 + 1     #add 1 if hour changed over during minute
                newUTCHour = newLocalHour + 1
            else:
                #newUTCHour = newLocalHour + 4
                newUTCHour = newLocalHour
        
            #replace bad hour in timestamp string with correct hour
            newUTCHour = str(newUTCHour)
            newUTCHour = newUTCHour.zfill(2)
        
            replaced = str(headerTime).replace('T' + hour, 'T' + newUTCHour).strip('b').strip(' \' ')
        
            #encode into bytes
            headerTime = replaced

        fileReadinEnd = timer.process_time()

        imagesData.append(image)
        imagesTimes.append(headerTime)

    '''make into array'''
    imagesData = np.array(imagesData, dtype='float64')
    
    '''reshape, make data type into floats'''
    if imagesData.shape[0] == 1:
        imagesData = imagesData[0]
        imagesData = imagesData.astype('float64')
        
    return imagesData, imagesTimes

def runParallel(folder, MasterBiasList, ricker_kernel, exposure_time):
    firstOccSearch(folder, MasterBiasList, ricker_kernel, exposure_time)
    gc.collect()
#TODO: probably need to add gain keyword

#############
# End RCD section
#############

def getBias(filepath, numOfBiases):
    """ get median bias image from a set of biases (length =  numOfBiases) from filepath
    input: filepath to bias image directory, number of bias  images to take median from
    return: median bias image"""
    
    print('Calculating median bias...')
    
    # Added a check to see if the fits conversion has been done.
    # Comment out if you only want to check for presence of fits files.
    # If commented out, be sure to uncomment the 'if not glob(...)' below
#TODO: path stuff
    if os.path.isfile(filepath + 'converted.txt') == False:
        with open(filepath + 'converted.txt', 'a'):
            os.utime(filepath + 'converted.txt')
            os.system("python .\\RCDtoFTS.py "+filepath)
    else:
        print('Already converted raw files to fits format.')
        print('Remove file converted.txt if you want to overwrite.')

    # if not glob(filepath[0]+'/*.fits'):
    #     print('converting biases to .fits')
    #     os.system("python ..\\..\\RCDtoFTS2.py "+filepath[0])
    
#TODO: path stuff
    '''get list of bias images to combine'''
    biasFileList = glob(filepath + '*.fits')
    biases = []   #list to hold bias data
    
#TODO: add in .rcd option here
    '''append data from each bias image to list of biases'''
    for i in range(0, numOfBiases):
        biases.append(fits.getdata(biasFileList[i]))
    
    '''take median of bias images'''
    biasMed = np.median(biases, axis=0)
    
    return biasMed

def getDateTime(folder):
    """function to get date and time of folder, then make into python datetime object
    input: filepath 
    returns: datetime object"""
    
#TODO: path stuff (may change with datetimes)
    #time is in format ['hour', 'minute', 'second', 'msec']
    #folderTime = folder.split('_')[-1].strip('\\').split('.')  #get time folder was created from folder name
    #folderDate = folder.split('_')[0].split('\\')[-1]          #get date folder was created from folder name
    #change for LINUX
    folderTime = folder.split('_')[-1].strip('/').split('.')  #get time folder was created from folder name
    folderDate = folder.split('_')[0].split('/')[-1]          #get date folder was created from folder name
    
    #date is in format ['year', 'month', 'day']
    folderDate = datetime.date(int(folderDate[:4]), int(folderDate[4:6]), int(folderDate[-2:]))  #convert to date object
    folderTime = datetime.time(int(folderTime[0]), int(folderTime[1]), int(folderTime[2]))       #convert to time object
    folderDatetime = datetime.datetime.combine(folderDate, folderTime)                     #combine into datetime object
    
    return folderDatetime

def makeBiasSet(filepath, numOfBiases):
    """ get set of median-combined biases for entire night that are sorted and indexed by time,
    these are saved to disk and loaded in when needed
    input: filepath (string) to bias image directories, number of biases images to combine for master
    return: array with bias image times and filepaths to saved biases on disk"""
#TODO: filepath stuff 
    biasFolderList = glob(filepath + '*/')  #list of bias folders
    
    ''' create folder for results, save bias images '''
    day_stamp = datetime.date.today()
    if not os.path.exists('./ColibriArchive/' + str(day_stamp)):
        os.makedirs('./ColibriArchive/' + str(day_stamp))
    
    if not os.path.exists('./ColibriArchive/' + str(day_stamp) + '/masterBiases/'):
        os.makedirs('./ColibriArchive/' + str(day_stamp) + '/masterBiases/')
        
    #make list of times and corresponding master bias images
    biasList = []
    
    #loop through each folder of biases
    for folder in biasFolderList:
        masterBiasImage = getBias(folder, numOfBiases)      #get median combined image from this folder
        
        #save as .fits file if doesn't already exist
        hdu = fits.PrimaryHDU(masterBiasImage)
        #change for / for LINUX
       # biasFilepath = './ColibriArchive/' + str(day_stamp) + '/masterBiases/' + folder.split('\\')[-2] + '.fits'
        biasFilepath = './ColibriArchive/' + str(day_stamp) + '/masterBiases/' + folder.split('/')[-2] + '.fits'
#TODO: filepath stuff        
        if not os.path.exists(biasFilepath):
            hdu.writeto(biasFilepath)
        
        folderDatetime = getDateTime(folder)
        
        biasList.append((folderDatetime, biasFilepath))
    
    #package times and filepaths into array, sort by time
    biasList = np.array(biasList)
    ind = np.argsort(biasList, axis=0)
    biasList = biasList[ind[:,0]]
    print('bias times: ', biasList[:,0])
    
    return biasList

def chooseBias(obs_folder, MasterBiasList):
    """ choose correct master bias by comparing time to the observation time
    input: filepath to current minute directory, 2D numpy array of [bias datetimes, bias filepaths]
    returns: bias image that is closest in time to observation"""
    
    #current hour of observations
    current_dt = getDateTime(obs_folder)
    print('current date: ', current_dt)
    
    '''make array of time differences between current and biases'''
    bias_diffs = np.array(abs(MasterBiasList[:,0] - current_dt))
    bias_i = np.argmin(bias_diffs)    #index of best match
    
    '''select best master bias using above index'''
    bias_dt = MasterBiasList[bias_i][0]
    bias_image = MasterBiasList[bias_i][1]
                
    #load in new master bias image
    print('current bias time: ', bias_dt)
    bias = fits.getdata(bias_image)
        
    return bias

 
def firstOccSearch(file, MasterBiasList, kernel, exposure_time):
    """ formerly 'main'
    Detect possible occultation events in selected file and archive results 
    
    input: name of current folder, Rickerwavelet kernel, camera exposure time
    
    output: printout of processing tasks, .npy file with star positions (if doesn't exist), 
    .txt file for each occultation event with names of images to be saved, the time 
    of that image, flux of occulted star in image
    """
#TODO: remove
    global prev_star_pos   #star positions from last image of previous mintue
    global radii           #star half light radii (determined by initial sep.extract)
    
    print (datetime.datetime.now(), "Opening:", file)
    
#TODO: path stuff
    ''' create folder for results '''
    day_stamp = datetime.date.today()
    if not os.path.exists('./ColibriArchive/' + str(day_stamp)):
        os.makedirs('./ColibriArchive/' + str(day_stamp))
        
        
    '''load in appropriate master bias image from pre-made set'''
    bias = chooseBias(file, MasterBiasList)

    ''' adjustable parameters '''
    ap_r = 3.   #radius of aperture for flux measuremnets

    ''' get list of image names to process'''
    if RCDfiles == True: # Option for RCD or fits import - MJM 20210901
#TODO: path stuff
        filenames = glob(file + '*.rcd')
        filenames.sort()
    else:
        filenames = glob(file + '*.fits')   
        filenames.sort()

    del filenames[0]
    
    #CHANGE SLASH FOR WINDOWS/LINUX
    #TODO: path stuff
    field_name = filenames[0].split('/')[2].split('_')[0]   #which of 11 fields are observed
 #@   field_name = filenames[0].split('\\')[2].split('_')[0]

    pier_side = filenames[0].split('-')[1].split('_')[0]     #which side of pier was scope on
    
    ''' get 2d shape of images, number of image in directory'''
    if RCDfiles == True:
        x_length, y_length, num_images = getSizeRCD(filenames) 
    else:
        x_length, y_length, num_images = getSizeFITS(filenames)

    print (datetime.datetime.now(), "Imported", num_images, "frames")
   
    '''check if enough images in folder'''
    #minNumImages = len(kernel.array)*3         #3x kernel length
    minNumImages = 30
    if num_images < minNumImages:
        print (datetime.datetime.now(), "Insufficient number of images, skipping...")
        return

    ''' load/create star positional data'''
    if RCDfiles == True: # Choose to open rcd or fits - MJM
        first_frame = importFramesRCD(file, filenames, 0, 1, bias)
        headerTimes = [first_frame[1]] #list of image header times
        last_frame = importFramesRCD(file, filenames, len(filenames)-1, 1, bias)
        print(first_frame[0].shape)
    else:
        first_frame = importFramesFITS(file, filenames, 0, 1, bias)      #data and time from 1st image
        headerTimes = [first_frame[1]] #list of image header times
        last_frame = importFramesFITS(file, filenames, len(filenames)-1, 1, bias) #data and time from last image
        print(first_frame[0].shape)

    headerTimes = [first_frame[1]]                             #list of image header times
    
    #star position file format: x  |  y  | half light radius
    #change for LINUX
  #  star_pos_file = './ColibriArchive/' + str(day_stamp) + '/' + field_name + '_' + pier_side + '_' + file.split('\\')[1] + '_2sig_pos.npy'   #file to save positional data
    star_pos_file = './ColibriArchive/' + str(day_stamp) + '/' + field_name + '_' + pier_side + '_' + file.split('/')[1] + '_2sig_pos.npy'   #file to save positional data
#TODO: filepath stuff
#TODO: star finding

    # Remove position file if it exists - MJM
    if os.path.exists(star_pos_file):
        os.remove(star_pos_file)

    # if no positional data for current field, create it from first_frame
    if not os.path.exists(star_pos_file):
        
        print (datetime.datetime.now(), field_name, 'starfinding...',)
        
        #find stars in first image
        star_find_results = tuple(initialFindFITS(first_frame[0]))
        
        #remove stars where centre is too close to edge of frame
        star_find_results = tuple(x for x in star_find_results if x[0] + ap_r + 3 < x_length and x[0] - ap_r - 3 > 0)
        star_find_results = tuple(y for y in star_find_results if y[1] + ap_r + 3 < x_length and y[1] - ap_r - 3 > 0)
        
      
         #check number of stars for bad first image
        i = 0  #counter used if several images are poor
        min_stars = 30  #minimum stars in an image
        while len(star_find_results) < min_stars:
            #print('too few stars, moving to next image ', len(star_find_results))

            if RCDfiles == True:
                first_frame = importFramesRCD(file, filenames, 1+i, 1, bias)
                headerTimes = [first_frame[1]]
                star_find_results = tuple(initialFindFITS(first_frame[0]))
            else:
                first_frame = importFramesFITS(file, filenames, 1+i, 1, bias)
                headerTimes = [first_frame[1]]
                star_find_results = tuple(initialFindFITS(first_frame[0]))

           # star_find_results = tuple(x for x in star_find_results if x[0] > 250)
        
            #remove stars where centre is too close to edge of frame
            star_find_results = tuple(x for x in star_find_results if x[0] + ap_r + 3 < x_length and x[0] - ap_r - 3 > 0)
            star_find_results = tuple(y for y in star_find_results if y[1] + ap_r + 3 < x_length and y[1] - ap_r - 3 > 0)
            i += 1
            #check if out of bounds
            if (1+i) >= num_images:
                print('no good images in minute: ', file)
                print (datetime.datetime.now(), "Closing:", file)
                print ("\n")
                return -1

        print('star finding file index: ', i)
        #save to .npy file
        np.save(star_pos_file, star_find_results)
        
        #save radii and positions as global variables
        star_find_results = np.array(star_find_results)
        radii = star_find_results[:,-1]
        prev_star_pos = star_find_results[:,:-1]
        
        print ('done')


   #load in initial star positions from last image of previous minute
    initial_positions = prev_star_pos   
	
    #remove stars that have drifted out of frame
    initial_positions = initial_positions[(x_length >= initial_positions[:, 0])]
    initial_positions = initial_positions[(y_length >= initial_positions[:, 1])]

    #save file with updated positions each minute
    #CHANGE SLASH FOR WINDOWS/LINUX
    #file_time_label = file.split('_')[1].split('\\')[0]   #time label for identification
    file_time_label = file.split('_')[1].split('/')[0]   #time label for identification
#TODO: remove 2nd file    
    newposfile = './ColibriArchive/' + str(day_stamp) + '/' + field_name + '_' + file_time_label + '_2sig_pos.npy'   #file to save positional data
    np.save(newposfile, initial_positions)
    
    num_stars = len(initial_positions)      #number of stars in image
    print(datetime.datetime.now(), 'number of stars found: ', num_stars) 
    
    ''' centroid refinements and drift check '''

    drift = False              # variable to check whether stars have drifted since last frame
    
    drift_pos = np.empty([2, num_stars], dtype=(np.float64, 2))  #array to hold first and last positions
    drift_times = []   #list to hold times for each set of drifted coords
    GaussSigma = np.mean(radii * 2. / 2.35)  # calculate gaussian sigma for each star's light profile

    #star positions and times for first image
    first_drift = refineCentroid(*first_frame, initial_positions, GaussSigma)
    drift_pos[0] = first_drift[0]
    drift_times.append(first_drift[1])

    #star positions and times for last image
    last_drift = refineCentroid(*last_frame, drift_pos[0], GaussSigma)
    drift_pos[1] = last_drift[0]
    drift_times.append(last_drift[1])
    prev_star_pos = drift_pos[1]
    
    # check drift rates
    driftTolerance = 2.5e-2   #px per s
    
    #get median drift rate [px/s] in x and y over the minute
    x_drift, y_drift = averageDrift(drift_pos, drift_times)
    
    #error if header time wrong
    if x_drift == -1:
        return -1
    
    if abs(x_drift) > driftTolerance or abs(y_drift) > driftTolerance:

        drift = True
#TODO: removed because we do have a lot of drift - may need to add in when this is fixed  
      #  if abs(np.median(x_drift)) > 1 or abs(np.median(y_drift)) > 1:
      #      print (datetime.datetime.now(), "Significant drift, skipping ", file)  # find how much drift is too much
      #      return -1
       
    ''' flux and time calculations with optional time evolution '''
      
    #image data (2d array with dimensions: # of images x # of stars)
    data = np.empty([num_images, num_stars], dtype=(np.float64, 4))
    
    #get first image data from initial star positions
    data[0] = tuple(zip(initial_positions[:,0], 
                        initial_positions[:,1], 
                        #sum_flux(first_frame[0], initial_positions[:,0], initial_positions[:,1], ap_r),
                        (sep.sum_circle(first_frame[0], initial_positions[:,0], initial_positions[:,1], ap_r)[0]).tolist(), 
                        np.ones(np.shape(np.array(initial_positions))[0]) * (Time(first_frame[1], precision=9).unix)))
    
    if drift:  # time evolve moving stars
    
        print('drifted - applying drift to photometry', x_drift, y_drift)
        for t in range(1, num_images):
           # imageLoopstart = timer.process_time()
            
           # imageImportstart = timer.process_time()
            #import image
            if RCDfiles == True:
                imageFile = importFramesRCD(file, filenames, t, 1, bias)
                headerTimes.append(imageFile[1])  #add header time to list
            else:
                imageFile = importFramesFITS(file, filenames, t, 1, bias)
                headerTimes.append(imageFile[1])  #add header time to list

          #  imageImportend = timer.process_time()
            
           # print('image import time: ', imageImportend - imageImportstart)
            
            #calculate star fluxes from image
           # fluxCalcstart = timer.process_time()
            data[t] = timeEvolveFITS(*imageFile, deepcopy(data[t - 1]), 
                                     x_drift, y_drift, ap_r, num_stars, x_length, y_length)
            
           # fluxCalcend = timer.process_time()
           # print('flux calculation time: ', fluxCalcend - fluxCalcstart)
            
           # imageLoopend = timer.process_time()
            
           # print('image loop time: ', imageLoopend - imageLoopstart)
            
    else:  # if there is not significant drift, don't account for drift  in photometry
        print('no drift')
        for t in range(1, num_images):
            #import image
            if RCDfiles == True:
                imageFile = importFramesRCD(file, filenames, t, 1, bias)
                headerTimes.append(imageFile[1])  #add header time to list
            else:
                imageFile = importFramesFITS(file, filenames, t, 1, bias)
                headerTimes.append(imageFile[1])  #add header time to list
            
            #calculate star fluxes from image
            data[t] = timeEvolveFITSNoDrift(*imageFile, deepcopy(data[t - 1]), 
                                     ap_r, num_stars, x_length, y_length)

    # data is an array of shape: [frames, star_num, {0:star x, 1:star y, 2:star flux, 3:unix_time}]
    print (datetime.datetime.now(), 'Photometry done.')
   

    ''' Dip detection '''
    
    #Parallel version
    #cores = multiprocessing.cpu_count()  # determine number of CPUs for parallel processing
    
    #perform dip detection and for all stars
    #results array: frame # of event (if found, -1 or -2 otherwise) | light curve for star
   # results = np.array(Parallel(n_jobs=1)(
   #     delayed(dipDetection)(data[:, star, 2], kernel, star) for star in range(0, num_stars)))
    
    #non parallel version (for easier debugging)
    results = []
    for star in range(0, num_stars):
        results.append(dipDetection(data[:, star, 2], kernel, star))
        
    results = np.array(results)
   
    
    event_frames = results[:,0]         #array of event frames (-1 if no event detected, -2 if incomplete data)
    light_curves = results[:,1]         #array of light curves (empty if no event detected)

#TODO: code this in somewhere else
    ''' data archival '''
    telescope = 'R'
    secondsToSave =  0.6    #number of seconds on either side of event to save 
    save_frames = event_frames[np.where(event_frames > 0)]  #frame numbers for each event to be saved
    save_chunk = int(round(secondsToSave / exposure_time))  #save certain num of frames on both sides of event
    save_curves = light_curves[np.where(event_frames > 0)]  #light curves for each star to be saved
    
    
    for f in save_frames:  # loop through each detected event
        date = headerTimes[f][0].split('T')[0]                                 #date of event
        time = headerTimes[f][0].split('T')[1].split('.')[0].replace(':','')   #time of event
        star_coords = initial_positions[np.where(event_frames == f)[0][0]]     #coords of occulted star
   
        
        print(datetime.datetime.now(), ' saving event in frame', f)
        
        star_all_flux = save_curves[np.where(save_frames == f)][0]  #total light curve for current occulted star
        
        #text file to save results in
        #saved file format: 'det_date_time_star#_telescope.txt'
        #columns: fits filename and path | header time (seconds) |  star flux
        savefile = './ColibriArchive/' + str(day_stamp) + "/det_" + date + '_' + time + "_" + str(np.where(event_frames == f)[0][0]) + '_' + telescope + ".txt"
#TODO: filepath stuff       
        #open file to save results
        with open(savefile, 'w') as filehandle:
            
            #file header
            filehandle.write('#\n#\n#\n#\n')
            filehandle.write('#    Event File: %s\n' %(filenames[f]))
            filehandle.write('#    Star Coords: %f %f\n' %(star_coords[0], star_coords[1]))
            filehandle.write('#    DATE-OBS: %s\n' %(headerTimes[f][0]))
            filehandle.write('#    Telescope: %s\n' %(telescope))
            #TODO: filepath stuff CHANGE SLASH FOR WINDOWS/LINUX
           # filehandle.write('#    Field: %s\n' %(filenames[f].split('_')[1]).split('\\')[1])
            filehandle.write('#    Field: %s\n' %(filenames[f].split('_')[1]).split('/')[1])
            filehandle.write('#\n#\n#\n#\n')
            filehandle.write('#filename     time      flux\n')
          
            
           #save data
            if f - save_chunk <= 0:  # if chunk includes lower data boundary, start at 0
        
                files_to_save = [filename for i, filename in enumerate(filenames) if i >= 0 and i < f + save_chunk]  #list of filenames to save
                star_save_flux = star_all_flux[np.where(np.in1d(filenames, files_to_save))[0]]                       #part of light curve to save
      
            #loop through each frame to be saved
                for i in range(0, len(files_to_save)):  
                    filehandle.write('%s %f  %f\n' % (files_to_save[i], float(headerTimes[:f + save_chunk][i][0].split(':')[2].split('Z')[0]), star_save_flux[i]))

            else:  # if chunk does not include lower data boundary
        
                if f + save_chunk >= num_images:  # if chunk includes upper data boundary, stop at upper boundary
        
                    files_to_save = [filename for i, filename in enumerate(filenames) if i >= f - save_chunk and i < num_images - f + save_chunk] #list of filenames to save
                    star_save_flux = star_all_flux[np.where(np.in1d(filenames, files_to_save))[0]]                                                #part of light curve to save
       
                    #loop through each frame to save
                    for i in range(0, len(files_to_save)): 
                        filehandle.write('%s %f %f\n' % (files_to_save[i], float(headerTimes[f - save_chunk:][i][0].split(':')[2].split('Z')[0]), star_save_flux[i]))

                else:  # if chunk does not include upper data boundary

                    files_to_save = [filename for i, filename in enumerate(filenames) if i >= f - save_chunk and i < f + save_chunk]   #list of filenames to save

                    star_save_flux = star_all_flux[np.where(np.in1d(filenames, files_to_save))[0]]                        #part of light curve to save                    
                   
                    #loop through each frame to save
                    for i in range(0, len(files_to_save)): 
                        filehandle.write('%s %f %f\n' % (files_to_save[i], float(headerTimes[f - save_chunk:f + save_chunk][i][0].split(':')[2].split('Z')[0]), star_save_flux[i]))
                    

    ''' printout statements'''
    print (datetime.datetime.now(), "Rejected Stars: ", round(((num_stars - len(save_frames)) / num_stars)*100, 2), "%")
    print (datetime.datetime.now(), "Total stars in field:", num_stars)
    print (datetime.datetime.now(), "Candidate events in this minute:", len(save_frames))
    print (datetime.datetime.now(), "Closing:", file)
    print ("\n")


"""---------------------------------CODE STARTS HERE-------------------------------------------"""
RCDfiles = True # If you want read in RCD files directly, this should be set to True. Otherwise, fits conversion will take place.
runPar = False # True if you want to run directories in parallel

#TODO: add in gain options
if __name__ == '__main__':
    '''get filepaths'''
    #TODO: path stuff
   # directory = './PipelineTesting/'
   # directory = './ColibriData/BiasHourTests/'         #directory that contains .fits image files for 1 night
    #directory = Path(directory)    #directory that contains subdirectories of images
#TODO: filepath stuff
    directory = './ColibriData/RedData/20210804/'
    folder_list = glob(directory + '*/')    #each folder has 1 minute of data (~2400 images)
  #  folder_list = list(directory.iterdir())  #list of subdirectories
    
   # bias_list = [f for f in folder_list if 'Bias' in f.parts]
    folder_list = [f for f in folder_list if 'Bias\\' not in f]  #don't run pipeline on bias images
    folder_list = [folder_list[0]]
    folder_list.sort()
    #folder_list = [folder_list[0]]
    
    
    print ('folders', folder_list)
         
    '''get median bias image to subtract from all frames'''
#TODO: path stuff
    NumBiasImages = 9                             #number of bias images to combine in median bias image
    #get 2d np array with bias datetimes and master bias filepaths
    MasterBiasList = makeBiasSet(directory + 'Bias/', NumBiasImages)

    ''' prepare RickerWavelet/Mexican hat kernel to convolve with light curves'''
    exposure_time = 0.025    # exposure length in seconds
    expected_length = 0.15   # related to the characteristic scale length, length of signal to boost in convolution, may need tweaking/optimizing
    refresh_rate = 2.        # number of seconds (as float) between centroid refinements

    kernel_frames = int(round(expected_length / exposure_time))   # width of kernel
    ricker_kernel = RickerWavelet1DKernel(kernel_frames)          # generate kernel

    '''variables to account for long term drift (over hours)'''
    prev_star_pos = []         #variable to hold star positions from last image of previous minute
    radii = []                 #list to hold half-light radii of stars (GaussSigma)

    ''''run pipeline for each folder of data'''
    if runPar == True:
        print('Running in parallel...')
        start_time = timer.time()
        pool_size = multiprocessing.cpu_count() - 2
        pool = Pool(pool_size)
        args = ((folder_list[f], MasterBiasList, ricker_kernel, exposure_time) for f in range(0,len(folder_list)))
        pool.starmap(runParallel,args)
        pool.close()
        pool.join()

        end_time = timer.time()
        print('Ran for %s seconds' % (end_time - start_time))
    else:
        for f in range(0, len(folder_list)):
           
            # Added a check to see if the fits conversion has been done. - MJM 
#TODO: path stuff
            #only run this check if we want to process fits files - RAB
            if RCDfiles == False:
                    if os.path.isfile(folder_list[f] + 'converted.txt') == False:
                        print('converting to .fits')
                        with open(folder_list[f] + 'converted.txt', 'a'):
                            os.utime(folder_list[f] + 'converted.txt')
                        os.system("python .\\RCDtoFTS.py " + folder_list[f])
                    else:
                        print('Already converted raw files to fits format.')
                        print('Remove file converted.txt if you want to overwrite.')

            print('running on... ', folder_list[f])

            start_time = timer.time()

            print('Running sequentially...')
            firstOccSearch(folder_list[f], MasterBiasList, ricker_kernel, exposure_time)
            
            gc.collect()

            end_time = timer.time()
            print('Ran for %s seconds' % (end_time - start_time))

    '''once initial folders complete, check if folders have been added until no more are added'''
    '''
    while (len(os.listdir(directory)) > (len(folder_list) + 1)):

        #get current list of folders in directory
        new_folder_list = glob(directory + '*/') 
        new_folder_list = [f for f in new_folder_list if 'Bias' not in f]
        
        #get list of new folders that have been added
        new_folders = list(set(new_folder_list).difference(set(folder_list)))
        
        #process new folders and add them to the list
        if new_folders:
            for f in range(0, len(new_folders)):
                
                print('running on... ', new_folders[f])
                firstOccSearch(new_folders[f], bias, ricker_kernel, exposure_time)
                folder_list.append(new_folders[f])
                gc.collect()
      '''         
