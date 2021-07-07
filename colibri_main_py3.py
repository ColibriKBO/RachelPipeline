"""July 2021 edit"""

import sep
import numpy as np
from glob import glob
from astropy.io import fits
from astropy.convolution import convolve_fft, RickerWavelet1DKernel
from astropy.time import Time
from joblib import delayed, Parallel
from copy import deepcopy
import multiprocessing
import datetime
import matplotlib.pyplot as plt
import os
import gc


def initialFindFITS(data):
    """ Locates the stars in the initial time slice 
    input: flux data in 2D array for a fits image
    returns: [x, y, half light radius] of all stars in pixels"""

    ''' Background extraction for initial time slice'''
    data_new = deepcopy(data)
    bkg = sep.Background(data_new)
    bkg.subfrom(data_new)
    thresh = 3. * bkg.globalrms  # set detection threshold to mean + 3 sigma

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
    l = r    #square aperture size -> centre +/- l pixels (total area: (2l+1)^2 )
    #sepfluxes = (sep.sum_circle(data, xClip, yClip, r)[0]).tolist()
    fluxes = sum_flux(data, xClip, yClip, l)
    
    '''set fluxes at edge to 0'''
    for i in EdgeInds:
        fluxes.insert(i, 0)
     #   sepfluxes.insert(i,0)
        
    '''returns x, y star positions, fluxes at those positions, times'''
    star_data = tuple(zip(x, y, fluxes, frame_time))
    #star_data = tuple(zip(x, y, sepfluxes, frame_time)
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
    l = r #square aperture size -> centre +/- l pixels (total area: (2l+1)^2 )
   # sepfluxes = (sep.sum_circle(data, xClip, yClip, r)[0]).tolist()
    fluxes = (sum_flux(data, xClip, yClip, l))
    
    #plot comparing two flux calcs 
    #TODO: remove this after testing
    '''
    plt.scatter(fluxes, sepfluxes)
    plt.plot(range(0,13000), range(0,13000))
    plt.xlabel('square sum function')
    plt.ylabel('sep sum function')
    plt.show()
    plt.close()  
    '''
    '''set fluxes at edge to 0'''
    for i in EdgeInds:
        fluxes.insert(i, 0)
       # sepfluxes.insert(i,0)
    
    '''returns x, y star positions, fluxes at those positions, times'''
    star_data = tuple(zip(x, y, fluxes, frame_time))
    #star_data = tuple(zip(x, y, sepfluxes, frame_time)
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
    

    star_fluxes = []  #empty array to hold star fluxes
    
    '''loop through each x and y coordinate, adding up flux in square (2l+1)^2'''
    
    for star in range(0, len(x_coords)):
        
        #set range to sum flux (side length of square = (2l+1))
        xrange = range(int(x_coords[star] - l), int(x_coords[star] + l) + 1)
        yrange = range(int(y_coords[star] - l), int(y_coords[star] + l) + 1)
        
        flux_sum = 0   #flux of star at current coord

        #add up flux within square
        for x in xrange:
            for y in yrange:
                flux_sum += data[y][x]

        star_fluxes.append(flux_sum)    #add this total flux to list

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
    minSNR = 10             #median/stddev limit
    NumBkgndElements = 200

    
    '''perform checks on data before proceeding'''
    if len(light_curve) < FramesperMin/4:
        print('Light curve too short: star', num)
        return -2, []  # reject stars that go out of frame to rapidly
    
    if abs(np.mean(light_curve[:FramesperMin]) - np.mean(light_curve[-FramesperMin:])) > np.std(light_curve[:FramesperMin]):
        print('Tracking failure: star ', num)
        return -2, []  # reject tracking failures
    
    if np.median(light_curve)/np.std(light_curve) < minSNR:
        print('Signal to Noise too low: star', num)
        return -2, []  # reject stars that are very dim, as SNR is too poor

    #TODO: remove this, just to save light curve of each star (doesn't look for dips)
   # if num == 0:
  #  print('returning star ', num)
  #  return num, light_curve
    

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


def importFramesFITS(filenames, start_frame, num_frames, bias):
    """ reads in frames from fits files starting at frame_num
    input: list of filenames to read in, starting frame number, how many frames to read in, 
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
        time = header['DATE-OBS']
        
        #uncomment below to save new bias subtracted images
      #  file[0].data = data
      #  file.writeto('ColibriArchive/biassubtracted/00/'+'sub_p100_'+filename.split('\\')[-1])
        file.close()

        imagesData.append(data)
        imagesTimes.append(time)
         
    '''make into array'''
    imagesData = np.array(imagesData, dtype='float64')
    
    '''reshape, make data type into floats'''
    if imagesData.shape[0] == 1:
        imagesData = imagesData[0]
        imagesData = imagesData.astype('float64')
        
    return imagesData, imagesTimes


def getBias(filepath, numOfBiases):
    """ get median bias image from a set of biases (length =  numOfBiases) from filepath
    input: filepath to bias image directory, number of bias  images to take median from
    return: median bias image"""
    
    print('Calculating median bias...')
    
    '''get list of bias images to combine'''
    biasFileList = glob(filepath[0] + '*.fits')
    #biasFileList = glob(filepath + '*.fits')
    biases = []   #list to hold bias data
    
    '''append data from each bias image to list of biases'''
    for i in range(0, numOfBiases):
        biases.append(fits.getdata(biasFileList[i]))
    
    '''take median of bias images'''
    biasMed = np.median(biases, axis=0)
    
    return biasMed

 
def firstOccSearch(file, bias, kernel, exposure_time):
    """ formerly 'main'
    Detect possible occultation events in selected file and archive results 
    
    input: name of current folder, bias image, Rickerwavelet kernel, 
    camera exposure time
    
    output: printout of processing tasks, .npy file with star positions (if doesn't exist), 
    .txt file for each occultation event with names of images to be saved, the time 
    of that image, flux of occulted star in image
    """
    global starfindTime    #time when star finding file was created
    global driftperSec     #per second drift rates (to account for significant drift across minutes)

    print (datetime.datetime.now(), "Opening:", file)

    ''' create folder for results '''
    day_stamp = datetime.date.today()
    if not os.path.exists('./ColibriArchive/' + str(day_stamp)):
        os.makedirs('./ColibriArchive/' + str(day_stamp))

    ''' adjustable parameters '''
    ap_r = 3.   #radius of aperture for flux measuremnets

    ''' get list of image names to process'''
    filenames = glob(file + '*.fits')   
    filenames.sort() 
    field_name = filenames[0].split('\\')[2].split('_')[0]
    
    ''' get 2d shape of images, number of image in directory'''
    x_length, y_length, num_images = getSizeFITS(filenames) 
    print (datetime.datetime.now(), "Imported", num_images, "frames")
   
    '''check if enough images in folder'''
    #minNumImages = len(kernel.array)*3         #3x kernel length
    minNumImages = 90
    if num_images < minNumImages:
        print (datetime.datetime.now(), "Insufficient number of images, skipping...")
        return

    ''' load/create star positional data file (.npy)
    star position file format: x  |  y  | half light radius'''
    #TODO: add pier side, add filename, unix time
    first_frame = importFramesFITS(filenames, 0, 1, bias)      #data and time from 1st image
    headerTimes = [first_frame[1]]                             #list of image header times
    last_frame = importFramesFITS(filenames, len(filenames)-1, 1, bias) #data and time from last image
    
    star_pos_file = './ColibriArchive/' + str(day_stamp) + '/' + field_name + '_pos_23.npy'   #file to save positional data

    # if no positional data for current field, create it from first_frame
    if not os.path.exists(star_pos_file):
        
        print (datetime.datetime.now(), field_name, 'starfinding...',)
        
        #find stars in first image
        star_find_results = tuple(initialFindFITS(first_frame[0]))
        
        #TODO: make this work properly
        #if num_stars < 5:
        #    print('too few stars, moving to next image')
        #    star_find_results = tuple(initialFindFITS(first_frame[0]))
        
        #unix time of image used to make star position file
        starfindTime = Time(first_frame[1], precision=9).unix
        
      #TODO: remove this once artifact is gone
        star_find_results = tuple(x for x in star_find_results if x[0] > 250)
        
        #save to .npy file
        np.save(star_pos_file, star_find_results)
        
        print ('done')

    #get time difference since star position file made to determine long term drift
    currentTime = Time(first_frame[1], precision=9).unix    #unix time of 1st frame in minute
    timeDiff = currentTime - starfindTime    #time between current minute and image used for star position file
    
    #load in initial star positions and radii
    initial_positions = np.load(star_pos_file, allow_pickle = True)   #change in python3, allow_pickle set to false by default
    radii = initial_positions[:,-1]                                   #make array of radii
    initial_positions = initial_positions[:,:-1]                      #remove radius column
    
    #apply overeall drift since star file creation to initial coords
    initial_positions[0] = initial_positions[0] + driftperSec[0]*timeDiff
    initial_positions[1] = initial_positions[1] + driftperSec[1]*timeDiff
    
    #TODO: remove this, for testing only
    newposfile = './ColibriArchive/' + str(day_stamp) + '/' + field_name + '_pos.npy'   #file to save positional data
    np.save(newposfile, initial_positions)
    
    num_stars = len(initial_positions)      #number of stars in image
    print(datetime.datetime.now(), 'number of stars found: ', num_stars) 
    
    ''' centroid refinements and drift check '''
    #TODO: set default back to false after testing
    drift = True              # variable to check whether stars have drifted since last frame
    
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
    
    # check drift rates
    #TODO: need new drift tolerance threshold
    driftTolerance = 1e-2   #px per frame
    
    #get median drift rate [px/s] in x and y over the minute
    x_drift, y_drift = averageDrift(drift_pos, drift_times)
    driftperSec = [x_drift, y_drift]           #set new global drift rate
    
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
                        sum_flux(first_frame[0], initial_positions[:,0], initial_positions[:,1], ap_r),
                        #(sep.sum_circle(first_frame[0], initial_positions[:,0], initial_positions[:,1], ap_r)[0]).tolist(), 
                        np.ones(np.shape(np.array(initial_positions))[0]) * (Time(first_frame[1], precision=9).unix)))
    
    #drift = False
    if drift:  # time evolve moving stars
    
        print('drifted - applying drift to photometry', x_drift, y_drift)
        for t in range(1, num_images):
            #import image
            imageFile = importFramesFITS(filenames, t, 1, bias)
            headerTimes.append(imageFile[1])  #add header time to list
            
            #calculate star fluxes from image
            data[t] = timeEvolveFITS(*imageFile, deepcopy(data[t - 1]), 
                                     x_drift, y_drift, ap_r, num_stars, x_length, y_length)
            
    else:  # if there is not significant drift, don't account for drift  in photometry
        print('no drift')
        for t in range(1, num_images):
            #import image
            imageFile = importFramesFITS(filenames, t, 1, bias)
            headerTimes.append(imageFile[1])  #add header time to list
            
            #calculate star fluxes from image
            data[t] = timeEvolveFITSNoDrift(*imageFile, deepcopy(data[t - 1]), 
                                     ap_r, num_stars, x_length, y_length)

    # data is an array of shape: [frames, star_num, {0:star x, 1:star y, 2:star flux, 3:unix_time}]
    print (datetime.datetime.now(), 'Photometry done.')
   

    ''' Dip detection '''
    '''
    #Parallel version
    cores = multiprocessing.cpu_count()  # determine number of CPUs for parallel processing
    
    #perform dip detection and for all stars
    #results array: frame # of event (if found, -1 or -2 otherwise) | light curve for star
    results = np.array(Parallel(n_jobs=cores, backend='threading')(
        delayed(dipDetection)(data[:, star, 2], kernel, star) for star in range(0, num_stars)))
   
    
    '''
    #non parallel version (for easier debugging)
    results = []
    for star in range(0, num_stars):
        results.append(dipDetection(data[:, star, 2], kernel, star))
        
    results = np.array(results)
    
    event_frames = results[:,0]         #array of event frames (-1 or -2 if no event detected)
    light_curves = results[:,1]         #array of light curves (empty if no event detected)

    ''' data archival '''
    telescope = 'G'
    secondsToSave =  30    #number of seconds on either side of event to save 
    save_frames = event_frames[np.where(event_frames > 0)]  #frame numbers for each event to be saved
    save_chunk = int(round(secondsToSave / exposure_time))  #save certain num of frames on both sides of event
    save_curves = light_curves[np.where(event_frames > 0)]  #light curves for each star to be saved
    
    
    for f in save_frames:  # loop through each detected event
        date = headerTimes[f][0].split('T')[0]                                 #date of event
        time = headerTimes[f][0].split('T')[1].split('.')[0].replace(':','')   #time of event
        star_coords = initial_positions[np.where(event_frames == f)[0][0]]     #coords of occulted star
   
        
        print(datetime.datetime.now(), ' saving event in frame', f, 'at time', headerTimes[f][0])
        
        star_all_flux = save_curves[np.where(save_frames == f)][0]  #total light curve for current occulted star
        
        #text file to save results in
        #saved file format: 'det_date_time_star#_telescope.txt'
        #columns: fits filename and path | header time (seconds) |  star flux
        savefile = './ColibriArchive/' + str(day_stamp) + "/det_" + date + '_' + time + "_" + str(np.where(event_frames == f)[0][0]) + '_' + telescope + ".txt"
        
        #open file to save results
        with open(savefile, 'w') as filehandle:
            
            #file header
            filehandle.write('#\n#\n#\n#\n')
            filehandle.write('#    Event File: %s\n' %(filenames[f]))
            filehandle.write('#    Star Coords: %f %f\n' %(star_coords[0], star_coords[1]))
            filehandle.write('#    DATE-OBS: %s\n' %(headerTimes[f][0]))
            filehandle.write('#    Telescope: %s\n' %(telescope))
            filehandle.write('#    Field: %s\n' %(filenames[f].split('_')[1]).split('\\')[1])
            filehandle.write('#\n#\n#\n#\n')
            filehandle.write('#filename     time      flux\n')
          
            
           #save data
            if f - save_chunk <= 0:  # if chunk includes lower data boundary, start at 0
        
                files_to_save = [filename for i, filename in enumerate(filenames) if i >= 0 and i < f + save_chunk]  #list of filenames to save
                star_save_flux = star_all_flux[np.where(np.in1d(filenames, files_to_save))[0]]                       #part of light curve to save
      
            #loop through each frame to be saved
                for i in range(0, len(files_to_save)):  
                  #  filehandle.write('%s %f  %f\n' % (files_to_save[i], float(headerTimes[:f + save_chunk][i][0].split(':')[2]), star_save_flux[i]))
                    filehandle.write('%s %f  %f\n' % (files_to_save[i], float(headerTimes[:f + save_chunk][i][0].split(':')[2].split('Z')[0]), star_save_flux[i]))

            else:  # if chunk does not include lower data boundary
        
                if f + save_chunk >= num_images:  # if chunk includes upper data boundary, stop at upper boundary
        
                    files_to_save = [filename for i, filename in enumerate(filenames) if i >= f - save_chunk and i < num_images - f + save_chunk] #list of filenames to save
                    star_save_flux = star_all_flux[np.where(np.in1d(filenames, files_to_save))[0]]                                                #part of light curve to save
       
                    #loop through each frame to save
                    for i in range(0, len(files_to_save)): 
                        #filehandle.write('%s %f %f\n' % (files_to_save[i], float(headerTimes[f - save_chunk:][i][0].split(':')[2]), star_save_flux[i]))
                        filehandle.write('%s %f %f\n' % (files_to_save[i], float(headerTimes[f - save_chunk:][i][0].split(':')[2].split('Z')[0]), star_save_flux[i]))

                else:  # if chunk does not include upper data boundary

                    #files_to_save = [filename for i, filename in enumerate(filenames) if i >= f - save_chunk and i < 1 + save_chunk * 2]   #list of filenames to save
                    files_to_save = [filename for i, filename in enumerate(filenames) if i >= f - save_chunk and i < f + save_chunk]   #list of filenames to save

                    star_save_flux = star_all_flux[np.where(np.in1d(filenames, files_to_save))[0]]                        #part of light curve to save                    
                   
                    #loop through each frame to save
                    for i in range(0, len(files_to_save)): 
                        #filehandle.write('%s %f %f\n' % (files_to_save[i], float(headerTimes[f - save_chunk:f + save_chunk][i][0].split(':')[2]), star_save_flux[i]))
                        filehandle.write('%s %f %f\n' % (files_to_save[i], float(headerTimes[f - save_chunk:f + save_chunk][i][0].split(':')[2].split('Z')[0]), star_save_flux[i]))
                    

    ''' printout statements'''
    print (datetime.datetime.now(), "Rejected Stars: ", round(((num_stars - len(save_frames)) / num_stars)*100, 2), "%")
    print (datetime.datetime.now(), "Total stars in field:", num_stars)
    print (datetime.datetime.now(), "Candidate events in this minute:", len(save_frames))
    print (datetime.datetime.now(), "Closing:", file)
    print ("\n")

"""---------------------------------CODE STARTS HERE-------------------------------------------"""

'''get filepaths'''     
directory = './ColibriData/202106023/'         #directory that contains .fits image files for 1 night
folder_list = glob(directory + '*/')    #each folder has 1 minute of data (~2400 images)

folder_list = [f for f in folder_list if 'Bias' not in f]  #don't run pipeline on bias images
#folder_list= [f for f in folder_list if '_01'  in f]
print ('folders', folder_list)
     
'''get median bias image to subtract from all frames'''
biasFilepath = glob(directory + '/Bias/'+ '*/')
#biasFilepath = directory + '/Bias/'
NumBiasImages = 9                             #number of bias images to combine in median bias image
bias = getBias(biasFilepath, NumBiasImages)    #take median of NumBiasImages to use as bias


''' prepare RickerWavelet/Mexican hat kernel to convolve with light curves'''
exposure_time = 0.025    # exposure length in seconds
expected_length = 0.15   # related to the characteristic scale length, length of signal to boost in convolution, may need tweaking/optimizing
refresh_rate = 2.        # number of seconds (as float) between centroid refinements

kernel_frames = int(round(expected_length / exposure_time))   # width of kernel
ricker_kernel = RickerWavelet1DKernel(kernel_frames)          # generate kernel

'''variables to account for long term drift (over hours)'''
driftperSec = [0., 0.]     #global variable to hold large scale drift rate
starfindtime = 0.0         #global variable to hold time that star find file was made

''''run pipeline for each folder of data'''
for f in range(0, len(folder_list)):
    print('running on... ', folder_list[f])
    firstOccSearch(folder_list[f], bias, ricker_kernel, exposure_time)
    
    gc.collect()

'''once initial folders complete, check if folders have been added until no more are added'''

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
            
