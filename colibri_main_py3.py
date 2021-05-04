"""April 2021 edit"""

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


def refineCentroid(data, coords, sigma):
    """ Refines the centroid for each star for a set of test slices of the data cube 
    input: flux data in 2D array for single fits image, coord of stars in image, weighting (Gauss sigma)
    returns: new [x, y] positions """

    '''initial x, y positions'''
    x_initial = [pos[0] for pos in coords]
    y_initial = [pos[1] for pos in coords]
    
    '''use an iterative 'windowed' method to get new position'''
    new_pos = np.array(sep.winpos(data, x_initial, y_initial, sigma, subpix=5))[0:2, :]
    x = new_pos[:][0].tolist()
    y = new_pos[:][1].tolist()
    
    '''returns tuple x, y (python 3: zip(x, y) -> tuple(zip(x,y)))'''
    return tuple(zip(x, y))


def averageDrift(positions, end, frames):
    """ Determines the per-frame drift rate of each star 
    input: array of [x,y] star positions, number of times to do drift check (?), number of frames checking
    drift rate for
    returns: median x, y drift per frame"""
    #TODO: come back and check this

    x_drift = np.array([np.subtract(positions[t, :, 0], positions[t - 1, :, 0]) for t in range(1, end)])
    y_drift = np.array([np.subtract(positions[t, :, 1], positions[t - 1, :, 1]) for t in range(1, end)])
    return np.median(x_drift[1:], 0) / frames, np.median(y_drift[1:], 0) / frames


def timeEvolveFITS(data, coords, x_drift, y_drift, r, stars, x_length, y_length, t):
    """ Adjusts aperture based on star drift and calculates flux in aperture using sep
    input: image data (flux in 2d array), star coords, x per frame drift rate, y per frame drift rate,
    aperture radius to sum flux in, number of stars, x image length, y image length, time
    returns: new star coords, image flux, times"""

    ''''add drift to each star's coordinates'''
    x = [coords[ind, 0] + x_drift[ind] for ind in range(0, stars)]
    y = [coords[ind, 1] + y_drift[ind] for ind in range(0, stars)]
    
    '''get list of indices near edge of frame'''
    EdgeInds = clipCutStars(x, y, x_length, y_length)
    EdgeInds = list(set(EdgeInds))
    EdgeInds.sort()
    
    '''remove stars near edge of frame'''
    xClip = np.delete(np.array(x), EdgeInds)
    yClip = np.delete(np.array(y), EdgeInds)
    
    '''get background levels in aperture area'''
    bkg = np.median(data) * np.pi * r * r
    
    '''add up all flux within aperture'''
    fluxes = (sep.sum_circle(data, xClip, yClip, r)[0] - bkg).tolist()
    
    '''set fluxes at edge to 0'''
    for i in EdgeInds:
        fluxes.insert(i, 0)
        
    '''returns x, y star positions, fluxes at those positions, times'''
    coords = tuple(zip(x, y, fluxes, [t] * len(x)))
    return coords


def timeEvolveFITSNoDrift(data, coords, r, stars, x_length, y_length, t):
    """ calculates flux in aperture using sep, not accounting for drift
    input: image data (flux in 2d array), star coords, aperture radius to sum flux in, 
    number of stars, x image length, y image length, time
    returns: new star coords, image flux, times"""
    
    ''''get each star's coordinates (not accounting for drift)'''
    x = [coords[ind, 0] for ind in range(0, stars)]
    y = [coords[ind, 1] for ind in range(0, stars)]
    
    '''get background levels within aperture area'''
    bkg = np.median(data) * np.pi * r * r
    
    '''add up all flux within aperture'''
    fluxes = (sep.sum_circle(data, x, y, r)[0] - bkg).tolist()
    
    '''returns x, y star positions, fluxes at those positions, times'''
    coords = tuple(zip(x, y, fluxes, [t] * len(x)))
    return coords


def clipCutStars(x, y, x_length, y_length):
    """ When the aperture is near the edge of the field of view, sets flux to zero to prevent fadeout
    input: x coords of stars, y coords of stars, length of image in x-direction, length of image in y-direction
    returns: indices of stars to remove"""

    edgeThresh = 20.          #number of pixels near edge of image to ignore
    
    xeff = np.array(x)
    yeff = np.array(y) 
    
    #gets list of indices where stars too near to edge
    ind = np.where(edgeThresh > xeff)
    ind = np.append(ind, np.where(xeff >= (x_length - edgeThresh)))
    ind = np.append(ind, np.where(edgeThresh > yeff))
    ind = np.append(ind, np.where(yeff >= (y_length - edgeThresh)))
    return ind


def dipDetection(fluxProfile, kernel, num, ):
    """ Detects dimming using Ricker Wavelet (Mexican Hat) kernel for dip detection
    input: light curve of star (array of fluxes), Ricker wavelet kernel, star number
    returns: -1 for no detection or -2 if data unusable, frame number of event if detected"""

    '''' Prunes profiles'''
    light_curve = np.trim_zeros(fluxProfile)
    if len(light_curve) == 0:
        return -2, []  # reject empty profiles
    med = np.median(light_curve)       
    
    #TODO: FramesperMIn = number of file in directory
    FramesperMin = 2400      #minimum number of frames
    minFlux = 5000           #corresponds to S/N = 9
  #  NumBkgndElements = 200

    
    '''perform checks on data before proceeding'''
 #   if len(trunc_profile) < FramesperMin:
 #       return -2  # reject objects that go out of frame rapidly, ensuring adequate evaluation of median flux
    if abs(np.mean(light_curve[:FramesperMin]) - np.mean(light_curve[-FramesperMin:])) > np.std(light_curve[:FramesperMin]):
        print('Tracking failure')
        return -2, []  # reject tracking failures
 #   if np.median(light_curve) < minFlux:
  #      print('Signal to Noise too low')
   #     return -2  # reject stars that are very dim, as SNR is too poor

 #   kernel_plot = kernel.array
    ''''geometric dip detection'''
    # astropy v2.0+ changed convolve_fft quite a bit... see documentation, for now normalize_kernel=False
    conv = convolve_fft(light_curve, kernel, normalize_kernel=False)
    minLoc = np.argmin(conv)   #index of minimum value of convolution
    minVal = min(conv)    #minimum of convolution
    
    geoDip = 0.6    #threshold for geometric dip
    
 
    # if geometric dip (greater than 40%), flag as candidate without template matching
    norm_trunc_profile = light_curve/np.median(light_curve)
    #norm_trunc_profile = conv/np.median(conv)
    
    if norm_trunc_profile[minLoc] < geoDip:
        
        critTime = np.where(fluxProfile == light_curve[minLoc])[0]
       # critTime = np.where(fluxProfile == conv[minLoc])[0]
        
        print (datetime.datetime.now(), "Detected >40% dip: frame", str(critTime) + ", star", num)
       # return critTime[0], norm_trunc_profile
        return critTime[0], light_curve

    ''' if no geometric dip, look for smaller diffraction dip'''
    KernelLength = 30
    Gain = 100.
    NumofBufferElementstoIgnore = 100
    
    #check if minimum is at least one kernel length from edge
    if KernelLength <= minLoc < len(light_curve) - KernelLength:
        
        #get median for area around the minimum
        med = np.median(
            np.concatenate((light_curve[minLoc - NumofBufferElementstoIgnore:minLoc - KernelLength], light_curve[minLoc + KernelLength:minLoc + NumofBufferElementstoIgnore])))
        
        
        #normalized fractional uncertainty
        sigmaP = ((np.sqrt(abs(light_curve) / np.median(light_curve) / Gain)) * Gain)   #Eq 7 in Pass 2018 (Poisson)
        sigmaP = np.where(sigmaP == 0, 0.01, sigmaP)
        light_curve /= med
        
    else:
        return -2, []  # reject events that are cut off at the start/end of time series
    
    edgeBuffer = 10     #how many elements from beginning/end of the convolution to exclude
    bkgZone = conv[edgeBuffer: -edgeBuffer]        #background
    
    dipdetection = 3.75  #dip detection threshold 

    if minVal < np.mean(bkgZone) - dipdetection * np.std(bkgZone):  # dip detection threshold
        print('found significant dip in star: ', num, ' at frame: ')

        critTime = np.where(fluxProfile == light_curve[minLoc])[0]
        return critTime[0], light_curve
        
    else:
        return -1, []  # reject events that do not pass dip detection


def getSizeFITS(filenames):
    """ gets dimensions of fits 'video' """
    """FITS images are in directories of 1 minute of data with ~2400 images
    input: list of filenames in directory
    returns: width, height of fits image, number of images in directory, list of unix times for each image"""
    
    '''get names of first and last image in directory'''
    filename_first = filenames[0]
    filename_last = filenames[-1]
    frames = len(filenames)     #number of images in directory

    '''get width/height of images from first image'''
    file = fits.open(filename_first)
    header = file[0].header
    width = header['NAXIS1']
    height = header['NAXIS1']

    '''get time and date of first image'''
    time_start = Time(header['DATE-OBS'], format='fits', precision=9).unix

    '''get time and date of last image'''
    file = fits.open(filename_last)
    header = file[0].header
    time_end = Time(header['DATE-OBS'], format='fits', precision=9).unix

    '''make list of times for each image'''
    time_list = np.linspace(time_start, time_end, frames)

    return width, height, frames, time_list


def importFramesFITS(filenames, start_frame, num_frames, bias):
    """ reads in frames from fits files starting at frame_num
    input: list of filenames to read in, starting frame number, how many frames to read in, bias image
    returns: array of image data arrays"""
    
    imagesData = []    #array to hold image data
    
    '''list of filenames to read between starting and ending points'''
    files_to_read = [filename for i, filename in enumerate(filenames) if i >= start_frame and i < start_frame + num_frames]

    '''get data from each file in list of files to read, subtract bias frame (add 100 first, don't go neg)'''
    for filename in files_to_read:
        
         #need to add 100 to each flux value before subtracting bias to avoid negative values
         data = fits.getdata(filename) + 100.0 - bias
         imagesData.append(data)
         
    '''make into array'''
    imagesData = np.array(imagesData, dtype='float64')
    
    '''reshape, make data type into floats'''
    if imagesData.shape[0] == 1:
        imagesData = imagesData[0]
        imagesData = imagesData.astype('float64')
        
    return imagesData


def getBias(filepath, numOfBiases):
    """ get median bias image from a set of biases (length =  numOfBiases) from filepath
    input: filepath to bias image directory, number of bias  images to take median from
    return: median bias image"""
    
    print('Calculating median bias...')
    
    '''get list of bias images to combine'''
    biasFileList = glob(filepath + '*.fits')
    biases = []   #list to hold bias data
    
    '''append data from each bias image to list of biases'''
    for i in range(0, numOfBiases):
        biases.append(fits.getdata(biasFileList[i]))
    
    '''take median of bias images'''
    biasMed = np.median(biases, axis=0)
    
    return biasMed

    
def processImages(file, field_name, bias, kernel, exposure_time, evolution_frames):
    """ formerly 'main'
    Detect possible occultation events in selected file and archive results 
    
    input: name of current folder, name of field, bias image, Rickerwavelet kernel, 
    camera exposure time, number of frames between centroid refinements
    
    output: printout of processing tasks, .npy file with star positions, .txt file for each occultation
    event with names of images to be saved, the unix time of that image, flux of occulted star in image
    """

    print (datetime.datetime.now(), "Opening:", file)

    ''' create folder for results '''
    day_stamp = datetime.date.today()
    if not os.path.exists(str(day_stamp)):
        os.makedirs(str(day_stamp))

    ''' adjustable parameters '''
    ap_r = 5.  # radius of aperture for flux measurements

    ''' get list of image names to process'''
    filenames = glob(file + '*.fits')   
    filenames.sort()

    ''' get 2d shape of images, number of images, make list of unix times'''
    x_length, y_length, num_images, time_list = getSizeFITS(filenames) 
    print (datetime.datetime.now(), "Imported", num_images, "frames")
   
    '''check if enough images in folder'''
    minNumImages = 90         #3x kernel length
    if num_images < minNumImages:
        print (datetime.datetime.now(), "Insufficient length data cube, skipping...")
        return

    ''' load/create star positional data file (.npy)
    star position file format: x  |  y  | half light radius'''
    #TODO: add filename, unix time
    first_frame = importFramesFITS(filenames, 0, 1, bias)
    star_pos_file = str(day_stamp) + '/' + field_name + '_pos.npy'   #file to save positional data

    # if no positional data for current field, create it from first_frame
    if not os.path.exists(star_pos_file):
        print (datetime.datetime.now(), field_name, 'starfinding...',)
        star_find_results = tuple(initialFindFITS(first_frame))
        np.save(star_pos_file, star_find_results)
        print ('done')

    #load in initial star positions and radii
    initial_positions = np.load(star_pos_file, allow_pickle = True)   #change in python3, allow_pickle set to false by default
    radii = initial_positions[:,-1]                                   #make array of radii
    initial_positions = initial_positions[:,:-1]                      #remove radius column

    num_stars = len(initial_positions)      #number of stars in image
    print(datetime.datetime.now(), 'number of stars found: ', num_stars) 

    ''' centroid refinements and drift check '''
    drift = False                  # variable to check whether stars have drifted since last frame
    check_frames = num_images // evolution_frames      # number of images to check
    
    drift_pos = np.empty([check_frames, num_stars], dtype=(np.float64, 2))  #array for star positions for drift check
    GaussSigma = np.mean(radii * 2. / 2.35)  # calculate gaussian sigma for each star's light profile

    # refine centroids for drift measurements using first frame as a check
    drift_pos[0] = refineCentroid(first_frame, initial_positions, GaussSigma)
    for t in range(1, check_frames):
        drift_pos[t] = refineCentroid(importFramesFITS(filenames, t * evolution_frames, 1, bias), deepcopy(drift_pos[t - 1]), GaussSigma)

    # check drift rates
    driftTolerance = 1e-2
    x_drift, y_drift = averageDrift(drift_pos, check_frames, evolution_frames)  # drift rates per frame
    if abs(np.median(x_drift)) > driftTolerance or abs(np.median(y_drift)) > driftTolerance:
        drift = True
        #TODO: what is this section?
        #if abs(np.median(x_drift)) > 1 or abs(np.median(y_drift)) > 1:
        #    print datetime.datetime.now(), "Significant drift, skipping..."  # find how much drift is too much
        #    return

    ''' flux and time calculations with optional time evolution '''
    
    #image data (2d array with dimensions: # of images x # of stars)
    data = np.empty([num_images, num_stars], dtype=(np.float64, 4))
    bkg_first = np.median(first_frame) * np.pi * ap_r * ap_r          #background level in aperture area
    
    #get first image data from initial star positions
    data[0] = tuple(zip(initial_positions[:,0], 
                        initial_positions[:,1], 
                        (sep.sum_circle(first_frame, initial_positions[:,0], initial_positions[:,1], ap_r)[0] - bkg_first).tolist(), 
                        np.ones(np.shape(np.array(initial_positions))[0]) * time_list[0]))

    if drift:  # time evolve moving stars
        for t in range(1, num_images):
            data[t] = timeEvolveFITS(importFramesFITS(filenames, t, 1, bias), deepcopy(data[t - 1]), 
                                     x_drift, y_drift, ap_r, num_stars, x_length, y_length, time_list[t])
            
    else:  # if there is not significant drift, don't account for drift  in photometry
        for t in range(1, num_images):
            data[t] = timeEvolveFITSNoDrift(importFramesFITS(filenames, t, 1, bias), deepcopy(data[t - 1]), 
                                            ap_r, num_stars, x_length, y_length, time_list[t])

    # data is an array of shape: [frames, star_num, {0:star x, 1:star y, 2:star flux, 3:unix_time}]
    print (datetime.datetime.now(), 'Photometry done.')
   

    ''' Dip detection '''
    
    '''Parallel version
    cores = multiprocessing.cpu_count()  # determine number of CPUs for parallel processing
    
    #perform dip detection and for all stars
    results array: frame # of event (if found, -1 or -2 otherwise) | light curve for star
    results = np.array(Parallel(n_jobs=cores, backend='threading')(
        delayed(dipDetection)(data[:, index, 2], kernel, index) for index in range(0, num_stars)))
   '''
    
   
    #non parallel version (for easier debugging)
    results = []
    for index in range(0, num_stars):
        results.append(dipDetection(data[:, index, 2], kernel, index))
        
    results = np.array(results)

    event_frames = results[:,0]         #array of event frames (-1 or -2 if no event detected)
    light_curves = results[:,1]         #array of light curves (empty if no event detected)

    ''' data archival '''
    secondsToSave =  1    #number of seconds on either side of event to save 
    name_stamp = str(file.split('\\')[-2])   
    save_frames = event_frames[np.where(event_frames > 0)]  #frame numbers for each event to be saved
    save_chunk = int(round(secondsToSave / exposure_time))  #save certain num of frames on both sides of event
    save_curves = light_curves[np.where(event_frames > 0)]  #light curves for each star to be saved
    
    
    for t in save_frames:  # loop through each detected event
    
        star_coords = initial_positions[np.where(event_frames == t)[0][0]]
        print(datetime.datetime.now(), ' saving event in frame', t)
        
        star_all_flux = save_curves[np.where(save_frames == t)][0]  #total light curve for current occulted star
        
        #save results in a file
        #saved file format: 'results-field#-star#.txt'
        #first line: star coords
        #columns: fits filename and path | unix time |  star flux
        
        if t - save_chunk <= 0:  # if chunk includes lower data boundary, start at 0
        
            files_to_save = [filename for i, filename in enumerate(filenames) if i >= 0 and i < t + save_chunk]  #list of filenames to save
            star_save_flux = star_all_flux[np.where(np.in1d(filenames, files_to_save))[0]]                       #part of light curve to save
            
            with open("./" + str(day_stamp) + "/results-" + name_stamp + "-" + str(np.where(event_frames == t)[0][0]) + ".txt", 'w') as filehandle:
                filehandle.write('%f %f\n' %(star_coords[0], star_coords[1]))
                #loop through each frame to be saved
                for i in range(0, len(files_to_save)):  
                    filehandle.write('%s %f  %f\n' % (files_to_save[i], time_list[:t + save_chunk][i], star_save_flux[i]))
        

        else:  # if chunk does not include lower data boundary
        
            if t + save_chunk >= num_images:  # if chunk includes upper data boundary, stop at upper boundary
        
                files_to_save = [filename for i, filename in enumerate(filenames) if i >= t - save_chunk and i < num_images - t + save_chunk] #list of filenames to save
                star_save_flux = star_all_flux[np.where(np.in1d(filenames, files_to_save))[0]]                                                #part of light curve to save

                with open("./" + str(day_stamp) + "/results-" + name_stamp + "-" + str(np.where(event_frames == t)[0][0]) + ".txt", 'w') as filehandle:
                    filehandle.write('%f %f\n' %(star_coords[0], star_coords[1]))
                    
                    #loop through each frame to save
                    for i in range(0, len(files_to_save)): 
                        filehandle.write('%s %f %f\n' % (files_to_save[i], time_list[t - save_chunk:][i], star_save_flux[i]))

            else:  # if chunk does not include upper data boundary
            
                files_to_save = [filename for i, filename in enumerate(filenames) if i >= t - save_chunk and i < 1 + save_chunk * 2]   #list of filenames to save
                star_save_flux = star_all_flux[np.where(np.in1d(filenames, files_to_save))[0]]                        #part of light curve to save

                with open("./" + str(day_stamp) + "/results-" + name_stamp + "-" + str(np.where(event_frames == t)[0][0]) + ".txt", 'w') as filehandle:
                    filehandle.write('%f %f\n' %(star_coords[0], star_coords[1]))
                    
                    #loop through each frame to save
                    for i in range(0, len(files_to_save)): 
                        filehandle.write('%s %f %f\n' % (files_to_save[i], time_list[t - save_chunk:t + save_chunk][i], star_save_flux[i]))
                   

    ''' printout statements'''
    print (datetime.datetime.now(), "Rejected Stars: ", round(((num_stars - len(save_frames)) / num_stars)*100, 2), "%")
    print (datetime.datetime.now(), "Total stars in field:", num_stars)
    print (datetime.datetime.now(), "Candidate events in this minute:", len(save_frames))
    print (datetime.datetime.now(), "Closing:", file)
    print ("\n")

"""---------------------------------CODE STARTS HERE-------------------------------------------"""

'''get filepaths'''
directory = '.\\data\\'                  #directory that contains .fits image files for 1 night
folder_list = glob(directory + '*\\')    #each folder has 1 minute of data (~2400 images)
field = '25ms'
print ('folders', folder_list)
     
'''get median bias image to subtract from all frames'''
biasFilepath = '../test_data/bias/'
NumBiasImages = 10                             #number of bias images to combine in median bias image
bias = getBias(biasFilepath, NumBiasImages)    #take median of NumBiasImages to use as bias

''' prepare RickerWavelet/Mexican hat kernel to convolve with light curves'''
exposure_time = 0.025    # exposure length in seconds
expected_length = 0.15   # related to the characteristic scale length, length of signal to boost in convolution, may need tweaking/optimizing
refresh_rate = 2.        # number of seconds (as float) between centroid refinements

kernel_frames = int(round(expected_length / exposure_time))   # width of kernel
ricker_kernel = RickerWavelet1DKernel(kernel_frames)          # generate kernel
evolution_frames = int(round(refresh_rate / exposure_time))   # number of frames between centroid refinements
    
''''run pipeline for each minute of data'''
for f in range(0, len(folder_list)):
    print('running on... ', folder_list[f])
    processImages(folder_list[f], field, bias, ricker_kernel, exposure_time, evolution_frames)
    gc.collect()

