"""
Created March 8, 2022 by Rachel Brown

Update: March 10, 2022 by Rachel Brown

-secondary Colibri pipeline for matching identified events to a kernel
"""


import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft, RickerWavelet1DKernel
from astropy.time import Time
from joblib import delayed, Parallel
from copy import deepcopy
import multiprocessing
import time
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import datetime
import lightcurve_looker


def plotKernel(lightcurve, kernel, start_i, eventFrame):
    '''make plot of the convolution of lightcurve and best fitting kernel'''
    
    trimmedcurve = lightcurve[start_i:]
    
    eventFrame = eventFrame - start_i

    plt.plot(trimmedcurve)
    plt.plot(kernel, label = 'Best fitting kernel')
    plt.vlines(eventFrame, min(trimmedcurve), max(trimmedcurve), color = 'red', label = 'Event middle')
    plt.title('Normalized light curve')
    plt.xlabel('Frame number since beginning of convolution')
    plt.ylabel('Flux normalized by median')
    plt.legend()
    plt.show()
    plt.close()
    
    # plt.plot(lightcurve)
    # plt.plot(shiftedkernel, label = 'Best fitting kernel')
    # plt.vlines(eventFrame, min(trimmedcurve), max(trimmedcurve), color = 'red', label = 'Event middle')
    # plt.title('Normalized light curve (trimmed)')
    # plt.xlabel('Frame number since beginning of convolution')
    # plt.ylabel('Flux normalized by median')
    # plt.legend()
    # plt.show()
    # plt.close()
    
    # plt.plot(kernel)
    # plt.vlines(eventFrame, min(kernel), max(kernel), color = 'red')
    # plt.xlabel('frame number')
    # plt.ylabel('Intensity')
    # plt.title('Matched kernel')
    # plt.show()
    # plt.close()
    
    return
    


def diffMatch(template, data, sigmaP):
    """ Calculates the best start position (minX) and the minimization constant associated with this match (minChi) """

    minChi = np.inf       #statistical minimum variable
    minX = np.inf         #starting index variable
    
    matchTolerance = 3    #number of frames difference to tolerate for match (~150 ms)

    #loop through possible starting values for the best match dip
    for dipStart in range((int(len(data) / 2) - int(len(template) / 2)) - matchTolerance, int(len(data) / 2) - int(len(template) / 2) + matchTolerance):
        
        chiSquare = 0
        
        #loop through each flux value in template and compare to data
        for val in range(0, len(template)):
            chiSquare += (abs(template[val] - data[dipStart + val])) / abs(sigmaP[dipStart + val])  #chi^2 expression
            
        #if the curren chi^2 value is smaller than previous, set new stat. minimum and get new index
        if chiSquare < minChi:
            minChi = chiSquare
            minX = dipStart
            
    return minChi, minX


def kernelDetection(fluxProfile, dipFrame, kernel, kernels, num):
    """ Detects dimming using Mexican Hat kernel for dip detection and set of Fresnel kernels for kernel matching """

   # light_curve = fluxProfile
    med = np.median(fluxProfile)         

   # trunc_profile = np.where(light_curve < 0, 0, light_curve)  
   ## trunc_profile = light_curve
    
   # FramesperMin = 2400      
   # NumBkgndElements = 200

    #normalized sigma corresponding to RHS of Eq. 2 in Pass et al. 2018
    #sigmaNorm = np.std(trunc_profile[(FramesperMin-NumBkgndElements):FramesperMin]) / np.median(trunc_profile[(FramesperMin-NumBkgndElements):FramesperMin])
    #really not sure how this matches what is in Emily's paper.....
    sigmaNorm = np.std(fluxProfile) / np.median(fluxProfile)
    
 #   plt.plot(trunc_profile)
 #   plt.title('truncated profile')
 #   plt.vlines(dipFrame, min(trunc_profile), max(trunc_profile), color = 'red')
 #   plt.show()
    
       
   # plt.plot(kernel)
   # plt.title('kernel for initial convolution')
#    plt.vlines(dipFrame, min(kernel), max(kernel), color = 'red')
   # plt.show()

    """ Dip detection"""
    # astropy v2.0+ changed convolve_fft quite a bit... see documentation, for now normalize_kernel=False
#    conv = convolve_fft(trunc_profile, kernel, normalize_kernel=False)
  #  minLoc = dipFrame   #index of minimum value of convolution
  #  minVal = conv[dipFrame]   #minimum value of convolution
    
 
    
 #   plt.plot(conv)
 #   plt.title('convolved light curve')
 #   plt.vlines(minLoc, min(conv), max(conv), color = 'red')
 #   plt.show()
    

    # if dip not as large, try to match a kernel
  #  KernelLength = 30
    Gain = 100.
  #  NumofBufferElementstoIgnore = 100
    
    #check if minimum is at least one kernel length from edge
 #   if KernelLength <= minLoc < len(trunc_profile) - KernelLength:
        
        #get median for area around the minimum
        #don't need because we've already trimmed the light curve
    #      med = np.median(
  #          np.concatenate((trunc_profile[minLoc - NumofBufferElementstoIgnore:minLoc - KernelLength], trunc_profile[minLoc + KernelLength:minLoc + NumofBufferElementstoIgnore])))
        
        #trim the light curve to include the minimum value +/- one kernel length
  #      trunc_profile = trunc_profile[(minLoc - KernelLength):(minLoc + KernelLength)]
        
        #normalized fractional uncertainty
    sigmaP = ((np.sqrt(np.abs(fluxProfile) / np.median(fluxProfile) / Gain)) * Gain)   #Eq 7 in Pass 2018 (Poisson)
    sigmaP = np.where(sigmaP == 0, 0.01, sigmaP)
    fluxProfile /= med
    
  #  edgeBuffer = 10     #how many elements from beginning/end of the convolution to exclude
  #  bkgZone = conv[edgeBuffer: -edgeBuffer]        #background
    
    #bkgZone = conv  
    #dipdetection = 3.75  #dip detection threshold 
    noise = 0.8   #detector noise levels (??)

 #   if minVal < np.mean(bkgZone) - dipdetection * np.std(bkgZone):  # dip detection threshold
        
    """ Kernel matching"""

    StatMin = np.inf    #statistical minimum used for finding best matching kernel
        
    #loop through each kernel in set, check to see if it's a better fit
    for ind in range(0, len(kernels)):
            
        #check if deepest dip in a kernel is greater than expected background noise 
        if min(kernels[ind]) > noise:
            continue
            
        #get a statistical minimum and location of the starting point for the kernel match
        new_StatMin, loc = diffMatch(kernels[ind], fluxProfile, sigmaP)
            
        #checks if current kernel is a better match
        if new_StatMin < StatMin:
            active_kernel = ind
            MatchStart = loc    #index of best starting position for kernel matching
            StatMin = new_StatMin
                
    #list of Poisson uncertainty values for the event
    eventSigmaP = sigmaP[MatchStart:MatchStart + len(kernels[active_kernel])]   
    thresh = 0   #starting for sum in LHS of Eq. 2 in Pass 2018
        
    for P in eventSigmaP:
        thresh += (abs(sigmaNorm)) / (abs(P))  # kernel match threshold - LHS of Eq. 2 in Pass 2018
            
    #check if dip is significant to call a candidate event
    if StatMin < thresh:      #Eq. 2 in Pass 2018
            
      #  critFrame = np.where(fluxProfile == fluxProfile[dipFrame])[0]    #time of event
     #   critFrame = dipFrame    
     #   if len(critFrame) > 1:
     #       raise ValueError
        
            
        plotKernel(fluxProfile, kernels[active_kernel], MatchStart, dipFrame)
        
        return active_kernel, StatMin, MatchStart  # returns location in original time series where dip occurs
        
    else:
        return -1  # reject events that do not pass kernel matching
    
    

def readFile(filepath):
    '''read in a .txt detection file and get information from it'''
    
    #make dataframe containing image name, time, and star flux value for the star
    starData = pd.read_csv(filepath, delim_whitespace = True, 
           names = ['filename', 'time', 'flux'], comment = '#')

    first_frame = int(starData['filename'][0].split('_')[-1].split('.')[0])
    
    #get header info from file
    with filepath.open() as f:
        
        #loop through each line of the file
        for i, line in enumerate(f):
            

            #get event frame number
            if i == 4:
                event_frame = int(line.split('_')[-1].split('.')[0])

            #get star coords
            elif i == 5:
                star_coords = line.split(':')[1].split(' ')[1:3]
                star_x = float(star_coords[0])
                star_y = float(star_coords[1])
            
            #get event time
            elif i == 6:
                event_time = line.split('T')[2].split('\n')[0]
                
        #reset event frame to match index of the file
        event_frame = event_frame - first_frame

    return starData, event_frame, star_x, star_y, event_time
    
'''-----------code starts here -----------------------'''

runPar = False          #True if you want to run directories in parallel
telescope = 'Red'       #identifier for telescope
gain = 'high'           #gain level for .rcd files ('low' or 'high')
obs_date = datetime.date(2021, 8, 4)    #date observations 
process_date = datetime.date(2022, 3, 11)
base_path = pathlib.Path('/', 'home', 'rbrown', 'Documents', 'Colibri')  #path to main directory


if __name__ == '__main__':
    
    ''' prepare RickerWavelet/Mexican hat kernel to convolve with light curves'''
    
    exposure_time = 0.025    # exposure length in seconds
    expected_length = 0.15   # TODO: come back to this - related to the characteristic scale length, length of signal to boost in convolution, may need tweaking/optimizing

    kernel_frames = int(round(expected_length / exposure_time))   # width of kernel
    ricker_kernel = RickerWavelet1DKernel(kernel_frames)          # generate kernel

    kernel_set = np.loadtxt(base_path.joinpath('kernelsMar2022.txt'))
    
    refresh_rate = 2

    #create RickerWavelet/Mexican Hat kernel to convolve with light profil
    kernel_frames = int(round(expected_length / exposure_time)) #width of kernel
    ricker_kernel = RickerWavelet1DKernel(kernel_frames)       #generate kernel
    
    #what is this?
    evolution_frames = int(round(refresh_rate / exposure_time))  # determines the number of frames in X seconds of data


    '''get filepaths to results directory'''
    
    #directory containing detection .txt files
    archive_dir = base_path.joinpath('ColibriArchive', telescope, str(process_date))
    
    #list of filepaths to .txt detection files
    detect_files = [f for f in archive_dir.iterdir() if 'det' in f.name]
    
    
    '''loop through each file'''
    
    results = []
    
    for filepath in detect_files:
        
        #number id of occulted star
        star_num = filepath.name.split('star')[1].split('_')[0]
        
        #read in file data
        starData, event_frame, star_x, star_y, event_time = readFile(filepath)
        
        lightcurve_looker.plot_event(archive_dir, starData, event_frame, star_num, [star_x, star_y])
        results.append((star_num, star_x, star_y, event_time, kernelDetection(list(starData['flux']), event_frame, ricker_kernel, kernel_set, star_num)))
       
        #save list of best matched kernels in a .txt file
        
    save_file = archive_dir.joinpath(str(obs_date) + '_' + telescope + '_kernelMatches.txt')
        
    with open(save_file, 'w') as file:
        
        file.write('#starNumber    starX     starY     eventTime      kernelIndex      Chi2       startLocation\n')
            
        for line in results:
            
            file.write('%s %f %f %s %i %f %i\n' %(line[0], line[1], line[2], line[3], line[4][0], line[4][1], line[4][2]))
            





