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

def getKernelParams(kernel_i):
    '''get parameters for best fitting kernel from the output .txt file'''
    
    param_filename = base_path.joinpath('params_kernels_031522.txt')
    kernel_params = pd.read_csv(param_filename, delim_whitespace = True)
    
    return kernel_params.iloc[kernel_i]


def plotKernel(lightcurve, kernel, start_i, eventFrame, starNum, directory, params):
    '''make plot of the convolution of lightcurve and best fitting kernel'''
    
    trimmedcurve = lightcurve[start_i:]
    
    eventFrame = eventFrame - start_i

    fig, ax = plt.subplots()
    
    textstr = '\n'.join((
    'Object R [m] = %.0f' % (params[2], ),
    'Star d [mas] = %.2f' % (params[3], ),
    'b [m] = %.0f' % (params[4], ),
    'Shift [frames] = %.2f' % (params[5], )))
    
    ax.plot(trimmedcurve)
    ax.plot(kernel, label = 'Best fitting kernel')
    ax.vlines(eventFrame, min(trimmedcurve), max(trimmedcurve), color = 'red', label = 'Event middle')
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(1.02, 0.95, textstr, transform=ax.transAxes, fontsize=12,
    verticalalignment='top', bbox=props)

    ax.set_title('Normalized light curve matched with kernel ' + str(int(params[0])))
    ax.set_xlabel('Frame number since beginning of convolution')
    ax.set_ylabel('Flux normalized by median')
    ax.legend()
    
    #plt.show()
    plt.savefig(directory.joinpath('star' + starNum + 'matched.png'), bbox_inches = 'tight')
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


def kernelDetection(fluxProfile, fluxTimes, dipFrame, kernels, num, directory):
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
    


    """ Dip detection"""
    # astropy v2.0+ changed convolve_fft quite a bit... see documentation, for now normalize_kernel=False
#    conv = convolve_fft(trunc_profile, kernel, normalize_kernel=False)
  #  minLoc = dipFrame   #index of minimum value of convolution
  #  minVal = conv[dipFrame]   #minimum value of convolution
    


    # if dip not as large, try to match a kernel
  #  KernelLength = 30
    Gain = 16.5     #For high gain images: RAB - 031722
    #Gain = 2.8     #For low gain images: RAB - 031722
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
    

    
    #get curve with dip removed (called 'background')
    half_dip_width = 10     #approximate length of a dip event [frames]
    
    #set the beginning of the dip region
    if dipFrame - half_dip_width > 0:    
        dip_start = dipFrame - half_dip_width
    
    #if beginning should be the beginning of the light curve
    else:
        dip_start = 0
        
    #set the end of the dip zone
    if dipFrame + half_dip_width < len(fluxProfile):
        dip_end = dipFrame + half_dip_width
    
    #if end should be the end of the light curve
    else:
        dip_end = len(fluxProfile) - 1
    
    #copy the original flux profile and list of times
    background = fluxProfile[:]
    background_time = fluxTimes[:]
    
    #remove the dip region from the background flux and time arrays
    del background[dip_start:dip_end]
    del background_time[dip_start:dip_end]
    
    #fit line to background portion
    bkgd_fitLine = np.poly1d(np.polyfit(background_time, background, 1))
    
    #divide original flux profile by fitted line
    fluxProfile =  fluxProfile/bkgd_fitLine(fluxTimes)

    
  #  edgeBuffer = 10     #how many elements from beginning/end of the convolution to exclude
  #  bkgZone = conv[edgeBuffer: -edgeBuffer]        #background
    
    #bkgZone = conv  
    #dipdetection = 3.75  #dip detection threshold 
    

 #   if minVal < np.mean(bkgZone) - dipdetection * np.std(bkgZone):  # dip detection threshold
        
    """ Kernel matching"""

    StatMin = np.inf    #statistical minimum used for finding best matching kernel
        
    #loop through each kernel in set, check to see if it's a better fit
    for ind in range(0, len(kernels)):
            

            
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
    
    #unnecessary if we're sure kernels span 20-40% dip RAB Mar 15 2022
    for P in eventSigmaP:
        thresh += (abs(sigmaNorm)) / (abs(P))  # kernel match threshold - LHS of Eq. 2 in Pass 2018
            
    #check if dip is significant to call a candidate event
    if StatMin < thresh:      #Eq. 2 in Pass 2018
            
      #  critFrame = np.where(fluxProfile == fluxProfile[dipFrame])[0]    #time of event
     #   critFrame = dipFrame    
     #   if len(critFrame) > 1:
     #       raise ValueError
        
        params = getKernelParams(active_kernel)
        
        plotKernel(fluxProfile, kernels[active_kernel], MatchStart, dipFrame, num, directory, params)
        
        return active_kernel, StatMin, MatchStart, params  # returns location in original time series where dip occurs
        
    else:
        print('Event in star %s did not pass threshold' % num)
        
        params = getKernelParams(active_kernel)
        
        plotKernel(fluxProfile, kernels[active_kernel], MatchStart, dipFrame, num, directory, params)
        
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
                
            elif i == 9:
                event_type = line.split(':')[1].split('\n')[0].strip(" ")
                
        #reset event frame to match index of the file
        event_frame = event_frame - first_frame

    return starData, event_frame, star_x, star_y, event_time, event_type
    
'''-----------code starts here -----------------------'''

runPar = False          #True if you want to run directories in parallel
telescope = 'Red'       #identifier for telescope
gain = 'high'           #gain level for .rcd files ('low' or 'high')
obs_date = datetime.date(2021, 8, 4)    #date observations 
process_date = datetime.date(2022, 3, 17)
base_path = pathlib.Path('/', 'home', 'rbrown', 'Documents', 'Colibri')  #path to main directory


if __name__ == '__main__':
    
    ''' prepare RickerWavelet/Mexican hat kernel to convolve with light curves'''
    
    exposure_time = 0.025    # exposure length in seconds
    expected_length = 0.15   # TODO: come back to this - related to the characteristic scale length, length of signal to boost in convolution, may need tweaking/optimizing

    kernel_frames = int(round(expected_length / exposure_time))   # width of kernel
    ricker_kernel = RickerWavelet1DKernel(kernel_frames)          # generate kernel

    kernel_set = np.loadtxt(base_path.joinpath('kernelsMar2022.txt'))
    
    #check if kernel has a detectable dip - moved out of dipdetection function RAB 031722
    noise = 0.8   #minimum kernel depth threshold RAB Mar 15 2022- detector noise levels (??) TODO: change - will depend on high/low
    for i in range(len(kernel_set)):
        
        #check if deepest dip in a kernel is greater than expected background noise 
        if min(kernel_set[i]) > noise:
            #remove kernel with dips less than expected background noise
            print('Removing kernel ', i)
            del(kernel_set[i])
            continue
    
   # refresh_rate = 2

    #create RickerWavelet/Mexican Hat kernel to convolve with light profil
   # kernel_frames = int(round(expected_length / exposure_time)) #width of kernel
   # ricker_kernel = RickerWavelet1DKernel(kernel_frames)       #generate kernel
    
    #what is this?
    #evolution_frames = int(round(refresh_rate / exposure_time))  # determines the number of frames in X seconds of data


    '''get filepaths to results directory'''
    
    #directory containing detection .txt files
    archive_dir = base_path.joinpath('ColibriArchive', telescope, str(process_date))
    
    #list of filepaths to .txt detection files
    detect_files = [f for f in archive_dir.iterdir() if 'det' in f.name]
    
    
    '''loop through each file'''
    
    diff_results = []
    geo_results = []  
    
    for filepath in detect_files:
        
        #number id of occulted star
        star_num = filepath.name.split('star')[1].split('_')[0]
        
        #read in file data
        starData, event_frame, star_x, star_y, event_time, event_type = readFile(filepath)
        
        lightcurve_looker.plot_event(archive_dir, starData, event_frame, star_num, [star_x, star_y], event_type)
        
        if event_type == 'diffraction':
            diff_results.append((star_num, star_x, star_y, event_time, kernelDetection(list(starData['flux']), list(starData['time']), event_frame, kernel_set, star_num, archive_dir)))
       
        if event_type == 'geometric':
            geo_results.append((star_num, star_x, star_y, event_time, kernelDetection(list(starData['flux']), list(starData['time']), event_frame, kernel_set, star_num, archive_dir)))

    #save list of best matched kernels in a .txt file
        
    diff_save_file = archive_dir.joinpath(str(obs_date) + '_' + telescope + '_diffraction_kernelMatches.txt')
    geo_save_file = archive_dir.joinpath(str(obs_date) + '_' + telescope + '_geometric_kernelMatches.txt')

    with open(diff_save_file, 'w') as file:
        
        file.write('#starNumber  starX  starY  eventTime  kernelIndex  Chi2  startLocation  ObjectD   StellarD   b    shift\n')
            
        for line in diff_results:
            
            #events that didn't pass kernel matching
            if line[4] == -1:
                continue
            
            file.write('%s %f %f %s %i %f %i %f %f %f %f\n' %(line[0], line[1], line[2], line[3], 
                                                  line[4][0], line[4][1], line[4][2], 
                                                  line[4][3][2], line[4][3][3], line[4][3][4], line[4][3][5]))
    
    with open(geo_save_file, 'w') as file:
        
        file.write('#starNumber  starX  starY  eventTime  kernelIndex  Chi2  startLocation  ObjectD   StellarD   b    shift\n')
        
        for line in geo_results:
            
            #events that didn't pass kernel matching
            if line[4] == -1:
                continue
            
            file.write('%s %f %f %s %i %f %i %f %f %f %f\n' %(line[0], line[1], line[2], line[3], 
                                                  line[4][0], line[4][1], line[4][2], 
                                                  line[4][3][2], line[4][3][3], line[4][3][4], line[4][3][5]))




