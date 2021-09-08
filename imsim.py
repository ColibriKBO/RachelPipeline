
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter, rotate
from scipy import ndimage, misc, stats, interpolate
from astropy.io import fits
from astropy.time import Time

def rebin(a, shape):
	sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
	return a.reshape(sh).sum(-1).sum(1)

def simulateImage(stars, xres, yres):
	interpolation = 2
	padding = 10

	imarr = np.zeros([(xres+2*padding)*interpolation,(yres+2*padding)*interpolation])

	num_stars = len(stars)
	xstar = (np.round(stars[:,1]+padding)*interpolation)
	ystar = (np.round(stars[:,2]+padding)*interpolation)
	mstar = np.round(stars[:,7])

	bstar = stars[:,11]
	bmean = np.mean(bstar)

	for i in range(num_stars):
		imarr[int(xstar[i])-1, int(ystar[i])-1] = int(mstar[i])

	stararr = imarr.copy()

	###############################################
	# Convolve with a Gaussian and plot w/o noise #
	###############################################
	stararr = gaussian_filter(stararr, sigma=0.75*interpolation)

	# plt.figure(figsize=(9,6))
	# plt.imshow(stararr, interpolation='nearest', origin='lower', vmin=0, vmax=20)
	# plt.tight_layout()
	# plt.show()

	##############################################
	# Resample the image array and plot w/ noise #
	##############################################
	stararr = stararr[padding*interpolation:-1*padding*interpolation, padding*interpolation:-1*padding*interpolation]

	rebinarr = rebin(stararr, (xres,yres))

	noisearr = np.random.normal(bmean,np.std(bstar),(xres,yres))
	finarr = np.add(rebinarr, noisearr)
	np.clip(finarr, 0, 2**16, finarr) # Clip values back to 16 bits
	finarr = finarr.astype(np.uint16)

	#####
	# May or may not need the following. If we do, we need to determine the intercept
	# I've done this for a single image and, oddly, the slope is less than 1
	#####
	# Mslope = 0.91
	# Mintercept = -13.9
	# Mg = 12
	# flux = 10**((Mslope*Mg+Mintercept)/-2.5)
	# print('A star with a magnitude of %s has %s counts above the background of %s +/- %s' % (Mg, round(flux,2), round(bmean,2), round(np.std(noisearr),2)))

	#####
	# Plotting section
	#####
	# plt.figure(figsize=(9,6))
	# plt.imshow(finarr, interpolation='nearest', origin='lower', vmin=np.mean(noisearr)-2*np.std(noisearr), vmax=70)
	# plt.tight_layout()
	# plt.show()

	return finarr

def writeRCD(image):
	print('Writing RCD file...')

def writeFITS(imagelist, file, lat, lon, exptime, timestamp):
	
	hdu = fits.PrimaryHDU(imagelist)
	hdr = hdu.header
	hdr.set('exptime', exptime)
	hdr.set('DATE-OBS', str(timestamp))
	hdr.set('SITELAT', lat)
	hdr.set('SITELONG', lon)
	hdu.writeto(file, overwrite=True)


if __name__ == "__main__":

	xres = 2048
	yres = 2048
	exptime = 0.025

	duration = 10 # length of image sequence in seconds
	numexps = round(duration/exptime)
	numexps = 2 # for testing

	# Load star catalog
	stars = np.loadtxt('./20210804-Field1.cat', skiprows=14)

	print(stars)

	nowtime = Time.now()
	nowtime.format = 'fits'

	for i in range(numexps):
		simarr = simulateImage(stars, xres, yres)

		currenttime = nowtime.jd + i*exptime/(3600*24)
		timestamp = Time(currenttime, format='jd')
		timestamp.format = 'fits'

		writeFITS(simarr,'image-' + str(i) + '.fts', -81.3, 43.2, exptime, timestamp)
		print('Save image %s to image-%s.fits...' % (i+1, i))



