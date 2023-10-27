#importing necessary modules
from pylab import *
from importlib import reload
import cv2
import numpy as np
from scipy import interpolate
from scipy.signal import argrelextrema
from scipy import optimize
from scipy.interpolate import interp1d


''' read image from a file in the format used by fargo
'''
def open_img(filename, ntheta=384, nr=128, image_rmin = 0.4, image_rmax=2.5, ylog=False):
    rho = fromfile(filename).reshape(nr,ntheta)
    if ylog:
        rho = logtolin(rho, image_rmin, image_rmax, nr)
    return rho


''' Warps disk image from polar to cartesian coordinates
    returns the image as narray.
'''
def warp_image_rolltodisk(data, ntheta=384, nr=128, image_rmin = 0.4, image_rmax=2.5, target_rmax=2.5, target_image_size=(200,200), ylog=False):
    
    #reading image
    if isinstance(data, str):
        try:
            rho = open_img(data, ntheta, nr, image_rmin=image_rmin, image_rmax=image_rmax, ylog=ylog)
        except:
            return None
    else:
        if isinstance(data, np.ndarray):
            rho = data
    
    #padding to standardize dimensions
    #TODO: outer padding may be done continuing the initial density profile. 
    #now all the padding are set to 0 
    inner_pad = int((nr*image_rmin)/(image_rmax-image_rmin))
    outer_pad = int((nr*(target_rmax-image_rmax))/(image_rmax-image_rmin))
    if outer_pad<0:
    	rho = rho[:outer_pad,:]
    	outer_pad=0
    
    rho2 = np.pad(rho, ((inner_pad, outer_pad),(0,0)),'constant', constant_values=(0,)).transpose()
    
    #warp image
    reload(cv2)
    rhoxy = cv2.warpPolar(rho2, target_image_size, (target_image_size[0]/2, target_image_size[1]/2), target_image_size[0]/2+1,cv2.WARP_INVERSE_MAP+cv2.WARP_FILL_OUTLIERS )
    
    return rhoxy


''' returns an azimuthally averaged profile
of density map given in polar coordinates
'''
def get_profile(data, ntheta=384, nr=128,image_rmin = 0.4, image_rmax=2.5, ylog=False):
    if isinstance(data, str):
        dat = open_img(data, ntheta, nr, image_rmin = image_rmin, image_rmax=image_rmax, ylog=ylog)
    else:
        if isinstance(data, np.ndarray):
            dat = data
            
            
    return np.mean(dat, axis=1)


''' returns a tuple (r, f(r)) containing the
    difference between the density profile and the 
    initial profile, normalized by rho0
'''
def get_normalized_profile(data, rho0, rhoslope, ntheta=384, nr=128, rmin=0.4, rmax=2.5, ylog=False):
    
    rho_azim_averaged_norm = get_profile(data, ntheta, nr, image_rmin = rmin, image_rmax=rmax, ylog=ylog)/rho0
    x = np.linspace(rmin, rmax, nr)
    rho0_prof = x**(-rhoslope)
    return (x, rho_azim_averaged_norm - rho0_prof)
    

''' analyze density profile and returns 
    features of found gaps
'''
import numpy as np
from scipy.signal import argrelextrema

def get_gap_width_fwhm(r_grid, prof, return_margins=False):
    
    try:
        cent_indx_1 = np.argmin(np.abs(r_grid-1))
        cent_indx = argrelextrema(prof, np.greater)[0][np.argmin(np.abs(argrelextrema(prof, np.greater)-cent_indx_1))]

        min_left = np.sort(argrelextrema(prof[:cent_indx], np.less))[0][::-1]
        max_left = np.sort(argrelextrema(prof[:cent_indx], np.greater))[0][::-1]

        min_right = np.sort(argrelextrema(prof[cent_indx:], np.less))[0] + cent_indx
        max_right = np.sort(argrelextrema(prof[cent_indx:], np.greater))[0] + cent_indx

        indx_max_left = max_left[max_left < min_left[0]][0]
        indx_min_left = min_left[0]
        mean_int_left = ( prof[indx_max_left]-prof[indx_min_left])/2 + prof[indx_min_left]
        near_tomean_indx_left = np.sort((np.argsort(np.abs(prof[indx_max_left:indx_min_left]-mean_int_left))+indx_max_left)[:2])
        left_margin = (mean_int_left-prof[near_tomean_indx_left[0]])*(r_grid[near_tomean_indx_left[1]]-r_grid[near_tomean_indx_left[0]])/(prof[near_tomean_indx_left[1]]-prof[near_tomean_indx_left[0]]) + r_grid[near_tomean_indx_left[0]]

        indx_max_right = max_right[max_right > min_right[0]][0]
        indx_min_right = min_right[0]
        mean_int_right = ( prof[indx_max_right]-prof[indx_min_right])/2 + prof[indx_min_right]
        near_tomean_indx_right = np.sort((np.argsort(np.abs(prof[indx_min_right:indx_max_right]-mean_int_right))+indx_min_right)[:2])
        right_margin = (mean_int_right-prof[near_tomean_indx_right[0]])*(r_grid[near_tomean_indx_right[1]]-r_grid[near_tomean_indx_right[0]])/(prof[near_tomean_indx_right[1]]-prof[near_tomean_indx_right[0]]) + r_grid[near_tomean_indx_right[0]]

        gap_width = right_margin-left_margin

        if return_margins:
            return (left_margin, right_margin, gap_width)
        return gap_width

    except BaseException:
        return None


def get_gap_width_devFromBaseState(r_grid, proff, Sigma0, SigmaSlope, return_margins=False):
    
    #obtaining normalized profile
    prof = proff/Sigma0 - r_grid**(-SigmaSlope)
    prof_interp = interp1d(r_grid, prof)

    #search zeros
    signs = prof/np.abs(prof)
    zeros = (signs[:-1]*signs[1:])==-1
    zz = (r_grid[:-1][zeros]+r_grid[1:][zeros])/2
    
    #compute gap
    #take the 4 zeros closer to r=1.0
    if np.min(zz)>1:
        left_margin = 0.4
        right_margin = np.min(zz)
    else:
        if len(zz)>=4:
            zz4 = zz[np.argsort(np.abs(zz-1))][:4]
            #check if there is the horse saddle 
            if prof_interp(1) > 0:
                left_margin = min(zz4[2], zz4[3])
                right_margin = max(zz4[2], zz4[3])
                #gw = zz4[3]-zz4[0]
            else:
                left_margin = min(zz4[0], zz4[1])
                right_margin = max(zz4[0], zz4[1])
                #gw = zz4[2]-zz4[1]
        else:
            if len(zz)>=2:
                zz2 = np.sort(zz[np.argsort(np.abs(zz-1))][:2])
                left_margin = zz2[0]
                right_margin = zz2[1]
                #gw = zz2[1] - zz2[0]<
            else:
                return (None, None, None)

    gap_width = right_margin-left_margin
    if return_margins:
        return (left_margin, right_margin, gap_width)
    return gap_width


def logtolin(image, ymin, ymax, ny):
    y_new_axis = np.linspace(ymin, ymax, ny)
    y_old_axis = np.logspace(np.log10(ymin), np.log10(ymax), ny)
    old_image = image
    new_image = old_image.copy()

    for j in range(old_image.shape[1]):
        f = interpolate.interp1d(y_old_axis, old_image[:, j])
        i = range(len(y_new_axis))
        new_image[i, j] = f(y_new_axis)
            
    return new_image


def rotate(image, center, gap_inner_edge_a, gap_inner_edge_b, new_size):
    pts1 = np.float32([center_old, gap_inner_edge_a, gap_inner_edge_b])
    pts2 = np.float32([[new_size/2, new_size/2],[new_size/2+new_size/5, new_size/2],[new_size/2,new_size/2+new_size/5]])
    M = cv.getAffineTransform(pts1,pts2)
    dst = cv.warpAffine(image,M,(new_size, new_size))
    return dst
