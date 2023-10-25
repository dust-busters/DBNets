#importing necessary modules
from pylab import *
from importlib import reload
import cv2
import numpy as np
from scipy import interpolate
    
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
    
    rho2 = np.pad(rho, ((inner_pad, outer_pad),(0,0)),'constant', constant_values=(0,)).transpose()
    
    #warp image
    reload(cv2)
    rhoxy = cv2.warpPolar(rho2, target_image_size, (target_image_size[0]/2, target_image_size[1]/2), target_image_size[0]/2+1,cv2.WARP_INVERSE_MAP+cv2.WARP_FILL_OUTLIERS )
    
    return rhoxy


''' returns an azimuthally averaged profile
of density map given in polar coordinates
'''
def get_profile(data, ntheta=384, nr=128):
    if isinstance(data, str):
        dat = open_img(data, ntheta, nr)
    else:
        if isinstance(data, np.ndarray):
            dat = data
    return np.mean(dat, axis=1)


''' returns a tuple (r, f(r)) containing the
    difference between the density profile and the 
    initial profile, normalized by rho0
'''
def get_normalized_profile(data, rho0, rhoslope, ntheta=384, nr=128, rmin=0.4, rmax=2.5):
    
    rho_azim_averaged_norm = get_profile(data, ntheta, nr)/rho0
    x = np.linspace(rmin, rmax, nr)
    rho0_prof = x**(-rhoslope)
    return (x, rho_azim_averaged_norm - rho0_prof)
    

''' analyze density profile and returns 
    features of found gaps
'''
def get_gap_features(x, y, rmin=0.4, rmax=2.5, nr=128):
    
    step = (rmax-rmin)/(nr-1)
    i_1 = int(0.6/step)

    sign_1 = y[i_1]/np.abs(y[i_1])
    n_gap = 0
    
    if sign_1>0:
        
        n_gap = 2
        #in this case there are usually two gapsdef logtolin(image, ymin, ymax, ny):
        #searching for the gap on the right
        i=i_1
        right_min = y[i]
        i_min_right = i
        while y[i]>0:
            if y[i]<right_min:
                right_min = y[i]
                i_min_right = i
            i+=1
        right_gap_x1 = (x[i]+x[i-1])*0.5
        while y[i]<0:
            if y[i]<right_min:
                right_min = y[i]
                i_min_right = i
            i+=1
        right_gap_x2 = (x[i]+x[i-1])*0.5


        #searching the gap on the left
        i=i_1
        left_min = y[i]
        i_min_left = i
        while y[i]>0:
            if y[i]<left_min:
                left_min = y[i]
                i_min_left = i
            i-=1
        left_gap_x1 = (x[i]+x[i+1])*0.5
        while y[i]<0:
            if y[i]<left_min:
                left_min = y[i]
                i_min_left = i
            i-=1
        left_gap_x2 = (x[i]+x[i+1])*0.5

        #computing gap properties
        gap_width_2 = right_gap_x2-right_gap_x1
        gap_width_1 = left_gap_x1-left_gap_x2
        gap_depth_2 = np.abs(right_min)
        gap_depth_1 = np.abs(left_min)
        
    else:
        #in this case there is usually only one gap
        i=i_1
        
        while y[i]<0:
            if y[i]<gmin:
                gmin = y[i]
            i+=1
        gap_x2 = (x[i]+x[i-1])*0.5
        while y[i]<0:
            if y[i]<gmin:
                gmin = y[i]
            i-=1
        gap_x1 = (x[i]+x[i+1])*0.5

        #computing gap properties
        gap_width_2 = 0
        gap_width_1 = gap_x2-left_gap_x1
        gap_depth_2 = 0
        gap_depth_1 = np.abs(gmin)

    return {'gap_width_1': gap_width_1, 'gap_width_2': gap_width_2, 'gap_depth_1': gap_depth_1, 'gap_depth_2': gap_depth_2}


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
