#projection and deprojection functions
#all the angles here a given in degrees
import numpy as np
from astropy.io import fits
from DBNets import oofargo

def to_rad(deg):
    return deg*np.pi/180

def project_coordinate(coord, inclination, posangle):
    x, y =coord[0], coord[1]
    posangle=to_rad(posangle)
    inclination=to_rad(inclination)
    xp = np.cos(posangle)*x - np.sin(posangle)*y*np.cos(inclination)
    yp = np.sin(posangle)*x + np.cos(posangle)*y*np.cos(inclination)
    return np.array([xp, yp])

def deproject_coordinate(coord, inclination, posangle):
    x, y = coord[0], coord[1]
    posangle=to_rad(posangle)
    inclination=to_rad(inclination)
    xp = np.cos(posangle)*x+np.sin(posangle)*y
    yp = -np.sin(posangle)*x/np.cos(inclination) + np.cos(posangle)*y/np.cos(inclination)
    return np.array([xp, yp])

#conversion functions
def au_to_deg(au, distance):
    #the distance must be in parsec
    return au/(distance*3600)

#in the following functions the pxscale 
#is always given in deg/px
def deg_to_px(deg, pxscale):
    return deg/pxscale

def au_to_px(au, distance, pxscale):
    return deg_to_px(au_to_deg(au, distance), pxscale)

import cv2 as cv
import numpy as np
def deproject_image(
    image,
    distance, 
    inclination,
    posangle,
    center,
    r_rif,
    pxscale=None,
    new_img_size=(128,128),
    new_rrif_pxpos=16,
    mask=True,
    smooth=False,
    original_res=0
):
    
    #open image if is not a data
    if isinstance(image, str):
        hdu = fits.open(image)
        disc_image = hdu[0].data
        disc_image = disc_image.reshape(disc_image.shape[-2:])
        disc_image = np.nan_to_num(disc_image)
        disc_image = (disc_image-disc_image.mean())/disc_image.std()

        if pxscale is None:
            try:
                pxscale = np.abs(hdu[0].header['CDELT1'])
            except KeyError:
                print('fits file does not contain the pixel scale, please provide it manually through the pxscale argument')
                return None
    else:
        disc_image = np.nan_to_num(image)
        disc_image = (disc_image-disc_image.mean())/disc_image.std()
            
    if pxscale is None:
        print('Please provide the pixel scale through the pxscale argument')
        return None

    #find the coordinate in pixels of the 3 points 
    #the center is given
    center = np.array(center)
    
    #first point
    p1 = np.array([r_rif, 0]) #au in the deprojected plane
    p1 = project_coordinate(p1, inclination, posangle) #au in the projected plane
    p1 = au_to_px(p1, distance, pxscale) + center #in pixels 
    
    #second point
    p2 = np.array([0, r_rif]) #au in the deprojected plane
    p2 = project_coordinate(p2, inclination, posangle) #au in the projected plane
    p2 = au_to_px(p2, distance, pxscale) + center #in pixels 
    
    #generating affine transformation
    new_center = np.array(new_img_size)/2
    new_p1 = np.array([new_rrif_pxpos, 0]) + new_center
    new_p2 = np.array([0, new_rrif_pxpos]) + new_center
    warp_mat = cv.getAffineTransform(np.array([center, p1, p2]).astype(np.float32), np.array([new_center, new_p1, new_p2]).astype(np.float32))
    warped_img = cv.warpAffine(disc_image, warp_mat, new_img_size)
    final_img = warped_img
    
    
    if mask:
        from scipy.ndimage import gaussian_filter
        mask_res = original_res if not smooth else 0.125
        mask_polar = np.ones((128,384))
        mask_cartesian = gaussian_filter(oofargo.warp_image_rolltodisk(mask_polar, target_image_size=(128,128), target_rmax=4, image_rmax=4),mask_res)
        final_img = warped_img*mask_cartesian

    if smooth:
        newsmooth2 = 0.125**2-original_res**2
        if newsmooth2 > 0:
            final_img = gaussian_filter(final_img, np.sqrt(newsmooth2)*16)

    final_img = (final_img-final_img.mean())/final_img.std()
	
    return final_img

import cv2

def augment(image, nx, ny, rmin, rmax, rtarget, slope):
    
    #create new grid of r
    r = np.arange(rmin, rtarget, (rmax-rmin)/ny)
    new_ny = len(r)
    
    #extrapolate profile
    if new_ny < ny:
        return image[:new_ny, :], new_ny
    else:
        padded_im = np.pad(image, ((0, new_ny-ny),(0,0)),'constant', constant_values=(0,))
        rgrid = np.ones((new_ny, nx))*r.reshape(-1,1)
        prof = image[-1, :]*(rgrid/rmax)**(-slope)*(np.arange(0,new_ny,1)>ny-1).astype(int).reshape(-1,1)
        return prof+padded_im, new_ny

def augment_and_warp(image, rtarg, nx, ny, rmin, rmax, slope):
    im_data, new_ny = augment(image, nx, ny, rmin, rmax, rtarg, slope)
    
    img =  oofargo.warp_image_rolltodisk(im_data, nx, new_ny, image_rmax = rtarg, target_rmax=4, target_image_size=(1280,1280))
    normalized = (img-img.mean())/(img.std())
    #norm_noisy = np.array(GaussianNoise(0.1*normalized.max())(normalized, True))
    #--> with gaussian filter
        #img = gaussian_filter(cv2.resize(normalized, (128,128), interpolation=cv2.INTER_AREA), 2)  
    #--> without gaussian filter
    img = cv2.resize(normalized, (128,128), interpolation=cv2.INTER_AREA)
    #imglog = img.copy()*(img>0.01).astype(int) + (img<=0.01).astype(int)*0.01
    #imglog = (np.log10(imglog)+2)/2
    return img

from .training import radiative_transfer as rt
import astropy.units as u
def fargo_density_to_intensity_standard(filename, ntheta, nr, rin, rout, h0, fi, St, sigma_slope, ylog=True):
    image = np.array(
        rt.radiative_transfer(
            filename,
              ntheta,
                nr,
                  rout,
                    h0,
                      fi,
                        St,
                          sigma_slope,
                            ylog=True)/u.K)
    image_aw = augment_and_warp(image, 4, ntheta, nr, rin, rout, sigma_slope)

    return image_aw