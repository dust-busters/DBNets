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
    smooth=True,
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
        mask_polar = np.ones((128,384))
        mask_cartesian = gaussian_filter(oofargo.warp_image_rolltodisk(mask_polar, target_image_size=(128,128), target_rmax=4, image_rmax=4),2)
        final_img = warped_img*mask_cartesian

    if smooth:
        newsmooth2 = 0.125**2-original_res**2
        if newsmooth2 > 0:
            final_img = gaussian_filter(final_img, np.sqrt(newsmooth2)*16)

    final_img = (final_img-final_img.mean())/final_img.std()
	
    return final_img
