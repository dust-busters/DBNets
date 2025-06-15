from scipy import interpolate
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import numpy as np
import argparse
from scipy.interpolate import LinearNDInterpolator as linterp
from scipy.interpolate import NearestNDInterpolator as nearest

import numpy as np
from scipy.interpolate import LinearNDInterpolator
import scipy.sparse as sp

def get_griddata_sparse(old_coord, new_coord):
    
    #print('generating new interpolation function')
    
    xi = np.array(new_coord).T
    old_xi_shape = xi.shape
    xi = xi.reshape(-1, xi.shape[-1])

    #old coord can be an array with shape (N, ndim) or a tuple of arrays of N elements
    if isinstance(old_coord, tuple):
        old_coord = np.array(old_coord).T
    ndim = old_coord.shape[1]
        
    #construct the triangulation using LinearNDInterpolator
    interp = LinearNDInterpolator(old_coord, np.ones((old_coord.shape[0], 1)))
    simplex_indices = interp.tri.find_simplex(xi)

    #construct the interpolation matrix in COO format
    row_indices, col_indices, values = [], [], []

    #TODO: I think this can be done without the python for loop
    for n in range(xi.shape[0]):
        isimplex = simplex_indices[n]
        if isimplex == -1:
            continue  
        
        indices = interp.tri.simplices[isimplex]
        weights = [*(interp.tri.transform[isimplex, :ndim, :ndim] @ 
                     (xi[n] - interp.tri.transform[isimplex, ndim, :])), 
                   1 - (interp.tri.transform[isimplex, :ndim, :ndim] @ 
                        (xi[n] - interp.tri.transform[isimplex, ndim, :])).sum()]

        row_indices.extend([n] * len(indices))
        col_indices.extend(indices)
        values.extend(weights)

    #Convert to CSR format (more efficient for matrix multiplication)
    c = sp.coo_matrix((values, (row_indices, col_indices)), 
                      shape=(xi.shape[0], old_coord.shape[0])).tocsr()

    return lambda values: (c @ values.T).reshape(*old_xi_shape[:-1], -1).T


class LinearNDInterpolatorExt(object):
    def __init__(self, points, values):
        self.funcinterp = linterp(points, values)
        self.funcnearest = nearest(points, values)
    
    def __call__(self, *args):
        z = self.funcinterp(*args)
        chk = np.isnan(z)
        if chk.any():
            return np.where(chk, self.funcnearest(*args), z)
        else:
            return z


#some fixed things
training_set = np.load('training_set.npy')
targ_red = np.load('red_targ.npy')
nr = 50
ntheta = 100
x = y = np.linspace(-4,4,128)
xx, yy = np.meshgrid(x,y )
old_r = np.sqrt(xx**2+yy**2).reshape(-1)
old_theta = np.arctan2(yy, xx).reshape(-1)

#define regridder to polar coordinatess
new_r_sing = np.linspace(0.3,3,50)
new_theta_sing = np.linspace(-np.pi, np.pi, 100)
new_r, new_theta = np.meshgrid(new_r_sing, new_theta_sing)
regrid = get_griddata_sparse(old_coord=(old_r, old_theta), new_coord=(new_r, new_theta))
polar_training_set = regrid(training_set.reshape(-1,128*128))  #--> shape (N_t, ntheta, nr)

#fit interpolator of training dataset
interpolator = LinearNDInterpolatorExt(targ_red, polar_training_set)


def get_cs(input_data, estimates, nsamples=10, mask_rin=0.5, mask_rout=3.0):
	#load input and estimates data
	print('loading all data and parameters')

	#check shape of inputs and best estimates and extract mean and std

	if len(input_data.shape) == 3:
		input_data = input_data.reshape(1, *input_data.shape[:-1])
     
	if len(input_data.shape) != 4:
		print(f'Input data has shape {input_data.shape} which is wrong. Exiting...')
		exit(1)
	else:
	    N = input_data.shape[0]

	polar_input_data = regrid(input_data.reshape(-1,128*128))  #--> shape (N, ntheta, nr)

	best_estimates = estimates
	if len(best_estimates.shape)==2:
		if best_estimates.shape[1] == 4:
			best_estimates = best_estimates.reshape(1,-1,4)
	if len(best_estimates.shape) != 3:
		print(f'Best estimates data has shape {best_estimates.shape} which is wrong. Exiting...')
		exit(2)

	print('Sample 10 estimates from the posteriror distribution') #--> shape (nsamples, N, 4)
	best_estimates = np.transpose(best_estimates, [1,0,2])
	if best_estimates.shape[0] < nsamples:
		print('Not enough samples from the inferred posterior! Exiting')
		exit(2)
	elif best_estimates.shape[0] > nsamples:
		indices = np.random.choice(best_estimates.shape[0], size=(10, *best_estimates.shape[1:]), replace=False)
		best_estimates_samples = np.take_along_axis(best_estimates, indices, axis=0)
	else:
		best_estimates_samples = best_estimates
	#interpolate training data to the best stimates
	print(f'Interpolating {nsamples} best estimates')
	bi = interpolator(best_estimates_samples) #--> shape (nsamples, N, ntheta, nr)
	    
	#standardize data and reshape input to match the shape of bi
	print('Standardize data')
	polar_input_data = np.nan_to_num(polar_input_data)
	bi = np.nan_to_num(bi)

	polar_input_data = (polar_input_data-polar_input_data.mean(axis=(1,2)).reshape(1,-1,1,1))/polar_input_data.std(axis=(1,2)).reshape(1,-1,1,1)
	best_interp = (bi-bi.mean(axis=(-1,-2)).reshape(nsamples,-1,1,1))/(bi.std(axis=(-1,-2)).reshape(nsamples,-1,1,1))

	    
	#compute similarity
	print('FFT')
	#np.save('polar_input.npy', polar_input_data)
	#np.save('best_input_polar.npy', best_interp)
	input_fft = np.fft.fft2(polar_input_data, axes=(-2,-1))
	bi_fft = np.fft.fft2(best_interp, axes=(-2,-1))

	print('done.\n Compute metric..')
	mse = np.mean(np.mean((np.abs(input_fft) - np.abs(bi_fft))**2, axis=(-2,-1)
		      )/np.mean(np.abs(input_fft)**2+np.abs(bi_fft)**2, axis=(-2,-1)), axis=0)

	print('DONE!')

	return mse
