import astropy.constants as k
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import dsharp_opac as opacity
import oofargo
import pandas as pd
import numba as nb

rho_dust = 1.6686*u.g/u.cm**3#g/cm^3
m_star = k.M_sun.cgs
r_planet = k.au.cgs*50
obs_wl = 1.3 * u.mm
mu = 2.3
Sigma0 = 1e-3

surf_rho_dim = m_star/(r_planet**2)

def get_grain_size(Sigma0, Stokes, sigma_slope, r):
    return 2*Sigma0*Stokes*(r**-sigma_slope)*surf_rho_dim/(np.pi*rho_dust) 


datafile = opacity.get_datafile('default_opacities_smooth.npz')
res = np.load(datafile)

obs_wl = 0.13
a_birn     = res['a']
lam   = res['lam']
k_abs = res['k_abs']
k_sca = res['k_sca']
res = opacity.size_average_opacity(0.13, a_birn, lam, k_abs, k_sca)

def get_opacity(a, lamb):
    value = np.interp(a.to('cm')/u.cm, a_birn, res['ka'][0, :])*u.cm**2/u.g
    return value


def get_opac_map(a_map, lamb):
    k_map = get_opacity(a_map, lamb)
    return k_map


def radiative_transfer(data_path, index, i):
    data = oofargo.open_img(data_path, 
                        ntheta=index.loc[i, 'nx'].astype(int),
                        nr = index.loc[i, 'ny'].astype(int),
                        image_rmax=index.loc[i, 'rout'],
                        ylog=True)
    
    h0 = index.loc[i, 'AspectRatio']
    fi = index.loc[i, 'FlaringIndex']
    St = index.loc[i, 'InvStokes1']**-1
    slope = index.loc[i, 'SigmaSlope']
    
    r = np.linspace(0.4, index.loc[i, 'rout'], index.loc[i, 'ny'].astype(int)).reshape(-1,1)
    r = r*np.ones((index.loc[i, 'ny'].astype(int), index.loc[i, 'nx'].astype(int) ))
    
    Td = ((mu*k.m_p/k.k_B)*(h0**2)*(k.G*m_star/r_planet)*(r**(2*fi-1))).to('K')
    a_map = get_grain_size(Sigma0, St, slope, r)
    opac = get_opac_map(a_map, obs_wl)
    tau = opac*data*surf_rho_dim
    
    Ts = Td*(np.ones(Td.shape)-np.exp(-tau))
    
    return Ts
