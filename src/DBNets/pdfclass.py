import numpy as np
import pandas as pd
import itertools as it
from .training import train
from scipy.special import ndtr 
from scipy.stats import rv_continuous
from scipy.stats import norm
import os
import re

#Custom pdf class
class sum_of_norm(rv_continuous):

    "Distribution generated from a sum of gaussians"

    def __init__(self, locs, scales, weights, ensemble_type='peers', threshold=0.25):
        super().__init__()
        self.locs = np.array(locs).reshape(-1, 1)
        self.scales = np.array(scales).reshape(-1, 1)
        self.weights = np.array(weights).reshape(-1)
        self.threshold=threshold
        self.set_ensemble_type(ensemble_type)
        self.set_reliability()

    def set_ensemble_type(self, type):
        if type=='experts':
            self.pweights = self.weights.reshape(-1)*(self.scales.reshape(-1)**-2)
        else:
        	if type=='peers':
                 self.pweights = self.weights
        self.set_reliability()
        #self.pweights = self.pweights.reshape(-1,1)

    def _pdf(self, x):
        xx = np.array(x).reshape(1, -1)
        return np.average(np.exp(-(xx-self.locs)**2 / (2.*(self.scales**2))) / (np.sqrt(2.0 * np.pi)*self.scales), weights=self.pweights, axis=0)

    def _cdf(self, x):
        t = (np.array(x).reshape(1,-1) - self.locs)/self.scales
        return np.average(ndtr(t), weights=self.pweights, axis=0)

    def mean(self):
        return np.average(self.locs, weights=self.pweights, axis=0)

    def var(self):
        return np.average(self.scales**2+self.locs**2, weights=self.pweights, axis=0) -  self.mean()**2

    def std(self):
        return self.var()**0.5
    
    def set_reliability(self):
        temp = self.summary_measure(return_log=True)
        self.reliable = ((temp[2]-temp[1])/2)<self.threshold
    
    def summary_measure(self, equivalent_sigma=1, return_log=False):
        return extract_prediction(self, equivalent_sigma=1, return_log=return_log)
    

# function that given the pdf and a reference prob. value returns
# the best prediction and an estimation for the uncertainty
def extract_prediction(logmrv, equivalent_sigma=1, return_log=False):
    log_m_predicted = logmrv.ppf(0.5)
    m_predicted = 10**log_m_predicted

    log_right_lim = logmrv.ppf(norm.cdf(equivalent_sigma))
    log_left_lim = logmrv.ppf(norm.cdf(-equivalent_sigma))

    right_lim = 10**log_right_lim
    left_lim = 10**log_left_lim

    if return_log:
        return log_m_predicted, log_left_lim, log_right_lim
    else:
        return m_predicted, left_lim, right_lim