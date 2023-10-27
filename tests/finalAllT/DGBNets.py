import numpy as np
import pandas as pd
import tensorflow as tf
import itertools as it
from scipy.special import ndtr 
from scipy.stats import rv_continuous
from scipy.stats import norm
from tf_keras_vis.saliency import Saliency
from tqdm import tqdm

class DBNets:

    def __init__(self, ensemble_path='training_code/alltimes', n_models=10, folds=range(1,6), times_l=range(20,60, 10), times_orb=range(200, 600, 100)):
        #_init__
        #variables that need to be set in order for the ensemble to work properly
        self.ensemble_path = ensemble_path
        self.n_models = n_models 
        self.folds = np.array(folds)
        self.times_l = np.array(times_l)
        self.times_orb = np.array(times_orb)

        #loading the models
        self.models = [[
            tf.keras.models.load_model(f'{ensemble_path}/time{tl}.{m}.{f}', compile=False)
            for m, f in tqdm(it.product(range(n_models), folds))]
            for tl in times_l
        ]
        #loaading the weights
        #int( ensemble_scores[(ensemble_scores.name==f'time{tl}.{m}') & (ensemble_scores.fold==f)]['test_score']<0)
        ensemble_scores = pd.read_csv(f'{self.ensemble_path}/scores.data')
        weights = np.array([[
                ensemble_scores[(ensemble_scores.name==f'time{tl}.{m}') & (ensemble_scores.fold==f)]['train_score']
            for m, f in tqdm(it.product(range(n_models), folds))]
            for tl in times_l
        ])
        #from shape (n_times, n_models*n_folds, 1) to (n_times, n_models*n_folds)
        
        self.weights = np.exp(-weights.reshape(weights.shape[:-1]))
        
        #loaing objets for saliency maps
        self.saliency = [[Saliency(model,
                    clone=True) for model in self.models[t]] for t in range(len(times_l))]


    #exposed functions

    #return the weights used internally
    def get_weights(self):
        return self.weights

    # Predict function returning the inferred pdf as
    # an instantiated object of the custom pdf class 
    # internally storing all the information needed for its definition
    # Return also the median and the width. (log or lin?)
    def predict(self, image, time, dropout_augm=1, ens_type='peers') -> 'p(log m | x, t)':
    
        
        if time == 'all':
        	mm = np.array(self.models).flatten()
        	w = np.repeat(self.weights.flatten(), dropout_augm)
        else:
        	t = np.argmin(np.abs(self.times_orb-time))
        	mm = self.models[t]
        	w = np.repeat(self.weights[t], dropout_augm)

        pred = np.array([model(image, training=dropout_augm>1) for _, model in it.product(range(dropout_augm), mm)]).reshape(-1, len(image), 2)

        logpred = pred[:,:,0]
        data_unc_log = pred[:,:,1]
        
        
        p_pred = [sum_of_norm(logpred[:,i], data_unc_log[:,i], w, ensemble_type=ens_type) for i in range(0, len(image))]

        return p_pred
        
    # function that given the pdf and a reference prob. value returns
    # the best prediction and an estimation for the uncertainty
    def extract_prediction(self, logmrv, equivalent_sigma=1, return_log=False):
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
            
     
    def get_saliency_map(self, image, time, score='m+'):
        if score=='m+':
            scoref = lambda x: x[:,0] 
        elif score=='m-':
            scoref = lambda x: -x[:,0]
        elif score=='s+':
            scoref = lambda x: x[:,1]
        elif score=='s-':
            scoref = lambda x: -x[:,1]

        t = np.argmin(np.abs(self.times_orb-time))
        
        saliency_maps = np.array([s(scoref,image.reshape(1,128,128,1),
        smooth_samples=50,smooth_noise=0.1) for s in tqdm(self.saliency[t])])

        saliencymap = np.average(saliency_maps, axis=0, weights=self.weights[t]).reshape(128,128)
        
        return saliencymap
    


#Custom pdf class
class sum_of_norm(rv_continuous):

    "Distribution generated from a sum of gaussians"

    def __init__(self, locs, scales, weights, ensemble_type='peers'):
        super().__init__()
        self.locs = np.array(locs).reshape(-1, 1)
        self.scales = np.array(scales).reshape(-1, 1)
        self.weights = np.array(weights).reshape(-1)
        self.set_ensemble_type(ensemble_type)

    def set_ensemble_type(self, type):
        if type=='experts':
            self.pweights = self.weights.reshape(-1)*(self.scales.reshape(-1)**-2)
        else:
        	if type=='peers':
                 self.pweights = self.weights
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


#random variable class for discrete marginalization
class discrete_marginalized_dist(rv_continuous):

    def __init__(self, pdfs, weights):
        super().__init__()
        self.pdfs = pdfs
        self.weights = np.array(weights)

    def _pdf(self, x):
        return np.average(np.array([p.pdf(x) for p in self.pdfs]), weights=self.weights)

    def _cdf(self, x):
        return np.average(np.array([p.cdf(x) for p in self.pdfs]), weights=self.weights)

