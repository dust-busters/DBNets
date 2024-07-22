import numpy as np
import pandas as pd
import tensorflow as tf
import itertools as it
#from .training import train
from scipy.special import ndtr 
from scipy.stats import rv_continuous
from scipy.stats import norm
from tf_keras_vis.saliency import Saliency
from tqdm import tqdm
import pkg_resources
import os
import re
from pdfclass import sum_of_norm, extract_prediction
class DBNets:
    '''
    Dust Busters Nets: ensemble of Convolutional Neural Networks trained to infer the mass of possible planets embedded in protoplanetary discs.
    This class handles the ensemble allowing its application to observations. It can be used both with single or multiple images.
    Please note that in order to obtain reliable predictions the input image should be rescaled to match the scale, size and orientation of the images used to train the ensemble.
    We provide methods to do that through the submodule DBNets.preproc
    
    Attributes
    -------------
    ensemble: str
    	path of the trained ensemble that should be used. A default choice is already set.
    n_models: int
    	number of trained groups of models in the ensemble
    folds: int
    	number of models in each group. Each fold was trained on a different portion of the training set
    
    Methods:
    -------------
    get_weights()
        returns the weights of the ensemble
    predict(image, time, dropout_augm)
        returns predictions for the given images
    extract_prediction(logmrv, equivalent_sigma, return_log)
        summarize a predicted pdf in a value and uncertainty
    get_saliency_map(image, time, score)
        return the saliency map for the prediction on the given image
    '''


    def __init__(self, ensemble='finalRI', n_models=10, folds=range(1,6), threshold=0.25):
        #_init__
        #variables that need to be set in order for the ensemble to work properly
        self.ensemble = ensemble
        self.n_models = n_models 
        self.threshold = threshold
        self.folds = np.array(folds)

        print('Initializing DBNets')

        #loading the models
        self.models = [
            tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), 'trained', f'{ensemble}/{m}.{f}'), compile=False)
            for m, f in tqdm(it.product(range(n_models), folds), desc='Loading the CNN ensemble', total=n_models*len(folds))
        ]

        #loaading the weights
        ensemble_scores = pd.read_csv(os.path.join(os.path.dirname(__file__), 'trained', f'{ensemble}/scores.data'))
        weights = np.array([
                ensemble_scores[(ensemble_scores.name==m) & (ensemble_scores.fold==f)]['train_score']
            for m, f in it.product(range(n_models), folds)]        )
        
        #exp +
        #from shape (n_times, n_models*n_folds, 1) to (n_times, n_models*n_folds)
        self.weights = np.exp(-weights.reshape(weights.shape[:-1]))
        
        #loading objets for saliency maps
        self.saliency = [Saliency(model,
                    clone=True) for model in self.models]


    #return the weights used internally
    def get_weights(self):
        return self.weights
    

    # Measuring function returning the inferred pdf as
    # an instantiated object of the custom pdf class 
    # internally storing all the information needed for its definition
    # Return also the median and the width. (log or lin?)
    def measure(self, image, dropout_augm=1, ens_type='peers', debug=False) -> 'p(log m | x, t)':
    
        image = image.reshape(-1, 128, 128, 1)
        if debug:
            pred = np.array([model(image, training=dropout_augm>1) 
                         for _, model in 
                         tqdm(it.product(range(dropout_augm), self.models))]).reshape(-1, len(image), 2)
        else:
            pred = np.array([model(image, training=dropout_augm>1) 
                         for _, model in 
                         it.product(range(dropout_augm), self.models)]).reshape(-1, len(image), 2)

        logpred = pred[:,:,0]
        data_unc_log = pred[:,:,1]
        
        w = np.repeat(self.weights, dropout_augm)
        p_pred = [sum_of_norm(logpred[:,i], data_unc_log[:,i], w, ensemble_type=ens_type) for i in range(0, len(image))]
        
        if len(p_pred)==1:
            return p_pred[0]
        else:
            return p_pred
        

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

