import numpy as np
import pandas as pd
import tensorflow as tf
import itertools as it
from .training import train
from scipy.special import ndtr 
from scipy.stats import rv_continuous
from scipy.stats import norm
from tf_keras_vis.saliency import Saliency
from tqdm import tqdm
import pkg_resources
import os
import re
class DBNets:
    '''
    Dust Busters Nets: ensemble of Convolutional Neural Networks trained to infer the mass of possible planets embedded in protoplanetary discs.
    This class handles the ensemble allowig its application to observations. It can be used both with single or multiple images.
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


    def __init__(self, ensemble='final', n_models=10, folds=range(1,6)):
        #_init__
        #variables that need to be set in order for the ensemble to work properly
        self.ensemble = ensemble
        self.n_models = n_models 
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

    
    def finetune(self, newdatax, newdatay, newdatax_test, newdatay_test, ftname, memory=0.8, epochs=20):
        folder = os.path.join(os.path.dirname(__file__), 'trained/', f'{ftname}')
        if os.path.exists(folder):
            folders = os.listdir(folder)
            mod = np.array([np.array(f.split('.')).astype(int) for f in folders if re.compile('\d.\d').match(f)])
            max_model = np.max(mod[:,0])
            max_fold = np.max([m[1] for m in mod if m[0]==max_model])
            print(f'This fine tuned version already exists. Restarting, last found model {max_model} fold {max_fold}')
        else:
            os.mkdir(folder)
            with open(f"{folder}/scores.data", "a") as scores_file:
                scores_file.write(f"name,fold,train_score_old,test_score_old,train_score, test_score\n")
            max_model = 0
            max_fold = 0

            print('starting fine tuning')
            i=-1
            for m, f in tqdm(it.product(range(self.n_models), self.folds)):
                i+=1
                if m<max_model:
                    break
                else:
                    if m==max_model:
                        if f <= max_fold:
                            break
                train.finetune(self.models[i], newdatax, newdatay, newdatax_test, newdatay_test, ftname, m, f, memory=memory, epochs=epochs)
                

    
    

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

#Custom pdf class
class sum_of_norm(rv_continuous):

    "Distribution generated from a sum of gaussians"

    def __init__(self, locs, scales, weights, ensemble_type='peers'):
        super().__init__()
        self.locs = np.array(locs).reshape(-1, 1)
        self.scales = np.array(scales).reshape(-1, 1)
        self.weights = np.array(weights).reshape(-1)
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
        self.reliable = ((temp[2]-temp[1])/2)<0.3
    
    def summary_measure(self, equivalent_sigma=1, return_log=False):
        return extract_prediction(self, equivalent_sigma=1, return_log=return_log)


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

