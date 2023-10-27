#/usr/bin/env python

#retrieve data in the index
import pandas as pd
import dblib.index as index
import numpy as np 
from astropy.io import fits
from tqdm import tqdm 
import gc
from sklearn.model_selection import KFold
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import oofargo
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter
import time
import json
import models
from tensorflow.python import debug as tf_debug
import keras.backend as K

import sys

n_i = int(sys.argv[1])
n_f = int(sys.argv[2])

configs = [
          {
'name': f'pNN.{i}',
'img_pixel_size': (128,128),
'optimizer': tf.keras.optimizers.Nadam(learning_rate=0.0001),
'activation': 'elu',
'smoothing': True,
'beam_pixel_size': 3,
'model_name': 'simpleCNNrotinvP',
'dropout': (0,0,0,0.1, 0.1, 0.1), #6 values needed for simpleCNN
'early_stopping': True,
'patience': 50,
'batch_size': 32,
'seed': 472844%(i+1)
} for i in range(n_i,n_f)]

data_path = '../../data/datarun2/'
para_file = f'{data_path}para_labelled.csv'
filename = 'dust1dens50.dat'

#tf.debugging.experimental.enable_dump_debug_info("/tmp/tfdbg4_logdir", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

for parameters in configs:
    #loading the dataset
    print('reading csv file with parameters')
    index = pd.read_csv(para_file)
    print(f'Parameters file contains {index.shape[0]} rows')
    print(f'Discarding {index[index.Discard==True].shape[0]} images showing 0 gaps')
    index = index.drop(index[index.Discard!=False].index)
    print('loading images')
    density_maps = [oofargo.warp_image_rolltodisk(f'{data_path}out_{key:05}/{filename}', target_image_size=parameters['img_pixel_size'], ylog=True, ntheta=640, nr=200) for key, data in tqdm(index.iterrows())]

    #eliminate missing images
    #density_maps_f = np.array([-1*el for el in density_maps if el is not None])

    #preprocessing
    if parameters['smoothing']:
    #density_maps_f = np.array([gaussian_filter(np.log(el, out=np.zeros_like(el), where=(el!=0)) - np.log(el.max()), sigma=2) for el in density_maps if el is not None])
        density_maps_f = np.array([gaussian_filter(el, sigma=parameters['beam_pixel_size']) for el in density_maps if el is not None])

    #load images and labels
    inputs = density_maps_f
    print(inputs.shape)
    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], inputs.shape[2],  1)

    targets = index['PlanetMass'].tolist()
    #for i, el in enumerate(index['PlanetMass'].tolist()):
     #   if density_maps[i] is None:
      #      targets.remove(el)
       #     index= index.drop(i)
    targets = np.log10(np.array(targets)*1e3)
    m_targets = targets.mean()
    std_targets = targets.std()
    #targets = (targets-m_targets)/std_targets
    index.reset_index()

    #kfold cross validation
    from sklearn.model_selection import KFold
    num_folds = 5
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=8)

    fold_no = 1
    input_shape = inputs.shape[1:]
    start_time = time.time()
    for train, test in kfold.split(inputs, targets):

        model = models.simpleCNNrotinvP(input_shape=(parameters['img_pixel_size'][0],parameters['img_pixel_size'][1],1), act=parameters['activation'], dropout=parameters['dropout'], seed=parameters['seed'])
        #model=models.test()
        parameters['model_summary'] = model.summary()

        opt =  parameters['optimizer']

        #warm up
        model.compile(loss = models.CustomLossWarmup(), optimizer=opt)

        #preparing callback for early stopping 1
        es1 = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=parameters['patience'])
        if(parameters['early_stopping']):
            cb = [es1]
        else:
            cb=[]
        
        model.fit(x=inputs[train], y=targets[train], batch_size=parameters['batch_size'], epochs=150, verbose=1, shuffle=True, validation_data=(inputs[test], targets[test]), callbacks=cb)
        
        #preparing callback for early stopping 2
        es2 = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=parameters['patience'])
        if(parameters['early_stopping']):
            cb = [es2]
        else:
            cb=[]
            
        #real training
        model.compile(loss = models.CustomLoss(), optimizer=opt)

        history = model.fit(x=inputs[train], y=targets[train], batch_size=parameters['batch_size'], epochs=300, verbose=1, shuffle=True, validation_data=(inputs[test], targets[test]), callbacks=cb)

        scores = model.evaluate(inputs[test], targets[test], verbose=0, return_dict=True)
        
        #computing r2
        #y_pred, _ = tf.split(model.predict(inputs[test]), num_or_size_splits=2, axis=1)
        #y_pred = y_pred.reshape(1, len(y_pred))[0]
        #r2_metric = tfa.metrics.r_square.RSquare()
        #r2_metric.update_state(targets[test], y_pred)
        
        saving_dir = f"{parameters['name']}.{fold_no}"
        os.mkdir(saving_dir)

        #doing MC dropout
        #y_samples = np.stack([model(inputs[test],training=True)
                 #   for sample in range(100)], axis=1)
        
        #save history
        #with open(f"{saving_dir}/mcdrop.data", 'wb') as file_mcd:
        #    pickle.dump(y_samples, file_mcd)
            
        #save history
        with open(f"{saving_dir}/indexes.data", 'wb') as file_ind:
            pickle.dump(np.array(index.index.tolist())[test], file_ind)

        #save history
        with open(f"{saving_dir}/history.hist", 'wb') as file_h:
            pickle.dump(history.history, file_h)
            
        #save model
        model.save(saving_dir)
        
        parameters[f'scores_fold{fold_no}'] = scores
        #parameters[f'r2_fold{fold_no}'] = r2_metric.result()

        #save score
        with open("scores.data", "a") as scores_file:
            #img size, fold, mse, mae
            scores_file.write(f"{parameters['name']},  {fold_no}, {scores['loss']}\n")
        
        fold_no = fold_no + 1

    #save configuration
    parameters['wall_time'] = time.time()-start_time
    #config = pd.DataFrame(parameters)
    #config.to_csv('config_history.csv', mode='a', header=not os.path.exists('config_history.csv'))
    #with open(f"{parameters['name']}.csv", 'w') as fp:
    #    json.dump(dict, fp)
