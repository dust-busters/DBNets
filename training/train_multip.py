#!/usr/bin/env python

#retrieve data in the index
import numpy as np 
import gc
import keras
import tensorflow as tf
import os
import pickle
import time
import models
import itertools
import wandb
import pandas as pd
import sys
import keras_cv
import os
from wandb.integration.keras import WandbCallback

os.environ["KERAS_BACKEND"] = "tensorflow"

project_name = 'dbnets2.0.0'

n_i = int(sys.argv[1])
n_f = int(sys.argv[2])

configs = [
          {
'name': f'test{i}',
'img_pixel_size': (128,128),
'optimizer': keras.optimizers.Adam(learning_rate=0.0001),
'activation': 'leaky_relu',
'smoothing': True,
'model_name': 'venus_multip',
'dropout': 0.1,
'noise': 0.1,
'maximum_augm_resolution': 0.2, #in units of a
'early_stopping': True,
'patience': 20,
'batch_size':16,
'seed': 47656344%(i+1),
'data_path': 'training_data/only_subs_nosmooth/',
'times': [500,1000,1500],
'saving_folder': 'trained/final_allt',
'override': True,
'resume': False
} for i in range(n_i,n_f)]


print(tf.__version__)

def train(params, fold):

    with wandb.init(project='emulator_unet', config=params, name=params['name']):
        
        #saving start time
        start_time = time.time()
        
        #loading the training data concatenating the different times
        train_inp = np.concatenate([np.expand_dims(data[f'time{t}'][f'inp_train{fold_no}'], axis=3) for t in params['times']], axis=0)
        target_train = np.concatenate([data[f'time{t}'][f'targ_train{fold_no}'].reshape(-1,6) for t in params['times']], axis=0)
        test_inp = np.concatenate([np.expand_dims(data[f'time{t}'][f'inp_test{fold_no}'], axis=3) for t in params['times']], axis=0)
        target_test = np.concatenate([data[f'time{t}'][f'targ_test{fold_no}'] for t in params['times']],axis=0)
        
        print(target_test.shape)
        #instantiating the CNN model
        model = models.venus_multip(input_shape=(128,128,1), act=params['activation'],
                                dropout=params['dropout'], seed=params['seed'], noise=params['noise'], maximum_res=params['maximum_augm_resolution'])
        
        print(target_train.shape)
        print(target_test.shape)
        print(train_inp.shape)
        print(test_inp.shape)
        optimizer = params['optimizer']

        #preparing callback for early stopping 2
        es2 = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=params['patience'], restore_best_weights=True)
        if(params['early_stopping']):
            cb = [es2]
        else:
            cb=[]
        
        #preparing metrics
        metrics = [models.custom_metric_output_i(i) for i in range(6)]
        
        #wandb callback
        wandb_callback = WandbCallback(monitor='val_loss',
                               log_weights=True,
                               log_evaluation=True,
                               validation_steps=5)
        cb = cb + wandb_callback

            
        #compiling model
        model.compile(loss = 'mse', optimizer=optimizer, metrics=metrics)
        
        
        #fitting
        history = model.fit(x=train_inp,
                            y=target_train,
                            batch_size=params['batch_size'],
                            epochs=500, verbose=1, shuffle=True,
                            validation_data=(test_inp, target_test),
                            callbacks=cb)
    
        #scores
        #scores_test = model.evaluate(test_inp, target_test, verbose=0, return_dict=True)
        #scores_train = model.evaluate(train_inp, target_train, verbose=0, return_dict=True)
        train_res = model(train_inp)
        
        
        #log to wandb
        
        #save models
        saving_dir = f"{params['saving_folder']}/{params['name']}.{fold_no}"
        os.mkdir(saving_dir)
    
        model.save(saving_dir)
        del model

        #save history
        with open(f"{saving_dir}/history.hist", 'wb') as file_h:
            pickle.dump(history.history, file_h)
        
        #save score
        with open(f"{params['saving_folder']}/scores.data", "a") as scores_file:
            #img size, fold, mse, mae
            scores_file.write(f"{params['name']},  {fold_no}, {scores_train['loss']}, {scores_test['loss']}, {time.time()-start_time}\n")
    

        #save configuration
        params['wall_time'] = time.time()-start_time
        del train_inp
        del test_inp
        gc.collect()


if __name__ == '__main__':
    
    for params in configs:
        #checking if exists and creating output directory if it does not
        if os.path.exists(params['saving_folder']):
            if params['override']:
                print('Saving directory exists, overriding old data as instructed.')
            else:
                print('WARNING! -> saving directory already exists, please run with Override=True')
                exit()
        else:
            os.mkdir(params['saving_folder'])
        
        #loading data
        data = {}
        for t in params['times']:
            data[f'time{t}'] = np.load(f'{params["data_path"]}{t}/data.npy', allow_pickle=True).item()

        #checking file with parameter history and adding this run
        if os.path.exists('parahist.csv'):
            oldpara = pd.read_csv('parahist.csv', index_col=0)
            params['index'] = oldpara.index[-1]+1
            newparafile = pd.concat([oldpara, pd.DataFrame([params]).set_index('index')])
        else:
            params['index'] = 0
            newparafile = pd.DataFrame([params]).set_index('index')
        newparafile.to_csv('parahist.csv')
    
        with wandb.init(project=project_name, config=params, name=params['name']):
            #begin train
            if params['resume']:
                if not os.path.exists(params['resume_from']):
                    print('Error! the model wich you want to resume from does not exist!\n Exiting...')
                    exit()
                else:
                    #TODO: implement possibility to resume
                    print('Error! Resuming not yet implemented')
                    exit()
            else:
                for fold_no in [1,2,3,4,5]:
                    params['fold'] = fold_no
                    train(params, fold_no)
