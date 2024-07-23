#!/usr/bin/env python

#retrieve data in the index
import numpy as np 
import gc
import tensorflow as tf
import os
from tensorflow import keras
import pickle
import time
import models
import itertools
import wandb
import sys

n_i = int(sys.argv[1])
n_f = int(sys.argv[2])

configs = [
          {
'name': f'time{t}.{i}',
'img_pixel_size': (128,128),
'optimizer': tf.keras.optimizers.legacy.SGD(learning_rate=0.0001, momentum=0.9),
'activation': 'leaky_relu',
'smoothing': True,
'beam_pixel_size': 2,
'model_name': 'venus_multip',
'dropout': (0,0,0,0.1, 0.1, 0.1), #6 values needed for simpleCNN
'early_stopping': True,
'patience': 20,
'batch_size':16,
'seed': 47656344%(i+1),
'filename': f'dust1dens{t}.dat'
} for i, t in itertools.product(range(n_i,n_f), [1000])]


data_path = 'training_data/final_allt/final'
#para_file = f'{data_path}para_labelled.csv'
times = [500,1000, 1500]
saving_folder='trained/final_allt'

if not os.path.exists(saving_folder):
    os.mkdir(saving_folder)

def main():
    
    #initialize wandb

    #loading data
    data = {}
    for t in times:
        data[f'time{t}'] = np.load(f'{data_path}{t}/data.npy', allow_pickle=True).item()

    for parameters in configs:
        
        for fold_no in [1,2,3,4,5]:
            
            parameters['fold'] = fold_no
            
            with wandb.init(project='emulator_unet', config=parameters, name=parameters['name']):
              
                start_time = time.time()
                
                #loading the training data concatenating the different times
                train_inp = np.concatenate([np.expand_dims(data[f'time{t}'][f'inp_train{fold_no}'], axis=3) for t in times], axis=0)
                target_train = np.concatenate([data[f'time{t}'][f'targ_train{fold_no}'].reshape(-1) for t in times], axis=0)
                test_inp = np.concatenate([np.expand_dims(data[f'time{t}'][f'inp_test{fold_no}'], axis=3) for t in times], axis=0)
                target_test = np.concatenate([data[f'time{t}'][f'targ_test{fold_no}'].reshape(-1) for t in times],axis=0)
                
                #instantiating the CNN model
                model = models.venus_multip(input_shape=(128,128,1), act=parameters['activation'],
                                        dropout=parameters['dropout'], seed=parameters['seed'])
                
                loss = tf.losses.MSE()
                optimizer = parameters['optimizer']
                
                wandb.watch(model, criterion=loss, log_freq=10)

                #preparing callback for early stopping 2
                es2 = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=parameters['patience'], restore_best_weights=True)
                if(parameters['early_stopping']):
                    cb = [es2]
                else:
                    cb=[]
                    
                #compiling model
                model.compile(loss = loss, optimizer=optimizer)
                
                #fitting
                history = model.fit(x=train_inp,
                                    y=target_train,
                                    batch_size=parameters['batch_size'],
                                    epochs=500, verbose=1, shuffle=True,
                                    validation_data=(test_inp, target_test),
                                    callbacks=cb)
            
                #scores
                scores_test = model.evaluate(test_inp, target_test, verbose=0, return_dict=True)
                scores_train = model.evaluate(train_inp, target_train, verbose=0, return_dict=True)
                
                #log to wandb
                
                #save models
                saving_dir = f"{saving_folder}/{parameters['name']}.{fold_no}"
                os.mkdir(saving_dir)
            
                model.save(saving_dir)
                del model

                #save history
                with open(f"{saving_dir}/history.hist", 'wb') as file_h:
                    pickle.dump(history.history, file_h)
               
                #save score
                with open(f"{saving_folder}/scores.data", "a") as scores_file:
                    #img size, fold, mse, mae
                    scores_file.write(f"{parameters['name']},  {fold_no}, {scores_train['loss']}, {scores_test['loss']}, {time.time()-start_time}\n")
            

        #save configuration
        parameters['wall_time'] = time.time()-start_time
        del train_inp
        del test_inp
        gc.collect()


if __name__ == '__main__':
    main()
