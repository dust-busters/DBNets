from . import models
import numpy as np
import os
import pickle
import tensorflow as tf
import gc


#this is needed because I am using the legacy optimizer
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from tf.keras.optimizers.legacy import Adam

class CustomLossFineTune(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__()
  def call(self, y_true, y_pred):
    o_mean = y_pred[:, :1]
    o_std = y_pred[:, 1:]
    loss = tf.reduce_mean(tf.math.log(o_std+1e-6) + tf.math.square((o_mean-y_true)/(o_std+1e-6))/2)
    return loss
    

def finetune(model, newdatax, newdatay, newdatax_test, newdatay_test, ftname, model_number, fold_no, memory=0.8, optimizer=Adam(1e-5), epochs=20):

    #freeze convolutional layers
    import keras
    for layer in model.layers:
        if layer.name in ['FC1', 'FC2', 'o_mean', 'o_std']:
            layer.trainable = True
        else:
            layer.trainable = False

    #load old data
    data = {}
    times=[500,1000,1500]
    for t in times:
        data[f'time{t}'] = np.load(f'{os.path.dirname(__file__)}/data/final/final{t}/data.npy', allow_pickle=True).item()
    train_inp = np.concatenate([np.expand_dims(data[f'time{t}'][f'inp_train{fold_no}'], axis=3) for t in times], axis=0)
    target_train = np.concatenate([data[f'time{t}'][f'targ_train{fold_no}'].reshape(-1) for t in times], axis=0)
    test_inp = np.concatenate([np.expand_dims(data[f'time{t}'][f'inp_test{fold_no}'], axis=3) for t in times], axis=0)
    target_test = np.concatenate([data[f'time{t}'][f'targ_test{fold_no}'].reshape(-1) for t in times],axis=0)

    weights = np.ones(len(target_train))*memory

    #join with new data
    train_inpf = np.concatenate([train_inp, newdatax])
    target_trainf = np.concatenate([target_train, newdatay])
    test_inpf = np.concatenate([test_inp, newdatax_test])
    target_testf = np.concatenate([target_test, newdatay_test])
    weights = np.concatenate([weights, np.ones(len(newdatay))*(1-memory)])

    #optimize
    model.compile(
        optimizer=optimizer,
        loss=CustomLossFineTune()
    )

    history = model.fit(x=train_inpf, y=target_trainf, batch_size=64, epochs=epochs, verbose=1, shuffle=True, validation_data=(test_inpf, target_testf), sample_weight=weights)

    scores_test_old = model.evaluate(test_inp, target_test, verbose=0, return_dict=True)
    scores_train_old = model.evaluate(train_inp, target_train, verbose=0, return_dict=True)

    scores_test_new = model.evaluate(newdatax_test, newdatay_test, verbose=0, return_dict=True)
    scores_train_new = model.evaluate(newdatax, newdatay, verbose=0, return_dict=True)

    saving_dir = os.path.join(os.path.dirname(__file__), '../trained/', f'{ftname}/{model_number}.{fold_no}')
    saving_folder = os.path.join(os.path.dirname(__file__), '../trained/', f'{ftname}')
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)

    #save history
    with open(f"{saving_dir}/history.hist", 'wb') as file_h:
        pickle.dump(history.history, file_h)
        
    #save model
    model.save(saving_dir)

    #save score
    with open(f"{saving_folder}/scores.data", "a") as scores_file:
        scores_file.write(f"{ftname}{model_number},  {fold_no}, {scores_train_old['loss']}, {scores_test_old['loss']}, {scores_train_new['loss']}, {scores_test_new['loss']}\n")
    
    del data
    del train_inpf
    del target_trainf
    del test_inpf
    del target_testf
    del train_inp
    del target_train
    del test_inp
    del target_test
    gc.collect()