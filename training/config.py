import numpy as np

def normalize_input_data(data):
    maximums = np.max(data.reshape(-1, 128*128), axis=1).reshape(-1,1,1,1)
    data = data/maximums
    return data

def standardize_input_data(data):
    data = (data-data.mean(axis=(1,2,3)))/data.std(axis=(1,2,3))
    return data

def log_and_standardize_input_data(data):
    data = np.log10(data+1e-12)
    data = (data-data.mean(axis=(1,2,3)))/data.std(axis=(1,2,3))
    return data

def log_and_normalize_input_data(data):
    data = np.log10(data+1e-12)
    maximums = np.max(data.reshape(-1, 128*128), axis=1).reshape(-1,1,1,1)
    data = data/maximums
    return data

configs = [
          {
'name': f'test',
'img_pixel_size': (128,128),
'n_res_blocks': 1,
'lr': 1e-4,
'activation': 'leaky_relu',
'smoothing': True,
'model_name': 'venus_multip',
'dropout': 0.1,
'maximum_translation_factor': 0.01,
'noise': 0.1,
'maximum_augm_resolution': 0.2, #in units of a
'early_stopping': True,
'patience': 20,
'batch_size':16,
'seed': 47656344%(58+1),
'data_path': 'training_data/only_subs_nosmooth_nonorm/',
'times': [500,1000,1500],
'saving_folder': 'trained/final_allt',
'override': True,
'resume': False,
'res_blocks': [32,64,128],
'dense_dimensions': [256,256,256,128],
'norm_input': standardize_input_data
}]
