import numpy as np

def normalize_input_data(data):
    maximums = np.max(data.reshape(-1, 128*128), axis=1).reshape(-1,1,1,1)
    data = data/maximums
    return data

def standardize_input_data(data):
    data = (data-data.mean(axis=(1,2,3)).reshape(-1,1,1,1))/data.std(axis=(1,2,3)).reshape(-1,1,1,1)
    return data

def log_and_standardize_input_data(data):
    data = np.log10(data+1e-12)
    data = (data-data.mean(axis=(1,2,3)).reshape(-1,1,1,1))/data.std(axis=(1,2,3)).reshape(-1,1,1,1)
    return data

def log_and_normalize_input_data(data):
    data = np.log10(data+1e-12)
    maximums = np.max(data.reshape(-1, 128*128), axis=1).reshape(-1,1,1,1)
    data = data/maximums
    return data

norm_functions = {
    'normalize_input_data': normalize_input_data,
    'standardize_input_data': standardize_input_data,
    'log_and_normalize_input_data': log_and_normalize_input_data,
    'log_and_standardize_input_data': log_and_standardize_input_data
}

configs = [
          {
'name': f'only4para2_long_norout_nocut',
'img_pixel_size': (128,128),
'n_res_blocks': 1,
'lr': 5e-5,
'activation': 'leaky_relu',
'smoothing': True,
'inf_para': [0,1,2,4],
'model_name': 'venus_multip',
'dropout': 0.2,
'maximum_translation_factor': 0.01,
'noise': 0.1,
'maximum_augm_resolution': 0.2, #in units of a
'early_stopping': False,
'patience': 50,
'batch_size':64,
'sweep': False,
'seed': 47656344%(58+1),
'data_path': 'training_data/only_subs_nosmooth_nonorm_noroutaugm_c/',
'times': [500,1000,1500],
'saving_folder': 'trained/',
'override': True,
'resume': False,
'regularizer': None,
'batch_normalization': False,
'epochs': 3000,
'res_blocks': [32,64,128],
'dense_dimensions': [256,128,128],
'norm_input': 'standardize_input_data'
}]