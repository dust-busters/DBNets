
configs = [
          {
'name': f'test',
'img_pixel_size': (128,128),
'lr': 1e-4,
'activation': 'leaky_relu',
'smoothing': True,
'model_name': 'venus_multip',
'dropout': 0.1,
'noise': 0.1,
'maximum_augm_resolution': 0.2, #in units of a
'early_stopping': True,
'patience': 20,
'batch_size':16,
'seed': 47656344%(58+1),
'data_path': 'training_data/only_subs_nosmooth/',
'times': [500,1000,1500],
'saving_folder': 'trained/final_allt',
'override': True,
'resume': False
}]