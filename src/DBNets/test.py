import numpy as np
import pandas as pd
import oofargo
from DBNets.DBNets import DBNets
import pickle
from scipy.ndimage import gaussian_filter
from tqdm import tqdm 
import os

#at the end of the iterations this script should generate these files:
# - images.npy
# - log_m_pred.npy
# - std of the predicted distribution
# - the complete predicted distribution
# - the target planet mass (or all the parameters)

'''
TODO: * extend code for images at other times
'''

ensemble = 'finalRI'
#opening data
name = 'evolution'
data_path = '../training/training_data/final/final'
#os.mkdir(name)
#importing index file with all the parameters


def test(out_dir, ensemble=ensemble, data_path=data_path, times=[500, 1000, 1500]):

	if not os.path.exists(out_dir):
		print('Error: output dir does not exist')
		return False

	target_m = np.array([])
	ids = np.array([])
	comppred = np.array([])
	predobj = np.array([])

	
	data = {}
	for t in times:
		data[f'time{t}'] = np.load(f'{data_path}{t}/data.npy', allow_pickle=True).item()

	#repeat for each fold
	for fold in tqdm([1,2,3,4,5]):

	    #loading models of the given fold
		dbnets = DBNets(ensemble=ensemble,
		            n_models=10, folds=[fold])

		test_imgs = np.concatenate([np.expand_dims(data[f'time{t}'][f'inp_test{fold}'], axis=3) for t in times], axis=0)
		target_test = np.concatenate([data[f'time{t}'][f'targ_test{fold}'].reshape(-1) for t in times],axis=0)

	    #extract predictions
		results = dbnets.measure(test_imgs, dropout_augm=1, ens_type='peers')
		cpred = [res.summary_measure(return_log=True) for res in results]
		comppred = np.append(comppred, cpred)
	    #target_m = np.append(target_m, np.log10(np.array(test_para['PlanetMass'])*1e3))
		target_m = np.append(target_m,target_test)
	    #ids = np.append(ids, indx)
		predobj = np.append(predobj, results)

	np.save(f'{out_dir}/mtarget1000s.npy', target_m)
	#np.save(f'{name}/ids.npy', ids)
	np.save(f'{out_dir}/comppred1000s.npy', comppred.reshape(-1,3))
	with open(f'{out_dir}/predobj1000s.pickle', 'wb') as pf:
	    pickle.dump(predobj, pf)
