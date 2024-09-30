import numpy as np

ALPHA=0
ASPECT_RATIO=1
INVSTOKES=2
PLANET_MASS = 4


params = {
'run_name': "",
'test_data': [
    f"../test/4SBI.{fold}.pkl"
    for fold in range(1, 6)
],
'testing_resolutions': [0.0, 0.05, 0.1, 0.15],
'inf_para': np.array([ALPHA, ASPECT_RATIO, INVSTOKES, PLANET_MASS]),
'only_test': False,
'load_posterior': None,
'times': [500,1000,1500],
'density_estimator': 'maf',
'npe_features': 1000,
'learning_rate': 1e-4,
'hidden_features': 50, #both for maf and nsf, default=50
'n_transforms': 10, #both for maf and nsf, default=5
'n_bins': 10, #only meaningful if using nsf, dafault=10 
'training_resolutions': [0.0, 0.05, 0.1, 0.15],
'train_batch_size': 64,
'min_train_epochs': 100,
'test_fold': 3,
'method': 'method2',
'concat_res': False,
'num_posterior_samples': 3000
}