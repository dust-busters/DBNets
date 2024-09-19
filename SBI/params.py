params = {
'run_name': "",
'test_data': [
    f"../test/test_data/test.onlydrop.{fold}.1724961550.063131.pkl"
    for fold in range(1, 5)
],
'testing_resolutions': [0.0, 0.05, 0.1, 0.15],
'only_test': False,
'load_posterior': None,
'times': [500,1000,1500],
'density_estimator': 'maf',
'npe_features': 500,
'learning_rate': 1e-4,
'hidden_features': 50, #both for maf and nsf, default=50
'n_transforms': 5, #both for maf and nsf, default=5
'n_bins': 10, #only meaningful if using nsf, dafault=10 
'training_resolutions': [0.0, 0.05, 0.1, 0.15],
'train_batch_size': 16,
'min_train_epochs': 50,
'test_fold': 3,
'method': 'method1',
'concat_res': False,
'num_posterior_samples': 1000
}