params = {
'run_name': "",
'test_data': [
    f"../test/test_data/test.onlydrop.{fold}.1724961550.063131.pkl"
    for fold in range(1, 5)
],
'testing_resolutions': [0.0, 0.05, 0.1, 0.15],
'times': [1500],
'density_estimator': 'maf',
'training_resolutions': [0.0, 0.05, 0.1, 0.15],
'train_batch_size': 16,
'min_train_epochs': 50,
'test_fold': 3,
'method': 'method1',
'concat_res': False,
'num_posterior_samples': 1000
}