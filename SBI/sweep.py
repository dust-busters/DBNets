import wandb
import os
import argparse
import time

import numpy as np

ALPHA=0
ASPECT_RATIO=1
INVSTOKES=2
PLANET_MASS = 4


global_params = {
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
'training_resolutions': [0.0, 0.05, 0.1, 0.15],
'method': 'method2',
'concat_res': False,
'num_posterior_samples': 3000
}


sweep_config = {
    "method": "random",
    "metric": {"name": "tarp_atc", "goal": "minimize"},
    "parameters": {
        #---
        'density_estimator': {'values': ['maf', 'nsf']},
        'npe_features': {'values': [500, 1000, 1500]},
        'learning_rate': {'values': [1e-3, 5e-4, 1e-4, 5e-5]},
        'hidden_features': {'values': [25,50,75,100]}, #both for maf and nsf, default=50
        'n_transforms': {'values': [5,10,15]},#both for maf and nsf, default=5
        'n_bins': {'values': [5,10, 15]}, #only meaningful if using nsf, dafault=10 
        'train_batch_size': {'values': [16,32,64,128]},
        'min_train_epochs': {'values': [50, 100,150, 200]},
        'test_fold': {'values': [1,2,3,4,5]},
        
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Wandb sweep launcher",
        description="Launches a sweep on slurm using wandb",
    )

    parser.add_argument("-ns", "--n-sweep-agents", required=False, default=50, type=int)
    parser.add_argument("-c", "--count", required=False, default=1, type=int)
    args = parser.parse_args()

    sweep_id = wandb.sweep(sweep_config, project="dbnets2.0.0_SBI")

    # sleeping for 30 seconds to get the server running
    #time.sleep(30)

    print("Please manually launch the agents with the following commands.")
    print("Doing it from python is currently not working due to mysterious forces.")
    print(f"sbatch runcpu_sweep.sh {sweep_id}")

    # launching agents
    # for i in range(args.n_sweep_agents):

    # for some reason if I launch the slurm jobs from python with the following line, wandb cannot find the sweep
    # Hence, manually launch it with the printe command.
    # os.system(f"sbatch runa100_sweep.sh {sweep_id} {args.count}")
