import wandb
import os
import argparse
import time
from config import normalize_input_data, standardize_input_data, log_and_standardize_input_data, log_and_normalize_input_data

global_params = {
    "name": "test_sweep",
    "times": [500, 1000, 1500],
    "data_path": "training_data/only_subs_nosmooth_nonorm/",
    "saving_folder_g": "trained/",
    "override": True,
    "resume": False,
}

sweep_config = {
    "method": "random",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "img_pixel_size": {"value": (128, 128)},
        "lr": {"distribution": "uniform", "min": 5e-6, "max": 1e-4},
        "activation": {"values": ["leaky_relu", "relu", "elu"]},
        "smoothing": {"value": True},
        "dropout": {"distribution": "uniform", "min": 0.1, "max": 0.5},
        "maximum_translation_factor": {
            "distribution": "uniform",
            "min": 0.01,
            "max": 0.05,
        },
        "noise": {"distribution": "uniform", "min": 0.01, "max": 0.1},
        "maximum_augm_resolution": {"distribution": "uniform", "min": 0.15, "max": 0.2},
        "early_stopping": {"value": False},
        "patience": {"value": 200},
        "batch_size": {"values": [8, 16, 32, 64, 128]},
        "seed": {"value": 47656344 % (58 + 1)},
        "sweep": {"value": True},
        "dense_dimensions": {"value": [[256,256,256,128], [256,256,128], [256, 128], [256,128,128], [256,128,64]]},
        "res_blocks": {"values": [[32,64,128], [64, 128, 256]]},
        "norm_input": {"values": [normalize_input_data, standardize_input_data, log_and_normalize_input_data, log_and_standardize_input_data]}
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

    sweep_id = wandb.sweep(sweep_config, project="dbnets2.0.0")

    # sleeping for 30 seconds to get the server running
    #time.sleep(30)

    print("Please manually launch the agents with the following commands.")
    print("Doing it from python is currently not working due to mysterious forces.")
    print(f"sbatch runa100_sweep.sh {sweep_id} {args.count}")

    # launching agents
    # for i in range(args.n_sweep_agents):

    # for some reason if I launch the slurm jobs from python with the following line, wandb cannot find the sweep
    # Hence, manually launch it with the printe command.
    # os.system(f"sbatch runa100_sweep.sh {sweep_id} {args.count}")
