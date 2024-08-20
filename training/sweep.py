import wandb
import os
import argparse

sweep_config = {
    "method": "random",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "name": {'value' : f"test_sweep"},
        "img_pixel_size": (128, 128),
        "lr": {"distribution": "uniform", "min": 5e-5, "max": 5e-3},
        "activation": {"values": ["leaky_relu", "relu", "elu"]},
        "smoothing": {'value':True},
        "dropout": {"distribution": "uniform", "min": 0.1, "max": 0.5},
        "maximum_translation_factor": {"distribution": "uniform", "min": 0.01, "max": 0.05},
        "noise": {"distribution": "uniform", "min": 0.01, "max": 0.1},
        "maximum_augm_resolution":  {"distribution": "uniform", "min": 0.05, "max": 0.2},
        "early_stopping": {'value': False},
        "patience": {'value': 200},
        "batch_size": {'values': [8, 16, 32, 64, 128]},
        "seed": {'value': 47656344 % (58 + 1)},
        "data_path": {'value': "training_data/only_subs_nosmooth_nonorm/"},
        "times": {'value': [500, 1000, 1500]},
        "saving_folder": {'value': "trained/"},
        "override": {'value': True},
        "resume": {'value': False},
        "sweep": {'value': True}
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

    sweep_id = wandb.sweep(sweep_config, project="dbnets2.0")

    for i in range(args.n_sweep_agents):
        os.system(f"sbatch runa100_sweep.sh {sweep_id} {args.count}")