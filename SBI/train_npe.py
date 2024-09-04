import sbi
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import wandb

sys.path.append("../training")
from train_multip import __LABELS__
from config import configs
import pickle
from ctypes import util
from sbi.inference import NPE
from sbi import utils
import torch
from numpy import float32
import yaml

def concat_dict(a, b):
    data = {}
    if a is None:
        return b
    if b is None:
        return a
    if (a is None) and (b is None):
        return None
    for key in b.keys():
        data[key] = np.concatenate([a[key], b[key]])

    return data

with open('config.yml', 'r') as file:
    params = yaml.safe_load(file)
    

if not os.path.exists("outputs"):
    os.mkdir("outputs")
out_folder = f"outputs/{params['run_name']}"
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

print(f'Saving folder: {out_folder}')

os.system(f"cp train_npe.py {out_folder}/train_npe.npy")

print(f'Loading data in test partition of all folds except {params['test_fold']}')
all_data = None
for fold, test_d in enumerate(params['test_data']):
    if fold + 1 != params['test_fold']:
        with open(test_d, "rb") as file:
            data = pickle.load(file)

        # load targets
        data_t = {}
        for t in params["times"]:
            data_t[f"time{t}"] = np.load(
                f'../training/{params["data_path"]}/{t}/data.npy', allow_pickle=True
            ).item()
        target_test = np.concatenate(
            [data_t[f"time{t}"][f"targ_test{fold+1}"] for t in params["times"]], axis=0
        )
        data["targets"] = target_test[:-1]
        all_data = concat_dict(all_data, data)
        del data_t

# now all_data contains the test data to be used for training the maf NPE


# preparing data for traininf MAF. COncatenating different resolutions.
n_sim = all_data["y_pred_r0.0"].shape[0]
x = torch.tensor(
    np.concatenate(
        [all_data[f"y_pred_r{res}"].reshape(n_sim, -1) for res in params['training_resolutions']]
    ),
    dtype=torch.float32,
)
theta = torch.tensor(
    np.concatenate([all_data["targets"] for res in params['training_resolutions']]),
    dtype=torch.float32,
)

# training
# preparing NPE

print('Data loaded. Starting training of the NPE.')
with wandb.init(project='dbnets2.0.0_SBI', config=params):
    prior = utils.BoxUniform(
        low=torch.tensor([-1, -1, -1, -1, -1, -1]), high=torch.tensor([1, 1, 1, 1, 1, 1])
    )
    inference = NPE(prior, density_estimator="maf")
    _ = inference.append_simulations(theta, x, proposal=prior)
    inference.train(
        training_batch_size=params['train_batch_size'],
        stop_after_epochs=params['min_train_epochs'],
        show_train_summary=True,
        validation_fraction=0.1,
    )
    posterior = inference.build_posterior()

    with open(f"{out_folder}/posterior.pkl", "wb") as f:
        pickle.dump(posterior, f)

    # run tests, compute metrics and generate plots

    # loading testing data -> I am using one of the 5 folds
    testing_data = None

    with open(params['test_data'][params['test_fold'] - 1], "rb") as file:
        testing_data = pickle.load(file)

    # load targets
    data_t = {}
    for t in params["times"]:
        data_t[f"time{t}"] = np.load(
            f'../training/{params["data_path"]}/{t}/data.npy', allow_pickle=True
        ).item()
    target_test = np.concatenate(
        [data_t[f"time{t}"][f"targ_test{params['test_fold']}"] for t in params["times"]], axis=0
    )
    testing_data["targets"] = target_test[:-1]
    del data_t

    # converting to torch tensors
    n_sim = testing_data["y_pred_r0.0"].shape[0]
    x = torch.tensor(
        np.concatenate(
            [
                testing_data[f"y_pred_r{res}"].reshape(n_sim, -1)
                for res in params['testing_resolutions']
            ]
        ),
        dtype=torch.float32,
    )
    theta = torch.tensor(
        np.concatenate([testing_data["targets"] for res in params['testing_resolutions']]),
        dtype=torch.float32,
    )

    #run sbc
    from sbi.diagnostics import run_sbc
    ranks, dap_samples = run_sbc(
        x, theta, posterior, num_posterior_samples=params['num_posterior_samples'], num_workers=12
    )

    #run checks on sbc results
    from sbi.diagnostics import check_sbc
    check_stats = check_sbc(ranks, theta, dap_samples, num_posterior_samples=params['num_posterior_samples'])
    wandb.log(check_stats)
   
    from sbi.analysis.plot import sbc_rank_plot
    f, ax = sbc_rank_plot(
        ranks=ranks,
        num_posterior_samples=params['num_posterior_samples'],
        plot_type="hist",
        num_bins=None,  # by passing None we use a heuristic for the number of bins.
    )
    wandb.log({"sbc_hist": plt})

    f, ax = sbc_rank_plot(ranks, num_posterior_samples=params['num_posterior_samples'], plot_type="cdf")
    wandb.log({"sbc_cdf": plt})

    #run tarp check
    ecp, alpha = run_tarp(
        theta,
        x,
        posterior,
        references=None,  # will be calculated automatically.
        num_posterior_samples=parms['num_posterior_samples'],
    )
    atc, ks_pval = check_tarp(ecp, alpha)
    wandb.log({'tarp_atc': atc, 'tarp_ks_pval': ks_pval})
    
    from sbi.analysis.plot import plot_tarp
    plot_tarp(ecp, alpha)
    wandb.log({"tarp_plot": plt})
    
