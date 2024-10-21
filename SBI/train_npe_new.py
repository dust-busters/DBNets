from arviz import output_file
import sbi
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import tarp
import wandb
import logging
import argparse
import time

sys.path.append("../training")
import pickle
from ctypes import util
from sbi.inference import NPE
from sbi import utils
import torch
from sbi.neural_nets import posterior_nn
from numpy import float32
import yaml
from sweep import global_params

__LABELS__ = [
    "Alpha",
    "AspectRatio",
    "InvStokes1",
    "FlaringIndex",
    "PlanetMass",
    "SigmaSlope",
]

logger = logging.getLogger(__name__)
logging.basicConfig(filename='lastlog.log', encoding='utf-8', level=logging.DEBUG)


def concat_dict(a, b):
    data = {}
    if a is None:
        return b.copy()
    if b is None:
        return a.copy()
    if (a is None) and (b is None):
        return None
    for key in b.keys():
        data[key] = np.concatenate([a[key], b[key]])

    return data

def init(params=global_params, sweep=True):
    
    #creating output folder
    logger.info('Creating output folder')
    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    out_folder = f"outputs/{params['run_name']}.{time.time()}"
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    print(f"Saving folder: {out_folder}")
    os.system(f"cp params.py {out_folder}/params.npy")

    wandbrun = None
    if not params['only_test']:
        wandbrun =  wandb.init(project="dbnets2.0.0_SBI", config=params)
        if sweep:
            wandb.config.update(params)
            params = wandb.config
    
    return wandbrun, params, out_folder
    
def load_data(params=global_params, sweep=True ):
    #loading data for training NPE models
    logger.info(f"Loading data for training NPE, except for folder {params['test_fold']}")
    all_data = None
    for fold, test_d in enumerate(params["test_data"]):
        if fold + 1 != params["test_fold"]:
            with open(test_d, "rb") as file:
                data = pickle.load(file)
                
            #some logging
            logger.debug(f"Opened {test_d}, data info:")
            for k in data.keys():
                logger.debug(f'data[{k}]: {data[k].shape}')
                
            if (params["method"] == "method2") or (params["method"]=="method3"):
                data["targets"] = data["y"][:,params["inf_para"]]
                all_data = concat_dict(all_data, data)
                print(f"shape of y_pred, fold {fold+1}: {data['y_pred'].shape}")
                
                #some logging
                logger.debug(f"all_data info after concatenate:")
                for k in all_data.keys():
                    logger.debug(f'data[{k}]: {data[k].shape}')
                    
            else:
                # load targets
                data_t = {}
                for t in params["times"]:
                    data_t[f"time{t}"] = np.load(
                        f'../training/{params["data_path"]}/{t}/data.npy', allow_pickle=True
                    ).item()
                target_test = np.concatenate(
                    [data_t[f"time{t}"][f"targ_test{fold+1}"] for t in params["times"]],
                    axis=0,
                )
                data["targets"] = target_test[:-1][:,params["inf_para"]]
                all_data = concat_dict(all_data, data)
                del data_t

        # now all_data contains the test data to be used for training the maf NPE


    # preparing data for traininf. COncatenating different resolutions.
    if params["method"] == "method1":
        n_sim = all_data["y_pred_r0.0"].shape[0]
        x = torch.tensor(
            np.concatenate(
                [
                    all_data[f"y_pred_r{res}"].reshape(n_sim, -1)
                    for res in params["training_resolutions"]
                ]
            ),
            dtype=torch.float32,
        )
        theta = torch.tensor(
            np.concatenate([all_data["targets"] for res in params["training_resolutions"]]),
            dtype=torch.float32,
        )
    elif params["method"] == "method2":
        n_sim = all_data["y_pred"].shape[0]
        print(all_data['y_pred'].shape)
        print(f'n_sim: {n_sim}, n_features: {params["npe_features"]}')
        random_indices = np.random.choice(
            all_data["y_pred"].shape[1], size=(n_sim, params["npe_features"]), replace=True
        )
        features = all_data[f"y_pred"][
            np.arange(n_sim)[:, None], random_indices, :
        ].reshape(n_sim, -1)
        if params["concat_res"]:
            features = np.concatenate([features, all_data["sigma"].reshape(-1, 1)], axis=1)
        x = torch.tensor(
            features,
            dtype=torch.float32,
        )
        theta = torch.tensor(
            all_data["targets"],
            dtype=torch.float32,
        )
    elif params['method']=="method3":
        n_sim = all_data["y_pred"].shape[0]
        print(all_data['y_pred'].shape)
        print(f'n_sim: {n_sim}, n_features: {params["npe_features"]}')
        features = all_data['y_pred']
        x = torch.tensor(
            features,
            dtype=torch.float32,
        )
        theta = torch.tensor(
            all_data["targets"],
            dtype=torch.float32,
        )
        
    return x, theta


def train_sbi(x, theta, out_folder, params=global_params, sweep=True):
    __LABELS__ = __LABELS__[params['inf_para']]
    n_inf_para = len(params['inf_para'])
    out = []
    posteriors = []
    
    #train the single posteriors
    for i in range(n_inf_para):
        out.append([i])
    
    #train the pair posteriors
    for i in range(n_inf_para-1):
        for j in range(i+1,n_inf_para):
            out.append([i, j])
    
    #train the 3 params posteriors
    for i in range(n_inf_para-2):
        for j in range(i+1,n_inf_para-1):
            for k in range(j+1,n_inf_para):
                out.append([i, j, k])
    
    #train the 4 params posterior
    out.append([0,1,2,3])
    
    for out_list in out:
        outfold_name = f'{out_folder}/p.'
        for i in out_list:
            outfold_name = outfold_name + f'{i}.'
        outfold_name = outfold_name + 'pkl'
        res = train_single_sbi(x, theta=theta[:, out_list], outfold=outfold_name, params=params, sweep=sweep)
        posteriors.append(res)
    
    return out, posteriors

def train_single_sbi(x, theta, outfold, params=global_params, sweep=True):
        
    # training
    # preparing NPE

    print("Data loaded. Starting training of the NPE.")
    prior = None
    density_estimator_funct = posterior_nn(model=params['density_estimator'], hidden_features=params['hidden_features'], num_transforms=params['n_transforms'], num_bins=params['n_bins'])
    inference = NPE(prior=None, density_estimator=density_estimator_funct,device=params['device'])
    
    _ = inference.append_simulations(theta, x, proposal=None)
    post = inference.train(
        training_batch_size=params["train_batch_size"],
        stop_after_epochs=params["min_train_epochs"],
        learning_rate=params["learning_rate"],
        show_train_summary=True,
        force_first_round_loss=True,
        validation_fraction=0.1
    )
    posterior = inference.build_posterior()

    with open(outfold, "wb") as f:
        pickle.dump(posterior, f)
        
    return posterior
        
def test(params, posteriors):
    # run tests, compute metrics and generate plots

    # loading testing data -> I am using one of the 5 folds
    testing_data = None

    with open(params["test_data"][params["test_fold"] - 1], "rb") as file:
        testing_data = pickle.load(file)

    if params['method']=='method1':
        # load targets
        data_t = {}
        for t in params["times"]:
            data_t[f"time{t}"] = np.load(
                f'../training/{params["data_path"]}/{t}/data.npy', allow_pickle=True
            ).item()
        target_test = np.concatenate(
            [
                data_t[f"time{t}"][f"targ_test{params['test_fold']}"]
                for t in params["times"]
            ],
            axis=0,
        )
        testing_data["targets"] = target_test[:-1][:,params["inf_para"]]
        del data_t

        # converting to torch tensors
        n_sim = testing_data["y_pred_r0.0"].shape[0]
        x = torch.tensor(
            np.concatenate(
                [
                    testing_data[f"y_pred_r{res}"].reshape(n_sim, -1)
                    for res in params["testing_resolutions"]
                ]
            ),
            dtype=torch.float32,
        )
        theta = torch.tensor(
            np.concatenate(
                [testing_data["targets"] for res in params["testing_resolutions"]]
            ),
            dtype=torch.float32,
        )
    elif params['method']=='method2':
        
        for k in testing_data.keys():
            testing_data[k] = testing_data[k][::30]
        
        n_sim = testing_data["y_pred"].shape[0]
        print(f'n_sim: {n_sim}, n_features: {params["npe_features"]}')
        random_indices = np.random.choice(
            testing_data["y_pred"].shape[1], size=(n_sim, params["npe_features"]), replace=True
        )
        features = testing_data[f"y_pred"][
            np.arange(n_sim)[:, None], random_indices, :
        ].reshape(n_sim, -1)
        if params["concat_res"]:
            features = np.concatenate([features, testing_data["sigma"].reshape(-1, 1)], axis=1)
        x = torch.tensor(
            features,
            dtype=torch.float32,
            device=params['device']
        )
        theta = torch.tensor(
            testing_data["y"][:,params["inf_para"]],
            dtype=torch.float32,
            device=params['device']
        )
    elif params['method']=='method3':
        
        for k in testing_data.keys():
            testing_data[k] = testing_data[k][::30]
        
        n_sim = testing_data["y_pred"].shape[0]
        print(f'n_sim: {n_sim}, n_features: {params["npe_features"]}')
        features = testing_data[f"y_pred"]
        x = torch.tensor(
            features,
            dtype=torch.float32,
            device=params['device']
        )
        theta = torch.tensor(
            testing_data["y"][:,params["inf_para"]],
            dtype=torch.float32,
            device=params['device']
        )

    #extract samples and test accuracy
    coll_samples = np.array([])
    for i in range(x.shape[0]):
        samples = posterior.set_default_x(x[i]).sample((params['num_posterior_samples'],)).cpu().numpy()
        if i == 0:
            coll_samples = samples.copy().reshape(1, *samples.shape)
        else:
            coll_samples = np.concatenate([coll_samples, samples.copy().reshape(1, *samples.shape)])

    #compute mse of medians 
    medians = np.median(coll_samples, axis=1, keepdims=True)
    mses = ((theta-medians[:,0,:])**2).mean(axis=(0), keepdims=False)
    stds = coll_samples.std(axis=1, keepdims=False).mean(axis=0, keepdims=False)
    for j,i in enumerate(params['inf_para']):
        wandb.log({f'mse_{__LABELS__[i]}': mses[j], f'std_{__LABELS__[i]}': stds[j]})

    #compute standard deviations for each dimension

    # run sbc
    from sbi.diagnostics import run_sbc

    print(x.shape)
    print(theta.shape)
    ranks, dap_samples = run_sbc(
        theta,
        x,
        posterior,
        num_posterior_samples=params["num_posterior_samples"],
        num_workers=12,
    )

    # run checks on sbc results
    from sbi.diagnostics import check_sbc

    check_stats = check_sbc(
        ranks, theta, dap_samples, num_posterior_samples=params["num_posterior_samples"]
    )
    wandb.log(check_stats)

    from sbi.analysis.plot import sbc_rank_plot

    f, ax = sbc_rank_plot(
        ranks=ranks,
        num_posterior_samples=params["num_posterior_samples"],
        plot_type="hist",
        num_bins=None,  # by passing None we use a heuristic for the number of bins.
    )
    wandb.log({"sbc_hist": wandb.Image(f)})

    f, ax = sbc_rank_plot(
        ranks, num_posterior_samples=params["num_posterior_samples"], plot_type="cdf"
    )
    wandb.log({"sbc_cdf": wandb.Image(f)})

    # run tarp checks
    from sbi.diagnostics import run_tarp
    from sbi.diagnostics import check_tarp

    ecp, alpha = run_tarp(
        theta,
        x,
        posterior,
        references=None,  # will be calculated automatically.
        num_posterior_samples=params["num_posterior_samples"],
    )
    atc, ks_pval = check_tarp(ecp, alpha)
    wandb.log({"tarp_atc": atc, "tarp_ks_pval": ks_pval})

    from sbi.analysis.plot import plot_tarp

    plot_tarp(ecp, alpha)
    wandb.log({"tarp_plot": wandb.Image(plt.gcf())})

    #run a separate tarp for each parameter
    fig, axs = plt.subplots(2,3, figsize=(12,8), sharex=True, sharey=True)
    axs = axs.flatten()
    for i, ax in enumerate(axs[:len(params['inf_para'])]):
        samples = np.swapaxes(coll_samples, 0, 1)[:,:,i:i+1]
        ecp, alpha = tarp.get_tarp_coverage(samples, theta[:,i:i+1].numpy())
        ax.plot(alpha, ecp)
        ax.set_title(__LABELS__[params['inf_para'][i]])
        ax.plot([0,1], [0,1], '--', color='gray')
        if i%3==0:
            ax.set_ylabel('Coverage')
        if i >2:
            ax.set_xlabel('Credibility Level')
        atc, ks_pval = check_tarp(torch.tensor(ecp), torch.tensor(alpha))
        wandb.log({f"tarp_atc_{__LABELS__[params['inf_para'][i]]}": atc, f"tarp_ks_pval_{__LABELS__[params['inf_para'][i]]}": ks_pval})
    axs[0].legend(title='Testing\nresolution', loc='upper left')
    wandb.log({f'sing_tarp': wandb.Image(fig)})

def train_pipeline(params, sweep=False):
    wandbrun, params, out_folder = init(params=params, sweep=False)
    x, theta = load_data(params=params, sweep=False, wandbrun=wandbrun, out_folder=out_folder)
    result = train_sbi(x, theta, out_folder, params=params, sweep=True)
    
if __name__ == '__main__':
    project_name='dbnets2.0.0_SBI'
    
    parser = argparse.ArgumentParser(
        prog="Wandb sweep launcher",
        description="Launches a sweep on slurm using wandb",
    )

    parser.add_argument("--sweep", required=False, default=None, type=str)
    args = parser.parse_args()
    
    if args.sweep is not None:
        print(f"Launching sweep agents performing {1} iterations...")
        print(f"Sweep id: {args.sweep}")
        wandb.agent(args.sweep, train_pipeline, count=1, project=project_name)
    else:
        from params_new import params
        print("Starting training using parameters in configs.py...")
        train_pipeline(params=params, sweep=False)