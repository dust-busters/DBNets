import sbi
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import wandb

sys.path.append("../training")
from train_multip import __LABELS__
from params_val import params
import pickle
from ctypes import util
from sbi.inference import NPE
from sbi import utils
import torch
from numpy import float32
from sbi import analysis
from sbi.diagnostics import run_sbc
import yaml

if not os.path.exists("outputs"):
    os.mkdir("outputs")
out_folder = f"outputs/{params['run_name']}_val"
if not os.path.exists(out_folder):
    os.mkdir(out_folder)
    
#load validation data
with open('../test/validation/val.pkl', 'rb') as file:
    val_data = pickle.load(file)
    
with open('val_data.pkl', 'rb') as file:
    dict_targ = pickle.load(file)
    
#load posterior
with open(f"outputs/{params['run_name']}/posterior.pkl", 'rb') as f:
    posterior = pickle.load(f)
    
with wandb.init(project='dbnets2.0.0_SBI', config=params, tags=['validation']):
    for res in [0, 0.05, 0.15, 0.2]:
        x = torch.tensor(val_data['y_pred_r{res}'].reshape(val_data['y_pred_r{res}'].shape[0], -1), dtype=torch.float32)
        targets = torch.tensor(dict_targ['val_targ'])[:-1]

        if not os.path.exists(f'{out_folder}/cplots_{res}/'):
            os.mkdir(f'{out_folder}/cplots_{res}/')
        coll_samples = []
        for i in range(len(x)):
            # generate posterior samples
            true_parameter = targets[i]
            x_observed = x[i]
            samples = posterior.set_default_x(x[i]).sample((params['n_samples'],))
            
            if i==0:
                coll_samples = samples.numpy().reshape(1, *samples.shape)
            else:
                coll_samples = np.concatenate([coll_samples, samples.numpy().reshape(1, *samples.shape)], axis=0)
            fig, ax = analysis.pairplot(
                samples,
                points=true_parameter,
                labels=__LABELS__,
                limits=[[-1, 1], [-1,1], [-1,1], [-1,1], [-1,1], [-1,1]],
                points_colors="r",
                points_offdiag={"markersize": 6},
                figsize=(10,10),
            )
            fig.savefig(f'f'{out_folder}/cplots_{res}/{i}.png', dpi=500)
            plt.close()
            
        #run SBC

        ranks, dap_samples = run_sbc(targets, x, posterior, num_posterior_samples=params['n_samples'], num_workers=12)
        #run checks on sbc results
        from sbi.diagnostics import check_sbc
        check_stats = check_sbc(ranks, targets, dap_samples, num_posterior_samples=params['n_samples'])
        wandb.log(check_stats, step=res)
    
        from sbi.analysis.plot import sbc_rank_plot
        f, ax = sbc_rank_plot(
            ranks=ranks,
            num_posterior_samples=params['n_samples'],
            plot_type="hist",
            num_bins=None,  # by passing None we use a heuristic for the number of bins.
        )
        wandb.log({"sbc_hist": wandb.Image(f)}, step=res)
        f, ax = sbc_rank_plot(ranks, num_posterior_samples=params['n_samples'], plot_type="cdf")
        wandb.log({"sbc_cdf": wandb.Image(f)}, step=res)
        
        #run TARP
        from sbi.diagnostics import run_tarp
        from sbi.diagnostics import check_tarp
        ecp, alpha = run_tarp(
            targets,
            x,
            posterior,
            references=None,  # will be calculated automatically.
            num_posterior_samples=params['n_samples'],
        )
        atc, ks_pval = check_tarp(ecp, alpha)
        wandb.log({'tarp_atc': atc, 'tarp_ks_pval': ks_pval}, step=res)
        
        from sbi.analysis.plot import plot_tarp
        plot_tarp(ecp, alpha)
        wandb.log({"tarp_plot": wandb.Image(plt.gcf())}, step=res)
        