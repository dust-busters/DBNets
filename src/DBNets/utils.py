
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../training")

def to_real(data, inv_st=True, log=True):
    nshape = len(data.shape) - 1
    mins_r = np.array([-4, 0.03, 1, -5]).reshape(*np.ones(nshape).astype(int), 4)
    maxs_r = np.array([-2, 0.1, 3, -2]).reshape(*np.ones(nshape).astype(int), 4)
    rdata = (data + 1) * 0.5 * (maxs_r - mins_r) + mins_r
    
    rsh = rdata.shape
    rdata = rdata.reshape(-1,4)
    if inv_st:
        rdata[:,2] = -rdata[:,2]
    
    if not log:
        for i in [0,2,3]:
            rdata[:, i] = 10**rdata[:,i]
            
    return rdata.reshape(*rsh)

def plot_corners(final_samples_nonorm, name, starmass=None, savepath=None, image=None):
    if len(final_samples_nonorm.shape)==2:
        plot_corner(final_samples_nonorm, name, starmass, savepath, image)
    elif len(final_samples_nonorm.shape)==3:
        for i in range(final_samples_nonorm.shape[0]):
            plot_corner(final_samples_nonorm[0], name[0], starmass[0], savepath[0], image[0])
    
def plot_corner(final_samples_nonorm, name, starmass=None, savepath=None, image=None):
    
    if starmass==None:
        star_mass = 1.
    else:
        star_mass = starmass
        
    final_samples_real = final_samples_nonorm
    
    for i in [0,2,3]:
        final_samples_real[:, i] = 10**final_samples_real[:, i]
    final_samples_real[:,3] = final_samples_real[:,3]*star_mass*1047

    mins_r = np.array([1e-4, 0.03, 1e-3, 1e-2])
    maxs_r = np.array([1e-2, 0.1, 1e-1, 10])
    fig, axs = plt.subplots(4, 4)

    __NICELABELS__ = ['$\\alpha$', '$h_0$', '$St$', '$M_p$']

    if starmass==None:
        __NICELABELS__[-1] = '$M_p/M_\star$'
       
    mp_units = '$\\text{M}_J$' if starmass!=None else ' ' 
    
    medians = np.median(final_samples_real, axis=0)
    p16 = np.percentile(final_samples_real, 16, axis=0)
    p84 = np.percentile(final_samples_real, 84, axis=0)
    props = [
        f'log $\\alpha$: ${np.log10(medians[0]):.2f}^'+'{+'+f'{np.log10(p84[0])-np.log10(medians[0]):.2f}'+'}_{-'+f'{np.log10(medians[0])-np.log10(p16[0]):.2f}'+'}$',
        f'$h_0$: ${medians[1]:.2f}^'+'{+'+f'{p84[1]-medians[1]:.2f}'+'}_{-'+f'{medians[1]-p16[1]:.2f}'+'}$',
        f'log $St$: ${np.log10(medians[2]):.2f}^'+'{+'+f'{np.log10(p84[2])-np.log10(medians[2]):.2f}'+'}_{-'+f'{np.log10(medians[2])-np.log10(p16[2]):.2f}'+'}$',
        f'{__NICELABELS__[-1]}: ${medians[3]:.2f}^'+'{+'+f'{(p84[3]-medians[3]):.2f}'+'}_{-'+f'{(medians[3]-p16[3]):.2f}'+'}$ '+f'{mp_units}'
    ]

    for i in range(4):
        for j in range(4):
            if i > j:
                if j==1:
                    binsx = np.linspace(mins_r[j], maxs_r[j], 50)
                else:
                    binsx = np.logspace(np.log10(mins_r[j]), np.log10(maxs_r[j]), 50)
                if i==1:
                    binsy = np.linspace(mins_r[i], maxs_r[i], 50)
                else:
                    binsy = np.logspace(np.log10(mins_r[i]), np.log10(maxs_r[i]), 50)
                #computing the histogram for devs wrt median
                axs[i,j].hist2d(
                final_samples_real[:, j], final_samples_real[:, i], bins=(binsx, binsy), cmap='Grays'
                )


                if i!=1:
                    axs[i, j].set_yscale('log')

                if j!=1:
                    axs[i, j].set_xscale('log')

            if i < j:
                axs[i, j].axis("off")
            if i == j:
                axs[i,i].set_title(props[i])
                axs[i,j].axvline(medians[i], color='black')
                if p16[i]>mins_r[i]:
                    axs[i,j].axvline(p16[i], color='black', linestyle='dashed')
                if p84[i]<maxs_r[i]:
                    axs[i,j].axvline(p84[i], color='black', linestyle='dashed')
                if i!=1:
                    bins = np.logspace(np.log10(mins_r[i]), np.log10(maxs_r[i]), 50)
                    axs[i, j].hist(
                        final_samples_real[:, i], bins=bins, histtype="step", color="black", 
                    )
                    axs[i,j].set_xscale('log')
                else:
                    bins = np.linspace((mins_r[i]), (maxs_r[i]), 50)
                    axs[i, j].hist(
                        final_samples_real[:, i], bins=bins, histtype="step", color="black", 
                    )
            if j == 0:
                axs[i, j].set_ylabel(f'{__NICELABELS__[i]}')
            if i ==3:
                axs[i, j].set_xlabel(f'{__NICELABELS__[j]}')
            if j != 0:
                axs[i,j].set_yticklabels([])
            if i!=3:
                axs[i,j].set_xticklabels([])
            if i==0 and j==3:
                if image is not None:
                    axs[i,j].imshow(image[::-1,::], cmap='inferno')
                    axs[i,j].set_title(f"")
    #axs[1,3].text(0,0.5,np.concatenate(props))
    fig.set_size_inches(10, 10)

    if savepath is not None:
        fig.savefig(savepath, dpi=500)
    #opens the cnn for extracting summary statistics