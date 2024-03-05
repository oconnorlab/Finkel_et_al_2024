
import numpy as np
import pandas as pd
import matplotlib.pylab as mpl
import matplotlib.patches as patches
from matplotlib import gridspec
import scipy

def plot_zscore_psth(data, xlims, bin_size, colors, labels = None):
    mpl.close('all')

    num_subplots = len(list(data.keys()))
    fig = mpl.figure(figsize = (17,4))

    gs1 = gridspec.GridSpec(1,num_subplots)
    gs2 = gridspec.GridSpec(1,num_subplots)

    gs1.update(bottom=0.25, top=0.8, left = 0.15, right = 0.9, wspace=0.2)
    gs2.update(bottom = 0.81, top=0.87, left = 0.15, right = 0.9, wspace=0.2)

    axs = [mpl.subplot(gs1[0, 0])]
    for num in range(1,num_subplots):
        axs.append(mpl.subplot(gs1[0, num],sharey=axs[0]))

    xvals = np.arange(xlims[0],xlims[1],bin_size)
    ## -1 is the first timepoint for all trials.
    ## need to subtract 1 bin from both ends to shift window back 
    ## 1 time bin because np.histogram counts spikes occuring from and including
    ## the "bin label" up to but not including the label for bin+1. 
    ## Otherwise it would look like activity starts before the stimulus onset
    ## for a stim responseive unit.
    xrange = np.array([abs(-1 - xvals[0]), abs(1+xvals[-1])])
    xrange = (xrange-bin_size)/bin_size
    inds = np.arange(int(round(xrange[0])),int(round(xrange[1])),1)

    for i, col in enumerate(list(data.keys())):
        trial_types = data[col]
        ax = mpl.subplot(gs2[0, i],sharex=axs[i])
        ax.add_patch(patches.Rectangle((0,0), 0.15, 0.25, facecolor = 'k'))
        ax.set_ylim(0,2)
        ax.axis('off')

        for tt_num, tt in enumerate(trial_types):
            z_map = tt[inds]
            m_psth = np.nanmean(z_map,axis = 0)

            sem_psth = scipy.stats.sem(z_map, axis = 0, nan_policy = 'omit')
            axs[i].plot(xvals[:-1],m_psth, color = colors[tt_num], linewidth = .5)
            axs[i].fill_between(xvals[0:-1], m_psth-sem_psth, m_psth+sem_psth,alpha=0.6,color = colors[tt_num],)
            axs[i].set_xticks(np.arange(-0.2,1.2,0.2))
            axs[i].set_xlim(xvals[0],xvals[-1])
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['top'].set_visible(False)
            axs[i].set_xlabel('Time from stim onset (s)')

    axs[0].set_ylabel('Mean Z-score')
    axs[1].set_ylim(-.5,2.5)
    axs[0].set_ylim(-.5,2.5)

    for j,(label,c)in enumerate(zip(labels, colors)):
        axs[-1].text(.15, .85-.07*j, label , transform=axs[i].transAxes, color = c, ha = 'left')
    ylim_hit_psth = axs[0].get_ylim()
    return fig

def create_heatmap(dataframe, columns, colors, titles, xlim, zlim, bin_size):
    
    fig = mpl.figure(figsize = (16,8))
    gs1 = gridspec.GridSpec(1,4)
    gs2 = gridspec.GridSpec(1,1)

    gs1.update(bottom=0.14, top=0.92, left = 0.15, right = 0.9, hspace=0.1, wspace=0.05)
    gs2.update(bottom=0.14, top=0.92, left = 0.91, right = 0.94, hspace=0.1, wspace=0.3)

    ax1 = mpl.subplot(gs1[0, 0])
    ax2 = mpl.subplot(gs1[0, 1], sharex = ax1)
    ax3 = mpl.subplot(gs1[0, 2], sharex = ax1)
    ax4 = mpl.subplot(gs1[0, 3], sharex = ax1)
    ax5 = mpl.subplot(gs2[0, 0])

    z_max = zlim[1]
    z_min= zlim[0]
    ## -1 is the first timepoint for all trials.
	## need to subtract 1 bin from both ends to shift window back 
	## 1 time bin because np.histogram counts spikes occuring from and including
	## the "bin label" up to but not including the label for bin+1. 
	## Otherwise it would look like activity starts before the stimulus onset
	## for a stim responseive unit.
    xmin = (xlim[0]+1-.025)/0.025 
    xmax = (xlim[1]+1-.025)/0.025 

    Col1 = dataframe[columns[0]].iloc[::-1].loc[:,xmin:xmax]
    Col2 = dataframe[columns[1]].iloc[::-1].loc[:,xmin:xmax]
    Col3 = dataframe[columns[2]].iloc[::-1].loc[:,xmin:xmax]
    Col4 = dataframe[columns[3]].iloc[::-1].loc[:,xmin:xmax]

    cols = [Col1, Col2, Col3, Col4]   #, Col3]

    axs = [ax1,ax2,ax3,ax4]
    heatmaps = []
    ticks = np.arange(0, Col1.shape[1], 0.25/bin_size)
    labels = np.arange(xlim[0],xlim[1], 0.25)
    for col_num in range(4):
        heatmaps.append(axs[col_num].imshow(np.flipud(cols[col_num]), cmap = colors, vmax=zlim[1], 
                                            vmin=zlim[0], aspect = 'auto'))
        axs[col_num].set_xticks(ticks)
        axs[col_num].set_xticklabels(labels)
        axs[col_num].set_title(titles[col_num])
        axs[col_num].set_xlabel('Time from\nstim onset (s)')
        axs[col_num].set_ylabel('Unit')
        axs[col_num].invert_yaxis()
        axs[col_num].plot([np.abs(xlim[0])/0.025,np.abs(xlim[0])/0.025], [0,axs[col_num].get_ylim()[1]], ':w', linewidth = 2)
        axs[col_num].plot([(np.abs(xlim[0])+0.15)/0.025,(np.abs(xlim[0])+0.15)/0.025], [0,axs[col_num].get_ylim()[1]], ':w', 
                          linewidth = 2)

        if col_num > 0:
            axs[col_num].set_yticks([])
            axs[col_num].set_ylabel('')
    
    mpl.colorbar(heatmaps[-1], cax = ax5)
    ax5.set_ylabel('Z-score')
    
    return fig