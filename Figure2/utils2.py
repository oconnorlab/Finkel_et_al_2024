import scipy as sp
import scipy.io
import os
import numpy as np
import pandas as pd
import glob
import csv
import random as rand
from tqdm import tnrange, tqdm_notebook
from collections import Iterable
import matplotlib.pylab as mpl
import random as rand
from ipywidgets import *
import colorlover as cl
from scipy import stats
import matplotlib.patches as patches
from matplotlib import gridspec

def load_data(directory):
    """
    function that loads, cleans, and performs some initial processing on log_df table that contains data from entire dataset.
    log_df table was generated from Intan recording data that was originally preprocessed in matlab using
    cat_session.mat function
    """

    log_df = pd.read_hdf(directory+ '\\log_df.h5', 'table')
    log_df['stim_onset'] = log_df['stim_onset'].fillna(0)
    log_df['spike_times(stim_aligned)'] = log_df['spike_times'] - log_df['stim_onset']
    log_df = log_df[~log_df['trial_type'].str.contains('NoStim')]
    licks = pd.concat([log_df['licks_right'] - log_df['stim_onset'] , log_df['licks_left']-log_df['stim_onset']], axis=1)
    licks = licks.applymap(lambda y: y[[0.1<y]] if len(y) > 0 else y)
    licks = licks.applymap(lambda y: y[[3>=y]] if len(y) > 0 else y)
    first_licks = licks.applymap(lambda y: min(y) if len(y) > 0 else np.nan)
    last_licks = licks.applymap(lambda y: max(y) if len(y) > 0 else np.nan)

    log_df['first_lick'] = first_licks.min(axis=1)
    log_df['last_lick'] = last_licks.max(axis=1)

    log_df['spike_times(stim_aligned)'] = log_df['spike_times(stim_aligned)'].apply(lambda x: np.concatenate(x) 
                                                                                       if len(list(x)) > 0 else x)

    log_df = log_df.sort_values(['mouse_name', 'date', 'cluster_name', 'first_lick'], ascending = [1,1,1,1])
    log_df['identified'] = 'unidentified'
    log_df = log_df.reset_index(drop=True)

    log_df['correct'] = 0

    # fwd contingencies - correct responses
    log_df.loc[~(log_df['mouse_name'].isin(['EF0091', 'EF0099', 'EF0101', 'EF0102'])) & (log_df['block_type'] == 'Whisker') &
           (log_df['trial_type'].str.contains('Som')) & (log_df['response'] == 1), 'correct'] = 1
    log_df.loc[~(log_df['mouse_name'].isin(['EF0091', 'EF0099', 'EF0101', 'EF0102'])) & (log_df['block_type'] == 'Visual') &
           (log_df['trial_type'].str.contains('Vis')) & (log_df['response'] == 2), 'correct'] = 1

    #rev contingencies - correct responses
    log_df.loc[(log_df['mouse_name'].isin(['EF0091', 'EF0099', 'EF0101', 'EF0102'])) & (log_df['block_type'] == 'Whisker') &
           (log_df['trial_type'].str.contains('Som')) & (log_df['response'] == 2), 'correct'] = 1
    log_df.loc[(log_df['mouse_name'].isin(['EF0091', 'EF0099', 'EF0101', 'EF0102'])) & (log_df['block_type'] == 'Visual') &
           (log_df['trial_type'].str.contains('Vis')) & (log_df['response'] == 1), 'correct'] = 1

    #correct rejections
    log_df.loc[(log_df['block_type'] == 'Whisker') & (log_df['trial_type'].str.contains('Vis'))
               & (log_df['response'] == 0), 'correct'] = 1
    log_df.loc[(log_df['block_type'] == 'Visual') & (log_df['trial_type'].str.contains('Som'))
               & (log_df['response'] == 0), 'correct'] = 1


    log_df['uni_id'] = (log_df['mouse_name'].apply(lambda x: x[-3:]) + log_df['date']+
                             log_df['cluster_name'].apply(lambda x: x[2] + x[-2:]))
    unit_key_df = log_df[['uni_id', 'mouse_name', 'date', 'cluster_name']].drop_duplicates().reset_index(drop = True)
    return log_df, unit_key_df

def filt_motion_trials(log_df, data_direc):
    """
    takes exclude_fn df and uses contents to filter out rows with high motion artifact
    in main log_df file
    """

    mat = sp.io.loadmat(data_direc + '\\trialsToExclude')
    log = mat['trialsToExclude']
    indv_log_df = pd.DataFrame(log, columns = ['mouse_name', 'date', 'trial_num'])

    for col in [0,1,2,2]:
        indv_log_df.ix[:,col] = indv_log_df.ix[:,col].str[0]

    log_df = log_df.reset_index()
    rows_to_exclude = pd.merge(log_df, indv_log_df, how='inner')
    inds_to_exclude = rows_to_exclude['index'].as_matrix()
    log_df = log_df.drop(inds_to_exclude).reset_index(drop=True)
    log_df.drop('index', axis = 1, inplace = True)
    

    return log_df

def chunk_trials(log_df):
    subset_dict = {}
    size_dict = {}
    subset_dict['None'] = 0
    size_dict['None'] = 0

    categories = np.concatenate([log_df['mouse_name'].unique(), log_df['identified'].unique()])

    for cat in categories:
            subset = log_df[log_df['mouse_name'] == cat]
            if subset.size == 0:
                subset = log_df[log_df['identified'] == cat]
                print(cat)
            subset_dict[cat] = subset
            unique_units = subset[['mouse_name', 'date', 'cluster_name']].drop_duplicates()
            size_dict[cat] = len(unique_units)
            #print(unique_units.size)
    return subset_dict

def plot_rasters(T_rasters, V_rasters, modality, window, bin_size, ylim_r = None, ylim_p = None):
    
    fig = mpl.figure(figsize=(4, 3.5))
    first_raster = T_rasters[0]
    gs1 = gridspec.GridSpec(1,1)
    gs2 = gridspec.GridSpec(1,1)
    gs3 = gridspec.GridSpec(1,1)
    gs4 = gridspec.GridSpec(1,1)
    gs1.update(bottom = 0.88, top=0.95, left = 0.2, right = 0.83)
    gs3.update(bottom=0.15, top=0.41, left = 0.2, right = 0.83)
    gs2.update(bottom=0.45, top=0.88, left = 0.2, right = 0.83)
    gs4.update(bottom=0.45, top=0.88, left = 0.83, right = 0.9)
    
    ax1 = mpl.subplot(gs1[0, 0])
    ax2 = mpl.subplot(gs2[0, 0])
    ax3 = mpl.subplot(gs3[0, 0])
    patch_ax = mpl.subplot(gs4[0, 0], sharey = ax2)

    trial_type = 0
    trial_total = 1
    hists = []
    colors= ['m', 'C3', 'C7', 'C2']
    
    if modality == 'Visual':
        rasters = V_rasters
        blocks =['Visual\nblock', 'Touch\nblock']
        ax1.add_patch(patches.Rectangle((0,0), 0.15, 1, facecolor = 'C1', alpha = 0.5))
        block_colors = ['C1', 'C0']
    else:
        rasters = T_rasters
        blocks =['Touch\nblock','Visual\nblock']
        ax1.add_patch(patches.Rectangle((0,0), 0.15, 1, facecolor = 'C0', alpha = 0.5))
        block_colors = ['C0', 'C1']
    block_lims = []
    for i in range(len(rasters)):
        ras = rasters[trial_type]
        spike_counts = []
        for trial, spike in enumerate(ras['spike_times(stim_aligned)']):
            spike = spike[(spike>window[0]) & (spike<=window[1])]
            ax2.vlines(spike, trial + trial_total - .5, trial + trial_total +.5)
            ax2.vlines(ras.iloc[trial]['first_lick'], trial + trial_total - .5, trial +
                       trial_total + .5, color = colors[i], linewidth = 5)
            
            spike = spike[(spike>window[0]) & (spike<=window[1])]
            edges = np.arange(window[0], window[1]+bin_size*2, bin_size)
            count, _ = np.histogram(spike,edges)
            spike_counts.append(count)

        patch_ax.add_patch(patches.Rectangle((0,trial_total -.5), 1,
                                       trial+1, facecolor = colors[trial_type], alpha = 0.5))
        if trial_type in [0,2]:
            patch_ax.plot([0,2],[trial_total-.5, trial_total-.5], '--k')
            block_lims.append(trial_total)
        elif trial_type == 3:
            patch_ax.plot([0,2],[trial_total+ trial, trial_total+trial], '--k')
            block_lims.append(trial_total+ trial+1)

        trial_total = trial_total + trial +1
        trial_type += 1
        average_hist = np.convolve(np.mean(spike_counts, axis=0)/bin_size, [1/3]*3, 'same')
        SE_hist = np.convolve(stats.sem(spike_counts)/bin_size, [1/3]*3, 'same')
        
        ax3.plot(edges[0:-1], average_hist, color = colors[i])
        ax3.fill_between(edges[0:-1], average_hist-SE_hist, average_hist+SE_hist, alpha = 0.5, color = colors[i])
    
    if modality == 'Touch':
        text_loc = 0.7
    else:
        text_loc = 0.15
    ax3.text(text_loc, .54, "CR", transform=ax3.transAxes, color = colors[0])
    ax3.text(text_loc, .67, "FA", transform=ax3.transAxes, color = colors[1])
    ax3.text(text_loc, .80, "Misses", transform=ax3.transAxes, color = colors[2])
    ax3.text(text_loc, .94, "Hits", transform=ax3.transAxes, color = colors[3])

    for ax in [ax1,ax2,ax3, patch_ax]:
        ax.set_xlim(window[0],window[1]-bin_size)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        
    ax1.axis('off')
    ax1.set_ylim(0,2)

    ax2.spines['bottom'].set_visible(False)
    ax2.set_ylabel('Trials')    
    ax2.axes.get_xaxis().set_ticks([])
    ax2.set_ylim(-1, trial_total+.5)

    ax3.set_xlabel('Time(s)')
    ax3.set_ylabel('Firing\nrate (Hz)') 
    
    og_ylim = ax2.get_ylim()
    if ylim_r != None:
        ax2.set_ylim(ylim_r)
        ax2.spines['left'].set_bounds(0, trial_total)
        ax2.set_yticks(np.arange(0, og_ylim[1], 20))
    if ylim_p != None:
        ax3.set_ylim(ylim_p)

    patch_ax.axis('off')
    patch_ax.set_xlim(0,2)
    patch_ax.text(1.2, (block_lims[1] - block_lims[0])/2.5, blocks[1], color = block_colors[1])
    patch_ax.text(1.2,  (block_lims[2] - block_lims[1])/2.5 + block_lims[1], blocks[0],  color = block_colors[0])
    
    return fig

def plot_unit(df_dict, mouse, n, x_min, x_max, modality = 'Touch', ylim=None, bin_size = 0.025):

        df = df_dict[mouse]
        #df = pd.DataFrame(df[0], index = [)
        ind_units = df[['mouse_name', 'date', 'cluster_name']].drop_duplicates()


        mouse = df['mouse_name'] == ind_units.iloc[n,0]
        date =  df['date'] == ind_units.iloc[n,1]
        cluster_name = df['cluster_name'] == ind_units.iloc[n,2]
        current_cell = df[mouse & date & cluster_name]


        cell_TTH = current_cell[(current_cell['block_type'] == 'Whisker') &
                               (current_cell['trial_type'] == 'Stim_Som_NoCue')&
                              (current_cell['correct'] == 1)]
        cell_TTM = current_cell[(current_cell['block_type'] == 'Whisker') &
                               (current_cell['trial_type'] == 'Stim_Som_NoCue')&
                              (current_cell['correct'] == 0)]
        cell_VTFA = current_cell[(current_cell['block_type'] == 'Visual') &
                       (current_cell['trial_type'] == 'Stim_Som_NoCue')&
                      (current_cell['correct'] == 0)]
        cell_VTCR = current_cell[(current_cell['block_type'] == 'Visual') &
                       (current_cell['trial_type'] == 'Stim_Som_NoCue')&
                      (current_cell['correct'] == 1)]
        
        cell_VVH = current_cell[(current_cell['block_type'] == 'Visual') &
                               (current_cell['trial_type'] == 'Stim_Vis_NoCue')&
                              (current_cell['correct'] == 1)]
        cell_VVM = current_cell[(current_cell['block_type'] == 'Visual') &
                               (current_cell['trial_type'] == 'Stim_Vis_NoCue')&
                              (current_cell['correct'] == 0)]
        cell_TVFA = current_cell[(current_cell['block_type'] == 'Whisker') &
                       (current_cell['trial_type'] == 'Stim_Vis_NoCue')&
                      (current_cell['correct'] == 0)]
        cell_TVCR = current_cell[(current_cell['block_type'] == 'Whisker') &
                       (current_cell['trial_type'] == 'Stim_Vis_NoCue')&
                      (current_cell['correct'] == 1)]

        t_rasters = [cell_TTH, cell_TTM, cell_VTFA, cell_VTCR][::-1]
        v_rasters = [cell_VVH, cell_VVM, cell_TVFA, cell_TVCR][::-1]
        
        max_ylim_r = max([pd.concat(t_rasters, axis = 0).shape[0], pd.concat(v_rasters, axis = 0).shape[0]])
        fig = plot_rasters(t_rasters, v_rasters, modality, [x_min, x_max], bin_size, ylim_r = (0,max_ylim_r), ylim_p = ylim)

        return fig