#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:37:02 2024
@author: chloedziego

"""

#ERP Script for On-Task Pasithea Data
import mne
import numpy as np
import os.path as op
import glob
import pandas as pd
import matplotlib.pyplot as plt
import csv

from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs
from autoreject import AutoReject, get_rejection_threshold

#Make sure that plots are interactive
%matplotlib

#Define windows of interest, pre-processing parameters
baseline = None
reject = {'eeg': 150e-6,
          'eog': 250e-6}

start_trigger = 10001
end_trigger = 10002

#Create epochs around hearing audio information
#Epoching parameters
tmin, tmax = -2, 1
baseline = None
reject = dict(eeg=150e-6)
flat = dict(eeg=5e-6)

#Events/Triggers
event_id = {}
event_id['solution'] = 101

#Windows for exporting/averaging
windows = {
        "prestim":(-200, 0),
        "p200":(100,300),
        "n400" : (300, 500),
        "p600": (700, 900)
        }

#Create empty list for data to export
epochs_for_export = list()

#Set up headers for csv to log/count the number of epochs
with open('solution_epoch_count.csv', 'w', newline = '') as output:
    writer2 = csv.writer(output)
    writer2.writerow(['subject', 'solution'])
    
#Get the list of processed files list
processed_files = glob.glob('processed_EEG_files/*.gz')

#Testing out a single participant to make sure each individual section is working
#processed_files = glob.glob('processed_EEG_files/CT01_1_processed_raw.fif.gz')

#Creating Epoch files for Grand Averages/Averaging across Windows
#Begin looping through participant data
for p in processed_files:
	#Collect relevant information for exporting data
    s_number = op.split(p)[1][0:4]
    session_no = op.split(p)[1][5:6]
    epochs_file = op.join('solution_epochs/' + s_number + '_' + session_no +'-epo.fif.gz')

    print("reading preprocessed data " + p)
    #Read in the raw data file
    raw = mne.io.read_raw_fif(p, preload=True)
    
    #Extract events
    events = mne.events_from_annotations(raw)[0]
    events_vis = mne.viz.plot_events(events, sfreq=None, first_samp=0, color=None, event_id=None, axes=None, equal_spacing=True, show=True, on_missing='raise', verbose=None)
    events_vis.savefig('check_data/visualised_events/' + s_number + '-' + session_no + '_visualised_events_plot.png')
    plt.close('all')
    
    #Initialise new events array
    events2 = events.copy()
    epochs = mne.Epochs(raw, events2, event_id=event_id, tmin=tmin, tmax=tmax, proj=False,
                        baseline=baseline, detrend=0, reject_by_annotation=False,preload=True)

    #Apply the autoreject function to the epoched data
    ar = AutoReject(thresh_method='bayesian_optimization', random_state=42)
    ar.fit(epochs) 
    epochs_clean,reject_log = ar.transform(epochs,return_log=True)
    evoked_clean = epochs_clean.average()
    evoked = epochs.average()
    plt.close('all')
    
    #Visualise the data before and after autoreject application
    print("Plotting effects of AutoReject algorithm")
    ar_figure_name = 'check_data/autoreject/' + s_number + '_' + session_no + '_autoreject.png'
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    for ax in axes:
        ax.tick_params(axis='x', which='both', bottom='off', top='off')
        ax.tick_params(axis='y', which='both', left='off', right='off')
    ylim = dict(eeg=(-100, 100))
    evoked.pick_types(eeg=True, exclude=[])
    evoked.plot(exclude=[], axes=axes[0], ylim=ylim, show=False)
    axes[1].set_title('Before autoreject')
    evoked_clean.pick_types(eeg=True, exclude=[])
    evoked_clean.plot(exclude=[], axes=axes[1], ylim=ylim)
    axes[1].set_title('After autoreject')
    plt.tight_layout()
    plt.savefig(ar_figure_name)
    plt.close('all')
    
    scalings = dict(eeg=100e-6)
    
    #Visualise the dropped data and rejections
    #rejections = reject_log.plot_epochs(epochs, scalings=scalings)
    rejections_drop = epochs_clean.plot_drop_log(show=False)
    rejections_drop.savefig('check_data/rejections/' + s_number + '-' + session_no + '_rejections_drop.png')
    plt.close('all')
    
    #Generate a power spectral density plot from the EPOCH data
    print("Plotting subject's new psd plot following referencing, filtering and epoching")
    psd_cleaned = epochs_clean.plot_psd(fmax = 30)
    psd_cleaned.savefig('check_data/epoch_plots/' + s_number + '_' + session_no +'_cleaned_psd.png') 
    plt.close('all')
    
    #Save the epochs file for plotting/statistical analyses scripts later
    print("Save epochs")
    epochs.save(epochs_file, overwrite = True)

    #Now count the number of trials per participant following removal and save to csv
    solution = len(epochs_clean['solution'])
        
    with open('solution_epoch_count.csv', 'a', newline = '') as outcsv:
        # create the csv writer
        writer = csv.writer(outcsv)
        # write a row to the csv file
        writer.writerow([s_number, solution])
    
    #Create dataframe for plotting (with resample/writing dataframe)
    # Downsample to 100 Hz for plotting purposes
    print('Original sampling rate:', epochs.info['sfreq'], 'Hz')
    epochs_resampled = epochs.copy().resample(100, npad='auto')
    print('New sampling rate for plotting:', epochs_resampled.info['sfreq'], 'Hz')

    print('Convert to data frame')
    df = epochs_resampled.to_data_frame()
    #df = epochs.to_data_frame()
    df['subj'] = s_number
    df['session'] = session_no
    epochs_for_export.append(df)
    
#Once looped through, and all downsampled epochs are appended, save the larger file.
df_export = pd.concat(epochs_for_export, ignore_index=False)
df_export.to_csv('solution_epochs_for_plotting.csv', sep=',')



#Extracting the window information for later analysis ----------------------
#Create (initially empty) lists to store the individual data in
dfs = []
retrieved = []
grand_avg = []

#Define windows of interest
windows = {
        "prestim": (-1.3, -1.0),
        "readiness": (-1.0, 0.0),
        }

#ERP infromation as per previous studies
#-1.05 to -0.05 before stimulis (Travers et al. 2020)
#The apparent onset of the RP is known to vary considerably between studies (Shibasaki and Hallett, 2006).
# -3 to - -0.05 seconds before (Travers et al. )
#early window -1000ms to -500ms, and then late window -500ms to onset (Vercillo 2018).

#Create list of epoch files
epoch_files = glob.glob('solution_epochs/*epo.fif.gz')
print(epoch_files)
print(len(epoch_files))

#Loop through epoch files to load for averaging and creating a grand average for plotting
for epoch_name in epoch_files:
    
    #Create empty to list to append to
    epochs_df = []
    
    #Extract participant number
    subj = op.basename(epoch_name).split('_')[0]
    session_no = op.split(epoch_name)[1][5:6]
    
    print('processing epochs for participant ' + subj)
    
    #Read epochs
    epochs = mne.read_epochs(epoch_name,preload=True)
    
    #Check sampling frequency
    sf = epochs.info['sfreq']
    print("sampling freq is", sf)
    
    #Create 'wins' and add the window data for each participant
    #This will be used for statistical analyses later
    wins = retrieve(epochs, windows)
    
    #wins = retrieve(epochs, windows, epoch_name, summary_fnc=dict(mean=np.mean,sem=pd.DataFrame.sem))
    wins['subj']=subj
    wins['session']=session_no
    retrieved.append(wins)
    
#Save csv file with windows from retrieved object
win_df=pd.concat(retrieved, ignore_index=False)
win_df.to_csv('solution_prestim_window_on_task.csv', sep=',', mode='a', header=True)
print("Saving data to window csv")


























    #begin averaging epochs
    ss_avg = dict() # new empty dict
    ss_wins = dict()

    #for e in epoch_files:
    evokeds = {cond:epochs[cond].average() for cond in event_id}
    ss_avg[] = {str(cond):epochs[str(cond)].average() for cond in event_id}
    wins = retrieve(epochs, windows, e)
    ss_wins[e] = wins
    retrieved.append(ss_wins[e])
    for cond in FILLER:
        ss_avg_by_cond_FILLER[cond].append(ss_avg[e][cond])
#        if "03" or "07" in s_no:
#            ss_avg_by_cond_FILLER_cp[cond].append(ss_avg[e][cond])
#        if "01" in s_no:
#            ss_avg_by_cond_FILLER_hc[cond].append(ss_avg[e][cond])
    for cond in CRITICAL:
        ss_avg_by_cond_CRITICAL[cond].append(ss_avg[e][cond])
#        if "03" or "07" in s_no:
#            ss_avg_by_cond_CRITICAL_cp[cond].append(ss_avg[e][cond])
#        if "01" in s_no:
#            ss_avg_by_cond_CRITICAL_hc[cond].append(ss_avg[e][cond])

df=pd.concat(retrieved, ignore_index=False)
# write csv file for R stats
df.to_csv('2022_penguagestats.csv',sep=',')