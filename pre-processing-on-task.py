
"""
Pasithea In-Task EEG Analysis
@author: Chloe Dziego 2022
"""

import mne
import numpy as np
import os.path as op
import glob
import pandas as pd
import matplotlib.pyplot as plt
import csv

from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs

#Make sure that plots are interactive
%matplotlib

#Define ICA function (as per utils script)
def compute_ica_correction(raw,subjnr,session):
    raw_copy = raw.copy()
    raw_copy.filter(1., 40., n_jobs = 2, fir_design = 'firwin')

    ica_picks = mne.pick_types(raw_copy.info, eeg=True, eog=False, misc=False, stim=False, exclude='bads')

    #set ICA parameters
    reject = dict(eeg=250e-6)
    method = 'fastica'
    decim = 3
    random_state = 23
    ica = ICA(n_components=None, method=method, random_state=random_state)

    #Fit ICA
    ica.fit(raw_copy, picks=ica_picks, decim=decim, reject=reject)

    #Find EOG artefacts
    eog_average = create_eog_epochs(raw_copy, reject=reject, picks=ica_picks).average()
    eog_epochs = create_eog_epochs(raw_copy, reject=reject)
    eog_inds, scores = ica.find_bads_eog(eog_epochs)

    #Plot components identified as corresponding to EOG artefacts and overlays
    ica_plot_base = 'ica/' + subjnr + '_' + session + '_ica_plot'
    ica_overlay_name = 'ica/' + subjnr + '_' + session + '_ica_overlay.jpg'

    ica.plot_overlay(eog_average, exclude=eog_inds, show=False).savefig(ica_overlay_name)
    plt.close()
    
    for comp in range(len(eog_inds)):
        index = str(comp+1)
        ica_plot_name = ica_plot_base + index + '.jpg'
        ica.plot_properties(eog_epochs, picks=eog_inds, psd_args={'fmax':35.}, image_args={'sigma':1.}, show=False)[comp].savefig(ica_plot_name)
    plt.close()
    
    #Apply ICA (to unfiltered data)
    ica.exclude.extend(eog_inds)
    ica.apply(raw)

    #Record components
    comps = ica.labels_
    ica_comp_filename = 'ica/' + subjnr + '_' + session + '_ica_comps.csv'
    w = csv.writer(open(ica_comp_filename, "w"))
    for key, val in comps.items():
        w.writerow([key, val])

    return raw

def mark_bads(bad_chan_file, raw, subjnr):
    if op.isfile(bad_chan_file):
        reader = csv.DictReader(open(bad_chan_file),dialect=csv.excel_tab)
        bad_channels = [r['Channel'] for r in reader if r['Subject'] == subjnr]
        
    if bad_channels == ['']: print("No channels to mark as bad")
    else: 
        raw.info['bads'] = bad_channels[0].split(',')
        print("The following channels were marked as bad: "+str(raw.info['bads']))

    return raw

#Get lists of on-task EEG recordings files
subjects = glob.glob('data_test/*_Test.vhdr')

#Some data need to be rescaled due to funky OpenVibe recording parameters
subjects_to_rescale = glob.glob('data_test/to_be_rescaled/*_Test.vhdr')

#Issues with subject 31... this is to be done separately
#subjects = glob.glob('data_test/additional_recordings/CT31_S1_Test.vhdr')

#Define pre-processing parameters, EEG montage and rejection criteria
start_trigger = 10001
end_trigger = 10002
montage = mne.channels.make_standard_montage('standard_1020')
reject_ica = dict(eeg = 150e-6)
run_ica = True 

#Loop through each subject and session for data that does not need to be rescaled
for s in subjects:
	#Extract characters from file name to carry through participant information
    s_number = op.split(s)[1][0:4]
    print(s_number)
    session_no = op.split(s)[1][6:7]
    print("processing subject " + s_number + " and processing session " + session_no)
    
    #Read in the raw data file
    raw = mne.io.read_raw_brainvision(s, preload = True, eog = ['FP1', 'FP2'], misc = ['Channel 33'])
    
    #Set EEG reference
    raw.set_eeg_reference(ref_channels = ['TP9','TP10'])
    
    #Remove channels that are no longer needed
    raw.drop_channels(['TP9', 'TP10', 'Channel 33']) #some may also need to drop Aux1
    
    #Set the montage
    raw.set_montage(montage)

	#Set the bad channel information
    if session_no == '1':
        raw = mark_bads('bad_chans_1.txt',raw, s_number)
    else: raw = mark_bads('bad_chans_2.txt',raw, s_number)
    
    #Find events (in this case, just start of cognitive training)
    events = mne.events_from_annotations(raw)[0]
    event_id = mne.events_from_annotations(raw)[1]
    
    start_time = events[1, 0] / raw.info["sfreq"]
    end_time = 2400
    
    #first raw.crop is for all participants
    raw.crop(tmin = start_time, tmax = end_time)
    
    #for participant 31 (there is noise from 822 seconds onwards, just use what I can for now)
    #raw.crop(tmin = start_time, tmax = 822)
    
    if run_ica:
        print("Computing ICA-based EOG correction")
        raw = compute_ica_correction(raw,s_number,session_no)

    #Interpolate bad channels
    raw.interpolate_bads(reset_bads=True)
    
    #Filter the data
    raw.filter(0.1, 40., l_trans_bandwidth = 'auto', h_trans_bandwidth = 'auto', 
               filter_length = 'auto', method = 'fir', fir_window = 'hamming', phase = 'zero', n_jobs=2)
    
    #Visualise filtered data to check for any issues
    processed_psd = raw.plot_psd(fmin = 0, fmax = 40)
    plt.close('all')
    processed_psd.savefig("check_data/processed_psds/" + s_number + "_" + session_no + "_processed.jpg")
	
	#Save the processed file
    outfile = 'processed_EEG_files/' + s_number + '_' + session_no + '_processed_raw.fif.gz'
    raw.save(outfile, fmt = 'single', overwrite = True)
       
#Loop through each subject and session for data that needs to be rescaled
for s in subjects_to_rescale:
	#Extract characters from file name to carry through participant information
    s_number = op.split(s)[1][0:4]
    print(s_number)
    session_no = op.split(s)[1][6:7]
    print("processing subject " + s_number + " and processing session " + session_no)
    
    #Due to firealarm, participant 15 has two separate EEG recordings; merge them here.
    if s_number == 'CT15' and session_no == '2':
        print('Found subject 15, session 2, concatenating two recordings...')
        raw_to_start = mne.io.read_raw_brainvision(s, scale = 0.0406901, preload = True, eog = ['FP1', 'FP2'], misc = ['Channel 33'])
        raw_to_add = mne.io.read_raw_brainvision('data_test/additional_recordings/second_part_CT15.vhdr', scale = 0.0406901, preload = True, eog = ['FP1', 'FP2'], misc = ['Aux1'])
        raw = mne.concatenate_raws([raw_to_start,raw_to_add])
    else: 
        print("Continuing as per usual...")
        raw = mne.io.read_raw_brainvision(s, scale = 0.0406901, preload = True, eog = ['FP1', 'FP2'], misc = ['Channel 33'])
    
    #Set the EEG reference
    raw.set_eeg_reference(ref_channels = ['TP9','TP10'])
    
    #Remove the channels that are no longer needed
    raw.drop_channels(['TP9', 'TP10', 'Channel 33']) #some may also need to drop Aux1
    
    #Set montage
    raw.set_montage(montage)

	#Mark the appropriate bad_channels
    if session_no == '1':
        raw = mark_bads('bad_chans_1.txt',raw, s_number)
    else: raw = mark_bads('bad_chans_2.txt',raw, s_number)
    
    #Find events (in this case, just start of cognitive training)
    events = mne.events_from_annotations(raw)[0]
    event_id = mne.events_from_annotations(raw)[1]
    
    start_time = events[1, 0] / raw.info["sfreq"]
    
    #Specialty treatment for CT08
    if s_number == 'CT08' and session_no == '2':
        end_time = 1367
    else: end_time = 2400
    
    raw.crop(tmin = start_time, tmax = end_time)
    
    #Run the ICA to find and remove artefacts
    if run_ica:
        print("Computing ICA-based EOG correction")
        raw = compute_ica_correction(raw,s_number,session_no)
        plt.close('all')
        
    #Interpolate bad channels
    raw.interpolate_bads(reset_bads=True)
    
    #Filter the data
    raw.filter(0.1, 40., l_trans_bandwidth = 'auto', h_trans_bandwidth = 'auto', 
               filter_length = 'auto', method = 'fir', fir_window = 'hamming', phase = 'zero', n_jobs=2)
    
    #Visually check the data after filtering
    processed_psd = raw.plot_psd(fmin = 0, fmax = 40)
    plt.close('all')
    processed_psd.savefig("check_data/processed_psds/" + s_number + "_" + session_no + "_processed.jpg")
	
	#Save the rescaled data file
    outfile = 'processed_EEG_files/' + s_number + '_' + session_no + '_processed_raw.fif.gz'
    raw.save(outfile, fmt = 'single', overwrite = True)
 
