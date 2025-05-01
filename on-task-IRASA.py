
"""
Script for applying IRASA and peak oscillation detection functions
Memory development task
@author: Zachariah R. Cross, November 2022, adapted by Chloe
"""

# %%
##############################################################################
# Load modules
##############################################################################

# modules required for IRASA
import os
import mne
import yasa
import glob
import numpy as np
import pandas as pd
import os.path as op
import seaborn as sns
import matplotlib.pyplot as plt

# set parameters for plotting figures
%matplotlib qt
sns.set(style='white', font_scale=1.2)

# %%
##############################################################################
# Prepare files and data frames
##############################################################################

#Make sure to remove CT15_S2 from the error folder before running (they are only an issue for epoching/triggers)
# get list of epoched working memory data files
processed_files = glob.glob('processed_EEG_files/*.fif.gz')

all_participant_slopes = []

# loop through each epoched file
for file_name in processed_files:
    
    subj = op.split(file_name)[1][0:4]
    session_no = op.split(file_name)[1][5:6]
    print('---------------------- ' + 'Processing participant ' + subj + ' Session number ' + session_no + ' ----------------------')
    
    # first read in processed epoched file
    raw = mne.io.read_raw_fif(file_name)
    raw = raw.drop_channels(['FP1', 'FP2'])
    epochs = mne.make_fixed_length_epochs(raw, duration = 20.0, preload = True)

    # extract information from epoched data
    data = epochs.get_data()
    sf = epochs.info['sfreq'] # set sf as the sampling frequency
    chan = epochs.ch_names # define channel as ch_names
    
    # initialise data frames
    dfs = []
    df_osc = []
    psd_total = []
    df_aperiodic = []

# %%
##############################################################################
# Run IRASA
##############################################################################

    # loop through each epoch to estimate trial-level aperiodic activity
    for idx in range(data.shape[0]):
        freqs, psd_aperiodic, psd_osc, fit_params = yasa.irasa(data[idx, :, :], sf, 
                                                               ch_names = chan, 
                                                               band = (1, 30), 
                                                               win_sec = 1, 
                                                               return_fit = True)
        
        # generate data frames required for later analysis
        fit_params.insert(loc=0, column="epoch", value=idx) # add epoch column
        fit_params['subj'] = subj # add subject column
        fit_params['session'] = session_no
        dfs.append(fit_params) # append each epoch data frame
            
        # append the psd arrays
        psd_total.append(psd_osc)
        
        # generate data frame for residual oscillatory activity
        df_osc_epoch = pd.DataFrame(psd_osc)
        df_osc_epoch.insert(loc=0, column="epoch", value=idx)
        df_osc_epoch['subj'] = subj
        df_osc_epoch['session'] = session_no
        df_osc.append(df_osc_epoch)
        
        # generate data frame for aperiodic psd
        df_aperiodic_epoch = pd.DataFrame(psd_aperiodic)
        df_aperiodic_epoch.insert(loc=0, column="epoch", value=idx)
        df_aperiodic_epoch['subj'] = subj
        df_aperiodic_epoch['session'] = session_no
        df_aperiodic.append(df_aperiodic_epoch)
     
    # concatenate each epoch structure
    df_slope_values = pd.concat(dfs)
    df_osc_df = pd.concat(df_osc)
    df_aperiodic_df = pd.concat(df_aperiodic)
    
    # save psd aperiodic and psd oscillatory output files for each subject
    df_osc_df.to_csv('power_psd_data/' + subj + '_' + session_no + '_power_psd.csv', header = True)
    df_slope_values.to_csv('irasa_data/' + subj + '_' + session_no + '_aperiodic_memdev.csv', header = True)
    df_aperiodic_df.to_csv('aperiodic_psd_data/' + subj + '_' + session_no + '_aperiodic_psd.csv', header = True)
    
    #append to a larger dataframe to save everyone together
    all_participant_slopes.append(df_slope_values)

    # plot the aperiodic component on a linear-log scale
    plt.plot(freqs, psd_aperiodic[2, :], 'k', lw=2.5)
    plt.fill_between(freqs, psd_aperiodic[2, :], cmap='Spectral')
    plt.xlim(1, 40)
    plt.yscale('log')
    sns.despine()
    plt.title('Aperiodic component at ' + chan[2], fontsize = 15)
    plt.xlabel('Frequency [Hz]',fontsize = 20)
    plt.ylabel('PSD log($uV^2$/Hz)',fontsize = 20)
    plt.savefig('irasa_figures/' + subj + '_' + session_no +'_aperiodic.png',dpi = 300, 
                bbox_inches='tight');
    plt.close();
    
    # and oscillatory component on a linear-linear scale
    plt.plot(freqs, psd_osc[2, :], 'k', lw=2.5)
    plt.fill_between(freqs, psd_osc[2, :], cmap='Spectral')
    plt.xlim(1, 40)
    sns.despine()
    plt.title('Oscillatory component at ' + chan[2],fontsize = 15)
    plt.xlabel('Frequency [Hz]',fontsize = 20)
    plt.ylabel('PSD log($uV^2$/Hz)',fontsize = 20)
    plt.savefig('irasa_figures/' + subj + '_' + session_no + '_oscillatory.png',dpi = 300, 
                bbox_inches='tight');
    plt.close();
    
#After everything has been run:     
# create larger IRASA data frame with each participant
df_export = pd.concat(all_participant_slopes)
if op.isfile('all_participant_aperiodic.csv'):
    df_export.to_csv('all_participant_aperiodic.csv', sep=',', mode='a', header=False)
else:
    df_export.to_csv('all_participant_aperiodic.csv', sep=',', mode='a', header=True)

# %%
##############################################################################
# Create Average Power Values Across Each Individualised Band
##############################################################################

#create list of all data csvs
all_frequencies = glob.glob('power_psd_data/*_power_psd.csv')

#read in iaf information for individualised frequency bands
iaf_info = pd.read_table('iaf_long.txt',dtype = {'subj':str, 'measure':str, 'value':np.float64}, na_values = 'None')

#bands for analysis
bands = ["alpha", "theta"]
#create mapping for renaming channels
mapping = {0:'Fz',
 1:'F3',
 2:'F7',
 3:'FT9',
 4:'FC5',
 5:'FC1',
 6:'C3',
 7:'T7',
 8:'CP5',
 9:'CP1',
 10:'Pz',
 11:'P3',
 12:'P7',
 13:'O1',
 14:'Oz',
 15:'O2',
 16:'P4',
 17:'P8',
 18:'CP6',
 19:'CP2',
 20:'Cz',
 21:'C4',
 22:'T8',
 23:'FT10',
 24:'FC6',
 25:'FC2',
 26:'F4',
 27:'F8'}

#initialise dataframe
all_subjects_list = []
df_all = []

#create custom rounding function
def custom_round(x):
    if x - int(x) >= 0.5:
        return int(x) + 1
    else:
         return int(x)

#individualised frequency estimates from IRASA outputs
for csvname in all_frequencies:

    bands = ['alpha', 'theta']
    subj = op.split(csvname)[1][0:4]
    session_no = op.split(csvname)[1][5:6]
    print('---------------------- Processing data for ' + subj + ' session number: ' + session_no + ' ----------------------')
    
    subj_data = pd.read_csv(csvname, na_values = 'None')
    
    for b in bands:
        print("processing " + b + " band")
    
        # Define frequency band limits
        band_lower = b + "_" + "lower"
        band_upper = b + "_" + "upper"
    
    	#Retrieve the relevant IAF information
        print(iaf_info['subj'] == subj)
        print(iaf_info['measure'] == band_lower)
        lower = (iaf_info.value[(iaf_info['subj'] == subj) & (iaf_info['measure'] == band_lower)])
        upper = (iaf_info.value[(iaf_info['subj'] == subj) & (iaf_info['measure'] == band_upper)])
    
        lower = lower.values
        upper = upper.values
        
        print("------------- HERE ARE THE LOWER AND UPPER BOUNDS FOR SUBJECT (before rounding)" + subj + " BAND: " + b + " ---------------- ")
        print(lower)
        print(upper)
    
        lower = custom_round(lower[0])
        upper = custom_round(upper[0])
        
        print("------------- HERE ARE THE LOWER AND UPPER BOUNDS FOR SUBJECT (after rounding)" + subj + " BAND: " + b + " ---------------- ")
        print(lower)
        print(upper)
        
        #Minusing 1 because the frequency labels are in index form (whereby 0 = 1Hz)
        lower = lower-1
        upper = upper -1
        lower = str(lower)
        upper = str(upper)

        lower_index = subj_data.columns.get_loc(lower)
        upper_index = subj_data.columns.get_loc(upper)
        print('Subject lower bound column index: ' + str(lower_index))
        print('Subject upper bound column index: ' + str(upper_index))
        
        #Extract the relevant band average and append to data frame
        if b == 'theta':
            subj_data['theta_means'] = subj_data.iloc[:,lower_index:(upper_index+1)].mean(axis=1)
        if b == 'alpha':
            subj_data['alpha_means'] = subj_data.iloc[:,lower_index:(upper_index+1)].mean(axis=1)  
    
    #Rename the channels    
    rename_chan = subj_data.rename(columns ={'Unnamed: 0' : 'chan'})
    chans_mapped = rename_chan.replace({"chan":mapping})
    #Select the relevant columns for exporting
    for_export = chans_mapped[['chan', 'subj', 'epoch', 'session', 'theta_means', 'alpha_means']]    
    all_subjects_list.append(for_export) 
    df_all = pd.concat(all_subjects_list)
        
#Save the entire dataframe for analysis in R
df_export = pd.DataFrame(df_all)
if op.isfile('power_means.csv'):
    df_export.to_csv('power_means.csv', sep = ',', mode = 'a', header = False)
else:
    df_export.to_csv('power_means.csv', sep = ',', mode = 'a', header = True)
        
#checking the mean power values
#-5.91955e-13 + 2.221405127745104e-12 + -3.915572904002011e-13 + -6.832865388369751e-13
#5.546062985079276e-13 / 4

#checking the mean power values
#-6.149882850742805e-13 + 3.162522021006071e-13 + 5.790475166066765e-13 + 3.4299408967374375e-14 + 	4.137946941764034e-13
#7.284055367767808e-13 / 5

# %%
##############################################################################
# Create a full data frame of all aperiodic slopes/intercepts
##############################################################################

#create list of all data csvs
all_slopes = glob.glob('aperiodic_psd_data/*_aperiodic_psd.csv')
all_slopes = glob.glob('aperiodic_psd_data/CT29_1_aperiodic_psd.csv')

#initialise dataframe
all_subjects_list = []
df_all = []

#Individualised frequency estimates from IRASA outputs
for csvname in all_slopes:

    subj = op.split(csvname)[1][0:4]
    session_no = op.split(csvname)[1][5:6]
    print('---------------------- Processing data for ' + subj + ' session number: ' + session_no + ' ----------------------')
    
    subj_data = pd.read_csv(csvname, na_values = 'None')
        
    rename_chan = subj_data.rename(columns ={'Unnamed: 0' : 'chan'})
    chans_mapped = rename_chan.replace({"chan":mapping})
    for_export = chans_mapped[['chan', 'subj', 'epoch', 'session']]
        
    all_subjects_list.append(for_export) 
    df_all = pd.concat(all_subjects_list)
     
#Save the full dataframe for analysis in R   
df_export = pd.DataFrame(df_all)
if op.isfile('aperiodic_time_all.csv'):
    df_export.to_csv('aperiodic_time_all.csv', sep = ',', mode = 'a', header = False)
else:
    df_export.to_csv('aperiodic_time_all.csv', sep = ',', mode = 'a', header = True)




    