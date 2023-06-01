"""
Neural Signal Analysis
"""

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import signal
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
from scipy import stats
import pandas as pd
import scipy.optimize as spo
import pandas as pd
import numpy.fft as fft_
import csv
import time
import scipy as spy

plt.rcParams["font.family"] = 'serif'
plt.rcParams.update({'font.size': 14})

data_url = 'https://github.com/JamesETyrrell/PhD_Analysis/blob/34f00eb585629ebcffbd282462128de12cda54fc/Data/Electrophysiological%20Sample%20Data.csv?raw=true'
neural_data = pd.read_csv(data_url)

time_values = np.array(neural_data['Drain Time'])
gate_voltage = np.array(neural_data['Gate voltmeter Voltage'])
drain_current = np.array(neural_data['Drain Current'])
time_values = time_values - time_values[0]
time_interval = time_values[1] - time_values[0]
sampling_frequency = 1/time_interval
sampling_total = time_values[-1]
DFT_res = 1/sampling_total
current_resolution = max([len(str(drain_current[1]).split('.')[1]),len(str(drain_current[0]).split('.')[1]), len(str(drain_current[2]).split('.')[1])])
minimum_current_value = (10)**(-1*current_resolution)
print('Current resolution: ' + str(minimum_current_value*1e6) + ' uA')
voltage_resolution = max([len(str(gate_voltage[1]).split('.')[1]),len(str(gate_voltage[0]).split('.')[1]), len(str(gate_voltage[2]).split('.')[1])])
minimum_voltage_value = (10)**(-1*voltage_resolution)
print('Voltage resolution: ' + str(minimum_voltage_value*1e6) + ' uV')
gate_p2p = max(gate_voltage) - min(gate_voltage)
print('Gate peak-to-peak: ', gate_p2p)

low_cutoff = 1
butterworth_order = 3
sig_type = 'ECoG'
fft_low_cutoff = low_cutoff
fft_butterworth = butterworth_order
high_value = 500
bins_1 = 70
bins_2 = 140
bins_3 = 400
cmap_ = plt.cm.jet

# Analysis Options
#transconductance_spread_analysis = 'no'
transconductance_spread_analysis = 'yes'
simulate_oect = 'no'
#simulate_oect = 'yes'
if transconductance_spread_analysis == 'no':
    simulate_oect = 'no'
bin_width_value = 1e-6
noise_lower_ = 0.1e-6
noise_upper_ = 1e-6
noise_step_ = 0.1e-6

"""------------------------------------------------------------------------"""

# Function describing the error from the drain error plots
def new_error_analysis(gate_voltage_drain_error, filter_type, gm):
    gate_voltage_drain_error = sorted(gate_voltage_drain_error)    
    first_gate = np.round(gate_voltage_drain_error[0][0],7)
    last_gate = np.round(gate_voltage_drain_error[-1][0],7)
    bin_width = np.abs((first_gate - last_gate)) / 25
    bin_width = 0.5e-6
    gate_bins = np.arange(first_gate, last_gate + bin_width, bin_width)
    binned_error_values = [[] for i in gate_bins]
    for i in range(len(gate_voltage_drain_error)): 
        gate = gate_voltage_drain_error[i][0] # round to nearest 1e-7
        gate = np.round(gate, 7)
        drain_error = gate_voltage_drain_error[i][1]
        gate_array = np.empty(len(gate_bins))
        gate_array.fill(gate)
        gate_differences = gate - gate_bins
        gate_differences_abs = np.abs(gate_differences)
        closest_gate_index = list(gate_differences_abs).index(min(gate_differences_abs))
        binned_error_values[closest_gate_index].append(drain_error)
    binned_error = [[gate_bins[i], binned_error_values[i]] for i in range(len(gate_bins)) if len(binned_error_values[i]) > 15]
    # [gate voltage change, error] is now placed into bins with [closest bin, [drain errors]]
    sd_away = 2
    for i in range(len(binned_error)):
        error_values = binned_error[i][1]
        mean = np.mean(error_values)
        sd = np.sqrt(np.var(error_values))
        # remove outliers of error values
        new_error_values = [j for j in error_values if (np.abs((j - mean)/sd)) < sd_away]
        deviation = 1.5*(stats.iqr(error_values))
        upper_limit = np.percentile(error_values, 75) + deviation
        lower_limit = np.percentile(error_values, 25) - deviation
        new_error_values = [j for j in error_values if j > lower_limit]
        new_error_values = [j for j in new_error_values if j < upper_limit]
        #sd_away = 2
        #new_error_values = [j for j in new_error_values if np.abs((j - np.mean(new_error_values))/np.sqrt(np.var(new_error_values))) < sd_away]
        binned_error[i][1] = new_error_values
    binned_error_gate = [i[0] for i in binned_error]
    mean_binned_error = [np.mean(i[1]) for i in binned_error]
    sd_binned_error = [np.sqrt(np.var(i[1])) for i in binned_error]
    the_list = [i for i in sd_binned_error if np.abs(binned_error_gate[sd_binned_error.index(i)]) < 5e-6 ]
    sd_within_5uV = np.mean(the_list)
    sd_of_sd = np.sqrt(np.var(the_list))
    sd_of_sd_uv = 4*sd_of_sd/gm
    two_sigma_gm = 4*sd_within_5uV/gm
    print('Gate voltage equivalent (4 sigma) of standard deviation within 10uV of 0 gate voltage change is: ', 1000000*two_sigma_gm, ' uV')
    # print('Average standard deviation within 5uV of 0 gate voltage change is ', 1000000*sd_within_5uV , ' $\mu$A')
    print('SD of standard deviation within 5uV of 0 gate voltage change is ', 1000000*sd_of_sd_uv , ' $\mu$V')
    #sd_binned_error = [(max(i[1])-min(i[1])) for i in binned_error]
    fig, ax1 = plt.subplots(figsize=(7.5,5))
    plt.title(filter_type + ' Current Error Mean and Standard Deviation')
    color = 'tab:blue'
    ax1.set_xlabel('Gate Voltage Change ($\mu$V)')
    #ax1.set_xlim([-16,16])
    ax1.set_ylabel('Error Mean ($\mu$A)', color=color) 
    ax1.scatter(1000000*np.array(binned_error_gate), 1000000*np.array(mean_binned_error), color=color, marker='x') 
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Error Standard Deviation (nA)', color=color)  # we already handled the x-label with ax1
    ax2.scatter(1000000*np.array(binned_error_gate), 1000000000*np.array(sd_binned_error), color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    #slope, intercept, r_value, p_value, std_err = stats.linregress([i[0] for i in gate_voltage_drain_error],[i[1] for i in gate_voltage_drain_error])
    #print('Gradient = ', slope)
    mean_binned_error = [i for i in mean_binned_error if np.abs(binned_error_gate[mean_binned_error.index(i)]) < 5e-6 ]
    binned_error_gate = [i for i in binned_error_gate if np.abs(i) < 5e-6 ]
    slope, intercept, r_value, p_value, std_err = stats.linregress(binned_error_gate,mean_binned_error)
    print('Gradient = ', slope, '+- ', std_err)
    return binned_error    
        

"""------------------------------------------------------------------------"""

# no filter

#raw transconductance / perfect signal
raw_transconductance = []
#noramlised_cutoff = low_cutoff / (sampling_frequency / 2) # Normalize the frequency
#b, a = signal.butter(butterworth_order, noramlised_cutoff, btype='highpass')
#trans_drain_current = signal.filtfilt(b, a, drain_current)
for i in range(len(time_values) - 1):
    gate = gate_voltage[i]
    next_gate = gate_voltage[i+1]
    drain = drain_current[i]
    next_drain = drain_current[i+1]
    gate_difference_value = next_gate - gate
    if gate_difference_value != 0:
        if np.abs(gate_difference_value) > 100e-6:
            transconductance_value = (next_drain - drain) / gate_difference_value
            raw_transconductance.append(transconductance_value)
#raw_transconductance_mean = np.mean(raw_transconductance)
raw_transconductance_mean = (max(drain_current) - min(drain_current))/(max(gate_voltage) - min(gate_voltage))
print('Transconductance mean = ', 1000*raw_transconductance_mean, ' mS')
minimum_measurable_voltage = minimum_current_value/raw_transconductance_mean
print('Minimum measurable voltage: ', minimum_measurable_voltage*1e6, ' uV')

# ideal response no filter
perf_oect_no_filter = [drain_current[0]]
for i in range(len(time_values)-1):
    gate_dif = gate_voltage[i+1] - gate_voltage[i]
    next_drain = gate_dif*raw_transconductance_mean + perf_oect_no_filter[i]
    perf_oect_no_filter.append(next_drain)
perf_oect_no_filter = np.array(perf_oect_no_filter)
simulated_drain_current = perf_oect_no_filter



# ABS Error for change in gate voltage
abs_error = []
gate_change = []
gate_drain_error = []
for i in range(len(perf_oect_no_filter)-1):
    drain_change = drain_current[i+1] - drain_current[i]
    perfect_change = perf_oect_no_filter[i+1] - perf_oect_no_filter[i]
    error = (drain_change - perfect_change)
    gate_change_value = gate_voltage[i+1] - gate_voltage[i]
    abs_error.append(error)
    gate_change.append(gate_change_value)
    gate_drain_error.append([gate_change_value, error])
# Drain error distribution
plt.figure(figsize=(7.5,5))
plt.title('No Filter Current Error for Gate Voltage Change')
plt.xlabel('Gate Change (uV)')
plt.ylabel('Drain Error ($\mu$A)')
#plt.scatter(1000000*np.array(gate_change), 1000000*np.array(abs_error), marker = 'x')
plt.hist2d(1000000*np.array(gate_change), 1000000*np.array(abs_error), bins=bins_1, norm=mpl.colors.LogNorm(), cmap=cmap_)
plt.colorbar()
(mu_no_filter, sigma_no_filter) = norm.fit(abs_error)

new_error_analysis(gate_drain_error,'No Filter', raw_transconductance_mean)

# residual sum of squares for no filter vs ideal response
RSS_no_filter = np.sum((drain_current - perf_oect_no_filter)**2) 

# plot raw without filter
fig, ax1 = plt.subplots()
ax1.title.set_text('No Filter')
color = 'tab:blue'
ax1.set_xlabel('Time (s)')
#ax1.set_xlim([0.92,1.01])
ax1.set_ylabel('Gate Voltage (uV)', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.plot(time_values, 1000000*gate_voltage, color = color, label = 'Gate Voltage', alpha=0.7)
ax2 = ax1.twinx() 
color = 'tab:red'
ax2.set_ylabel('Drain Current ($\mu$A)', color=color)
ax2.tick_params(axis='y', labelcolor=color)
#ax2.plot(time_values, 1000*perf_oect_no_filter, color = 'k', linestyle = 'dashed', label = 'Drain Current Simulation', alpha=0.7)
ax2.plot(time_values, 1000000*drain_current, color = color, label = 'Drain Current', alpha=0.7)
fig.tight_layout()



"""------------------------------------------------------------------------"""

# Bandpass the gate voltage and drain current

# high-pass drain
noramlised_cutoff = low_cutoff / (sampling_frequency / 2) # Normalize the frequency
b, a = signal.butter(butterworth_order, noramlised_cutoff, btype='highpass')
bandpassed_drain_current = signal.filtfilt(b, a, drain_current)
bandpassed_gate_voltage = signal.filtfilt(b, a, gate_voltage)

# low pass drain
freq_dom_values = fft_.fftfreq(len(gate_voltage) , time_interval)
max_freq = np.max(freq_dom_values)
if max_freq > 5e3:
    high_cutoff = high_value
    butterworth_order = 10
else:
    high_cutoff = max_freq - 1
    butterworth_order = 3
noramlised_cutoff = high_cutoff / (sampling_frequency / 2) # Normalize the frequency
b, a = signal.butter(butterworth_order, noramlised_cutoff, btype='lowpass')
bandpassed_drain_current = signal.filtfilt(b, a, bandpassed_drain_current)
bandpassed_gate_voltage = signal.filtfilt(b, a, bandpassed_gate_voltage)

# initial condition changed so need to re-find ideal OECT response
perf_oect = [bandpassed_drain_current[0]]
for i in range(len(time_values)-1):
    gate_dif = bandpassed_gate_voltage[i+1] - bandpassed_gate_voltage[i]
    next_drain = gate_dif*raw_transconductance_mean + perf_oect[i]
    perf_oect.append(next_drain)
perf_oect = np.array(perf_oect)
simulated_bandpass_drain_current = perf_oect


abs_error = []
gate_change = []
gate_drain_error = []
for i in range(len(perf_oect)-1):
    drain_change = bandpassed_drain_current[i+1] - bandpassed_drain_current[i]
    perfect_change = perf_oect[i+1] - perf_oect[i]
    error = (drain_change - perfect_change)
    gate_change_value = bandpassed_gate_voltage[i+1] - bandpassed_gate_voltage[i]
    abs_error.append(error)
    gate_change.append(gate_change_value)
    gate_drain_error.append([gate_change_value, error])
plt.figure(figsize=(7.5,5))
plt.title('Bandpass Current Error for Gate Voltage Change')
plt.xlabel('Gate Change (uV)')
plt.ylabel('Drain Error ($\mu$A)')
#plt.xlim([-16,16])
#plt.scatter(1000000*np.array(gate_change), 1000000*np.array(abs_error), marker = 'x')
plt.hist2d(1000000*np.array(gate_change), 1000000*np.array(abs_error), bins=bins_2, norm=mpl.colors.LogNorm(), cmap=cmap_)
plt.colorbar()
#plt.xlim([-16,16])
#plt.ylim([-0.35,0.35])
(mu_notched_bandpass, sigma_bandpassed) = norm.fit(abs_error)

new_error_analysis(gate_drain_error,'Bandpass', raw_transconductance_mean)

RSS_bandpassed = np.sum((bandpassed_drain_current - perf_oect)**2 )

fig, ax1 = plt.subplots()
ax1.title.set_text('Bandpass')
color = 'tab:blue'
ax1.set_xlabel('Time (s)')
#ax1.set_xlim([0.92,1.01])
ax1.set_ylabel('Gate Voltage (uV)', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.plot(time_values, 1000000*bandpassed_gate_voltage, color = color, label = 'Gate Voltage', alpha=0.7)
ax2 = ax1.twinx() 
color = 'tab:red'
ax2.set_ylabel('Drain Current ($\mu$A)', color=color)
ax2.tick_params(axis='y', labelcolor=color)
#ax2.plot(time_values, 1000*perf_oect, color = 'k', linestyle = 'dashed', label = 'Drain Current Simulation', alpha=0.7)
ax2.plot(time_values, 1000000*bandpassed_drain_current, color = color, label = 'Drain Current', alpha=0.7)
fig.tight_layout()


"""------------------------------------------------------------------------"""

# Notch filter the drain

notched_bandpassed_drain_current = bandpassed_drain_current

frequency_to_remove = np.arange(50,(sampling_frequency / 2),100)
#frequency_to_remove = np.arange(50,300,100)
for i in frequency_to_remove:
    noramlised_frequency = i/ (sampling_frequency / 2)
    Q = 50 # quality factor of notch filter
    b, a = signal.iirnotch(noramlised_frequency, Q)
    notched_bandpassed_drain_current = signal.filtfilt(b, a, notched_bandpassed_drain_current)
    #bandpassed_gate_voltage = signal.filtfilt(b, a, bandpassed_gate_voltage)

frequency_to_remove = np.arange(62.5,(sampling_frequency / 2),62.5)
for i in frequency_to_remove:
    noramlised_frequency = i/ (sampling_frequency / 2)
    Q = 50 # quality factor of notch filter
    b, a = signal.iirnotch(noramlised_frequency, Q)
    notched_bandpassed_drain_current = signal.filtfilt(b, a, notched_bandpassed_drain_current)

# initial condition changed so need to re-find ideal OECT response
perf_oect = [notched_bandpassed_drain_current[0]]
for i in range(len(time_values)-1):
    gate_dif = bandpassed_gate_voltage[i+1] - bandpassed_gate_voltage[i]
    next_drain = gate_dif*raw_transconductance_mean + perf_oect[i]
    perf_oect.append(next_drain)
perf_oect = np.array(perf_oect)
simulated_notched_bandpass_drain_current = perf_oect

#print('Max difference in drain current: ', 1000*(max(notched_bandpassed_drain_current) - min(notched_bandpassed_drain_current)))
#SNR = 10*np.log10(((253e-6)**2)/((0.007e-6)**2))
#print(SNR)

abs_error = []
gate_change = []
gate_drain_error = []
for i in range(len(perf_oect)-1):
    drain_change = notched_bandpassed_drain_current[i+1] - notched_bandpassed_drain_current[i]
    perfect_change = perf_oect[i+1] - perf_oect[i]
    error = (drain_change - perfect_change)
    gate_change_value = bandpassed_gate_voltage[i+1] - bandpassed_gate_voltage[i]
    abs_error.append(error)
    gate_change.append(gate_change_value)
    gate_drain_error.append([gate_change_value, error])
plt.figure(figsize=(7.5,5))
plt.title('Notch/Bandpass Current Error for Gate Voltage Change')
plt.xlabel('Gate Change (uV)')
plt.ylabel('Drain Error ($\mu$A)')
#plt.scatter(1000000*np.array(gate_change), 1000000*np.array(abs_error), marker = 'x')
plt.hist2d(1000000*np.array(gate_change), 1000000*np.array(abs_error), norm=mpl.colors.LogNorm(), bins= bins_3, cmap=cmap_)
plt.colorbar()
(mu_notched, sigma_notched_bandpassed) = norm.fit(abs_error)

new_error_analysis(gate_drain_error,'Notch/Bandpass', raw_transconductance_mean)

RSS_bandpass_notch = np.sum((notched_bandpassed_drain_current - perf_oect)**2 )

fig, ax1 = plt.subplots()
#fig, ax1 = plt.subplots(figsize=(8.5,5))
color = 'tab:blue'
ax1.set_xlabel('Time (s)')
#ax1.set_xlim([0.92,1.01])
ax1.tick_params(axis='y', labelcolor=color)
ax1.title.set_text('Notch/Bandpass')
ax1.plot(time_values, 1000000*bandpassed_gate_voltage, color = color, label = 'Gate Voltage', alpha=0.7)
ax1.set_ylabel('Gate Voltage ($\mu$V)', color=color)
ax2 = ax1.twinx() 
color = 'tab:red'
ax2.set_ylabel('Drain Current ($\mu$A)', color=color)
ax2.tick_params(axis='y', labelcolor=color)
#ax2.plot(time_values, 1000*perf_oect, color = 'k', linestyle = 'dashed', label = 'Drain Current Simulation', alpha=0.7)
ax2.plot(time_values, 1000000*notched_bandpassed_drain_current, color = color, label = '50Hz Notched Drain Current', alpha=0.7, linewidth = 2)
#plt.legend()
fig.tight_layout()
#plt.savefig('50Hz Notch Filter OECT recording ' + str(file_name) + '.png')


"""------------------------------------------------------------------------"""

print(['No filter, Bandpass Both, Notch Drain'])
RSS_comparison = [RSS_no_filter, RSS_bandpassed, RSS_bandpass_notch]
print(RSS_comparison)

"""------------------------------------------------------------------------"""

# Fourier Transform of Bandpassed and Notched Data

def fft_plot(gate_voltage_values, drain_current_values, time_interval_value, filter_type):
    drain_current_values = drain_current_values - np.mean(drain_current_values)
    gate_voltage_values = gate_voltage_values - np.mean(gate_voltage_values)
    
    fft_gate = fft_.fft(gate_voltage_values)
    fft_gate = np.abs(fft_gate) / max(np.abs(fft_gate) )
    fft_drain = fft_.fft(drain_current_values)
    fft_drain = np.abs(fft_drain) / max(np.abs(fft_drain))
    freq_dom = fft_.fftfreq(len(fft_gate) , time_interval_value)
    
    length = int(len(fft_gate)/2)-1
    freq_dom = freq_dom[1:length]
    fft_gate = fft_gate[1:length]
    fft_drain = fft_drain[1:length]
   
    fft_gate = gaussian_filter(fft_gate, sigma=5)
    fft_drain = gaussian_filter(fft_drain, sigma=5)
    
    fig, ax1 = plt.subplots(figsize=(11,5.5))
    color = 'tab:blue'
    ax1.set_xlabel('Frequency (Hz)')
    #ax1.set_xlim([98,10100])
    ax1.set_xscale('log')
    ax1.set_ylabel('Normalised Gate Amplitude', color=color)
    ax1.set_yscale('log')
    ax1.set_title('Fourier Transform ' + filter_type)
    ln1= ax1.plot(freq_dom, fft_gate, alpha = 0.7, color=color, label = 'Gate FFT')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Normalised Drain Amplitude', color=color)  # we already handled the x-label with ax1
    ax2.set_yscale('log')
    ln2 = ax2.plot(freq_dom, fft_drain, alpha = 0.7, color=color, label = 'Drain FFT')
    ax2.tick_params(axis='y', labelcolor=color)
    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    fig.tight_layout()
    
    return
    
fft_plot(gate_voltage, drain_current, time_interval, 'No Filter')
fft_plot(bandpassed_gate_voltage, bandpassed_drain_current, time_interval, 'Bandpass')
fft_plot(bandpassed_gate_voltage, notched_bandpassed_drain_current, time_interval, 'Notch/Bandpass')

"""------------------------------------------------------------------------"""

# Transconductance spread

def transconductance_spread(gate_voltage_type, drain_current_type, type_name, bin_width_):
    
    # Get a list of the [gate change, transconductance value]
    transconductance_gate_difference = []
    for i in range(len(gate_voltage_type) - 1):
        gate = gate_voltage_type[i]
        next_gate = gate_voltage_type[i+1]
        drain = drain_current_type[i]
        next_drain = drain_current_type[i+1]
        gate_difference_value = next_gate - gate
        if gate_difference_value != 0:
            transconductance_value = (next_drain - drain) / gate_difference_value
            transconductance_gate_difference.append([gate_difference_value,transconductance_value])
    transconductance_gate_difference = sorted(transconductance_gate_difference)
    trans_gate_dif = transconductance_gate_difference
    
    gate_difference_values = [i[0] for i in transconductance_gate_difference]
    transconductance_values = [i[1] for i in transconductance_gate_difference]
    # plot raw transconductance values
    plt.figure(figsize=(6.5,5))
    plt.title(type_name + ' Transconductance Values')
    plt.xlabel('Gate Voltage Change ($\mu$V)')
    plt.ylabel('Transconductance Value (mS)')
    plt.yscale('symlog')
    plt.xscale('symlog')
    plt.xlim([-350,350])
    #plt.ylim([-1001000,1001000])
    plt.plot(1000000*np.array(gate_difference_values),1000*np.array(transconductance_values), 'kx')
    plt.tight_layout()
    
    # put the gate changes into bins/lists then find the mean and sd of those lists
    first_gate = np.round(transconductance_gate_difference[0][0],7) #round to 1e-7
    last_gate = np.round(transconductance_gate_difference[-1][0],7)
    bin_width = bin_width_
    gate_bins = np.arange(first_gate, last_gate + bin_width, bin_width)
    binned_transconductance_values = [[] for i in gate_bins]
    for i in range(len(transconductance_gate_difference)):
        gate = transconductance_gate_difference[i][0] # round to nearest 1e-7
        gate = np.round(gate, 8)
        transconductance_val = transconductance_gate_difference[i][1]
        gate_array = np.empty(len(gate_bins))
        gate_array.fill(gate)
        gate_differences = gate - gate_bins
        gate_differences_abs = np.abs(gate_differences)
        closest_gate_index = list(gate_differences_abs).index(min(gate_differences_abs))
        binned_transconductance_values[closest_gate_index].append(transconductance_val)
    binned_transconductance = [[gate_bins[i], binned_transconductance_values[i]] for i in range(len(gate_bins)) if len(binned_transconductance_values[i]) > 3]        
    #sd_away = 2
    for i in range(len(binned_transconductance)): # remove outliers
        error_values = binned_transconductance[i][1]
        deviation = 1.5*(stats.iqr(error_values))
        upper_limit = np.percentile(error_values, 75) + deviation
        lower_limit = np.percentile(error_values, 25) - deviation
        new_error_values = [j for j in error_values if j > lower_limit]
        new_error_values = [j for j in new_error_values if j < upper_limit]
        #new_error_values = [j for j in error_values if np.abs(j) < 1]
        #mean = np.mean(error_values)
        #sd = np.sqrt(np.var(error_values))
        #new_error_values = [j for j in new_error_values if (np.abs((j - mean)/sd)) < sd_away]
        binned_transconductance[i][1] = new_error_values
    new_gate_bins = [i[0] for i in binned_transconductance]
    mean_binned_transconductance = [np.mean(i[1]) for i in binned_transconductance]
    sd_binned_transconductance = [np.sqrt(np.var(i[1])) for i in binned_transconductance]
    
    # Determine the point where the mean - variance becomes positive. 
    n_sigma = 3
    noise_value_array = list(np.array(mean_binned_transconductance) - n_sigma*np.array(sd_binned_transconductance))
    mean_trans = np.empty(len(mean_binned_transconductance))
    mean_trans.fill(raw_transconductance_mean)
    noise_value_array = list(mean_trans - n_sigma*np.array(sd_binned_transconductance))
    gate_bin_indices_neg = []
    gate_bin_indices_pos = []
    for i in range(len(noise_value_array)):
        if new_gate_bins[i] < 0:
            if noise_value_array[i] > 0:
                gate_bin_indices_neg.append(i)
    for i in range(len(noise_value_array)):
        if new_gate_bins[i] > 0:
            if noise_value_array[i] > 0:
                gate_bin_indices_pos.append(i)
    if len(gate_bin_indices_neg) > 0:
        if len(gate_bin_indices_pos) > 0:
            gate_bin_indices = [max(gate_bin_indices_neg),min(gate_bin_indices_pos)]
            print(type_name + ' The values for which the transconductance values remain positive within ', str(n_sigma), ' standard deviations are ', str(np.round(1e6*new_gate_bins[gate_bin_indices[0]])) , ' and ', str(np.round(1e6*new_gate_bins[gate_bin_indices[1]])) , ' uV')
            
    
    # Plot the results
    fig, ax1 = plt.subplots(figsize=(6.5,5))
    plt.title(type_name + ' Transconductance Mean & Standard Deviation')
    color = 'tab:blue'
    ax1.set_xlabel('Gate Voltage Change ($\mu$V)')
    ax1.set_ylabel('Transconductance Standard Deviation (mS)', color=color) 
    ax1.plot(1000000*np.array(new_gate_bins), 1000*np.array(sd_binned_transconductance), color=color, marker = '+')
    ax1.tick_params(axis='y', labelcolor=color)
    #ax1.set_xscale('symlog')
    ax1.set_xlim([-70,70])
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax1.set_yscale('symlog')
    ax2.set_ylabel('Transconductance Mean (mS)', color=color)  # we already handled the x-label with ax1
    ax2.plot(1000000*np.array(new_gate_bins), 1000*np.array(mean_binned_transconductance), marker = 'x', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.tight_layout()
    
    return binned_transconductance, trans_gate_dif
    
if transconductance_spread_analysis == 'yes':

    bin_no_filter, trans_gate_dif_no_filter = transconductance_spread(gate_voltage, drain_current, 'No Filter', bin_width_value)
    bin_bandpass, trans_gate_dif_bandpass = transconductance_spread(bandpassed_gate_voltage, bandpassed_drain_current, 'Bandpassed', bin_width_value)
    bin_notch, trans_gate_dif_notch = transconductance_spread(bandpassed_gate_voltage, notched_bandpassed_drain_current, 'Notch/Bandpass', bin_width_value)

"""------------------------------------------------------------------------"""

# OECT Simulation

def OECT_simulation(gate_voltage_type, measured_transconductance_value, measured_data, drain_current_type, type_name, bin_width_, noise_lower, noise_upper, noise_step):
    
    gate_noise_sd_values = np.arange(noise_lower,noise_upper,noise_step)
    spread_difference_values = []
    for i in gate_noise_sd_values:
        noise_sd = i
        sim_gate = gate_voltage_type
        noise_mean = 0 
        noise = np.random.normal(noise_mean,noise_sd,len(gate_voltage_type))
        sim_gate_noise = sim_gate + noise
        sim_oect = [drain_current_type[0]]
        for j in range(len(sim_gate_noise)-1):
            gate_dif = np.float(sim_gate_noise[j+1]) - np.float(sim_gate_noise[j])
            next_drain = np.float(gate_dif)*np.float(measured_transconductance_value) + np.float(sim_oect[j])
            sim_oect.append(next_drain)
        sim_oect = np.array(sim_oect)
        transconductance_gate_difference_sim = []
        for j in range(len(sim_gate) - 1):
            gate = sim_gate[j]
            next_gate = sim_gate[j+1]
            drain = sim_oect[j]
            next_drain = sim_oect[j+1]
            gate_difference_value = next_gate - gate
            if gate_difference_value != 0:
                transconductance_value = (next_drain - drain) / gate_difference_value
                transconductance_gate_difference_sim.append([gate_difference_value,transconductance_value])
        trans_gat_dif_sim = transconductance_gate_difference_sim
        
        transconductance_gate_difference_sim = sorted(transconductance_gate_difference_sim)
        # put the gate changes into bins/lists then find the mean and sd of those lists
        # first_gate = np.round(transconductance_gate_difference_sim[0][0],7) #round to 1e-7
        # last_gate = np.round(transconductance_gate_difference_sim[-1][0],7)
        first_gate = np.round(measured_data[0][0],7) #round to 1e-7
        last_gate = np.round(measured_data[-1][0],7)
        bin_width = bin_width_
        gate_bins = np.arange(first_gate, last_gate + bin_width, bin_width)
        binned_transconductance_values = [[] for k in gate_bins]
        for j in range(len(transconductance_gate_difference_sim)):
            gate = transconductance_gate_difference_sim[j][0] # round to nearest 1e-7
            gate = np.round(gate, 7)
            transconductance_val = transconductance_gate_difference_sim[j][1]
            gate_array = np.empty(len(gate_bins))
            gate_array.fill(gate)
            gate_differences = gate - gate_bins
            gate_differences_abs = np.abs(gate_differences)
            closest_gate_index = list(gate_differences_abs).index(min(gate_differences_abs))
            binned_transconductance_values[closest_gate_index].append(transconductance_val)
        binned_transconductance = [[gate_bins[k], binned_transconductance_values[k]] for k in range(len(gate_bins)) if len(binned_transconductance_values[k]) > 10]        
        for k in range(len(binned_transconductance)): # remove outliers
            error_values = binned_transconductance[k][1]
            deviation = 1.5*(stats.iqr(error_values))
            upper_limit = np.percentile(error_values, 75) + deviation
            lower_limit = np.percentile(error_values, 25) - deviation
            new_error_values = [j for j in error_values if j > lower_limit]
            new_error_values = [j for j in new_error_values if j < upper_limit]
            binned_transconductance[k][1] = new_error_values
        if noise_step < 1:
            new_gate_bins = [k[0] for k in binned_transconductance]
            mean_binned_transconductance = [np.mean(k[1]) for k in binned_transconductance]
            sd_binned_transconductance = [np.sqrt(np.var(k[1])) for k in binned_transconductance]
            new_gate_bins_measured = [k[0] for k in measured_data]
            mean_binned_transconductance_measured = [np.mean(k[1]) for k in measured_data]
            sd_binned_transconductance_measured = [np.sqrt(np.var(k[1])) for k in measured_data]
            spread_difference = []
            for j in range(int(min([len(sd_binned_transconductance_measured), len(sd_binned_transconductance)]))):
                if sd_binned_transconductance[j] < 0.5:
                        if sd_binned_transconductance_measured[j] < 0.5:
                            spread_difference_value = (sd_binned_transconductance_measured[j] - sd_binned_transconductance[j])**2
                            if spread_difference_value < 10e6:
                                spread_difference.append(spread_difference_value)
            total_spread_difference = np.sum(spread_difference)
            spread_difference_values.append(total_spread_difference)
            print('Completed analysis for ', str(np.around(1e6*noise_sd)), ' uV on ', type_name)
            min_spread = min(spread_difference_values)
            min_spread_gate_noise = gate_noise_sd_values[spread_difference_values.index(min_spread)]
        else:
            min_spread_gate_noise = gate_noise_sd_values[0]
    prob_95 = 4*min_spread_gate_noise
    print('The minimum difference in spread was with a sd of ', str(1e6*min_spread_gate_noise), ' uV .')
    print('Therefore, we are 95% sure that a gate change does not come from noise if greater than 4*sd =', str(np.round(1e6*prob_95)), ' uV .')  
    return min_spread_gate_noise, binned_transconductance, trans_gat_dif_sim

def plot_optimal(measured_bins, simulated_bins, type_name):
    
    gate_bins_measured = [k[0] for k in measured_bins]
    gate_bins_sim = [k[0] for k in simulated_bins]
    sd_binned_transconductance_measured = [np.sqrt(np.var(k[1])) for k in measured_bins]
    sd_binned_transconductance_sim = [np.sqrt(np.var(k[1])) for k in simulated_bins]
    mean_binned_transconductance_measured = [np.mean(k[1]) for k in measured_bins]
    mean_binned_transconductance_sim = [np.mean(k[1]) for k in simulated_bins]

    plt.figure(figsize = (7.5,6))
    plt.xlabel('Gate Voltage Change ($\mu$V)')
    plt.ylabel('Transconductance Standard Deviation (mS)')
    plt.title(type_name + ' Binned Transconductance Standard Deviation')
    plt.yscale('symlog')
    #plt.xscale('symlog')
    plt.xlim([-85, 85])
    plt.ylim([0, 1400*max(sd_binned_transconductance_measured)])
    plt.plot(1000000*np.array(gate_bins_measured), 1000*np.array(sd_binned_transconductance_measured), color='r', marker = '+', label = 'Measured')
    plt.plot(1000000*np.array(gate_bins_sim), 1000*np.array(sd_binned_transconductance_sim), marker = 'x', color='b', label = 'Simulated')
    plt.legend()
    """
    plt.figure(figsize = (9,7))
    plt.xlabel('Gate Voltage Change ($\mu$V)')
    plt.ylabel('Transconductance Mean (mS)')
    #plt.xscale('symlog')
    plt.title(type_name + ' Binned Transconductance Mean Measured vs Simulated')
    plt.plot(1000000*np.array(gate_bins_measured), 1000*np.array(mean_binned_transconductance_measured), color='r', marker = '+', label = 'Measured')
    plt.plot(1000000*np.array(gate_bins_sim), 1000*np.array(mean_binned_transconductance_sim), marker = 'x', color='b', label = 'Simulated')
    plt.legend()
    """
    return

if simulate_oect == 'yes':
    
    sigma_values_no_filter = []
    sigma_values_bandpass = []
    sigma_values_notch = []
    count = 0
    for i in range(3): # probabilitic process so do many times and take average
        optimal_sigma_value_no_filter, binned_trans = OECT_simulation(gate_voltage, raw_transconductance_mean, bin_no_filter, drain_current, 'No Filter', bin_width_value, noise_lower_, noise_upper_, noise_step_)
        sigma_values_no_filter.append(optimal_sigma_value_no_filter)
        optimal_sigma_value_bandpass, binned_bandpass = OECT_simulation(bandpassed_gate_voltage, raw_transconductance_mean, bin_bandpass, bandpassed_drain_current, 'Bandpass', bin_width_value, noise_lower_, noise_upper_, noise_step_)
        sigma_values_bandpass.append(optimal_sigma_value_bandpass)
        optimal_sigma_value_notch, binned_notch, trans_gat_dif_sim_notch = OECT_simulation(bandpassed_gate_voltage, raw_transconductance_mean, bin_notch, notched_bandpassed_drain_current, 'Notch', bin_width_value, noise_lower_, noise_upper_, noise_step_)
        sigma_values_notch.append(optimal_sigma_value_notch)
        count = count + 1
        print('Completed ', count)
    print('No filter sigma value mean ', np.mean(1e6*np.mean(sigma_values_no_filter)) , ' and sd ', 1e6*np.sqrt(np.var(sigma_values_no_filter)) ) 
    print('Bandpass sigma value mean ', np.mean(1e6*np.mean(sigma_values_bandpass)) , ' and sd ', 1e6*np.sqrt(np.var(sigma_values_bandpass)) ) 
    print('Notch sigma value mean ', np.mean(1e6*np.mean(sigma_values_notch)) , ' and sd ', 1e6*np.sqrt(np.var(sigma_values_notch)) ) 
    
    # Find the mean values of the optimal values as found by the RSS
    optimal_sigma_value_no_filter = np.mean(np.mean(sigma_values_no_filter))
    optimal_sigma_value_bandpass = np.mean(np.mean(sigma_values_bandpass))
    optimal_sigma_value_notch = np.mean(np.mean(sigma_values_notch))
    
    #optimal_sigma_value_no_filter, binned_none = OECT_simulation(gate_voltage, raw_transconductance_mean, bin_no_filter, drain_current, 'No Filter', bin_width_value, optimal_sigma_value_no_filter, optimal_sigma_value_no_filter+1, 10)
    #optimal_sigma_value_bandpass, binned_bandpass = OECT_simulation(bandpassed_gate_voltage, raw_transconductance_mean, bin_bandpass, bandpassed_drain_current, 'Bandpass', bin_width_value, optimal_sigma_value_bandpass, optimal_sigma_value_bandpass+1, 10)
    #optimal_sigma_value_notch, binned_notch, trans_gat_dif_sim_notch = OECT_simulation(bandpassed_gate_voltage, raw_transconductance_mean, bin_notch, notched_bandpassed_drain_current, 'Notch', bin_width_value, optimal_sigma_value_notch, optimal_sigma_value_notch+1, 10)
    
    #plot_optimal(bin_no_filter, binned_none, 'No Filter')
    #plot_optimal(bin_bandpass, binned_bandpass, 'Bandpass')
    plot_optimal(bin_notch, binned_notch, 'Notch/Bandpass')


