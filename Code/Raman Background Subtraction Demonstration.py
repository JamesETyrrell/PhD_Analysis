"""
Raman Characterisation

@author: James Tyrrell
"""

""" --------------------------------------------------- """
#Imports and Settings

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
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
import scipy as spy
from operator import itemgetter
import more_itertools as mit
from collections import OrderedDict

plt.rcParams["font.family"] = 'serif'
plt.rcParams.update({'font.size': 14})

# Options
analysis_type = ['Raw Data', 'Background Subtracted', 'Background Subtracted and Normalised']
noise_reduction = ['No Noise Reduction', 'Noise Reduction']

# Choices
analysis_type = 'Background Subtracted and Normalised'
noise_reduction = 'Noise Reduction'
sampling_frequency = 100
high_cutoff = 15
high_butterworth_order = 3
high_normalised_cutoff = high_cutoff / (sampling_frequency / 2)
number_of_peaks_ = 2

"""
--------------------------------------------------------------------
"""


def baseline_subtration(wavelength, intensity):
    
    sampling_frequency = 100
    high_cutoff = 20
    high_butterworth_order = 3
    high_normalised_cutoff = high_cutoff / (sampling_frequency / 2)
    d, c = signal.butter(high_butterworth_order, high_normalised_cutoff, btype='lowpass')
    raw_low_noise = signal.filtfilt(d, c, intensity)
       
    ints = pd.Series(raw_low_noise)
    rolling_length = 10
    rolling_std = ints.rolling(rolling_length).std()
    rolling_min = ints.rolling(rolling_length).min()
    reduction = []
    for i in range(len(raw_low_noise)):
        if np.array(rolling_std)[i] < np.percentile(np.array(rolling_std[rolling_length:]), 40):
            reduction.append(np.array(rolling_min)[i])
        elif i < 2*rolling_length:
            reduction.append(np.array(rolling_min)[i])
        elif i == len(intensity) - 1:
            reduction.append(raw_low_noise[i])
        else:
            reduction.append(-1000)
    
    if reduction[0] == -1000:
        reduction[0] = reduction[0]
    if reduction[-1] == -1000:
        reduction[-1] = reduction[-1]
    for i in range(len(reduction) - 1):
        if reduction[i] > 0:
            if reduction[i+1] == -1000:      
                start_index = i
                for j in range(i+1,len(reduction) - 1):
                    if reduction[j] == -1000:
                        if reduction[j+1] > 0:
                            end_index = j+1
                            break
                start = reduction[start_index]
                end = reduction[end_index]
                gradient = (end - start)/(end_index - start_index)
                count = 1
                for j in range(start_index+1,end_index+1):
                    if reduction[j] == -1000:
                        reduction[j] = start + count*gradient
                        count += 1
                        break                
 
    low_cutoff = 3
    low_butterworth_order = 3
    low_normalised_cutoff = low_cutoff / (sampling_frequency / 2)
    b, a = signal.butter(low_butterworth_order, low_normalised_cutoff, btype='lowpass')
    baseline_reduction = signal.filtfilt(b, a, reduction[rolling_length:])
    smaller_intensity = np.array(intensity[rolling_length:])
    
    # background trend
    low_cutoff = 1
    low_butterworth_order = 3
    low_normalised_cutoff = low_cutoff / (sampling_frequency / 2)
    b, a = signal.butter(low_butterworth_order, low_normalised_cutoff, btype='lowpass')
    baseline_reduction_background = signal.filtfilt(b, a, baseline_reduction)
    
    # fast trend
    low_cutoff = 10
    low_butterworth_order = 3
    low_normalised_cutoff = low_cutoff / (sampling_frequency / 2)
    b, a = signal.butter(low_butterworth_order, low_normalised_cutoff, btype='lowpass')
    baseline_reduction_fast = signal.filtfilt(b, a, intensity[rolling_length:])
    
    gradient_trend_length = 100
    trends = []
    for i in range(len(baseline_reduction_background)):
        try:
            this_trend = []
            values = baseline_reduction_background[i - int(gradient_trend_length/2):i + int(gradient_trend_length/2)]
            for i in range(len(values) - 1):
                if values[i+1] > values[i]:
                    this_trend.append(1)
                else:
                    this_trend.append(-1)
            direction = np.sign(np.sum(this_trend))
        except:
            direction = 0
        trends.append(direction)

    for i in range(len(baseline_reduction)):
        if smaller_intensity[i] < baseline_reduction[i]:
            baseline_reduction[i] = smaller_intensity[i]    

    baseline_reduction_copy = []
    identified_indices = []
    for i in range(len(baseline_reduction)-1):
        baseline_reduction_change = baseline_reduction[i+1] - baseline_reduction[i]
        background_change = baseline_reduction_background[i+1] - baseline_reduction_background[i]
        if np.sign(baseline_reduction_change) == -1*np.sign(background_change):
            baseline_reduction_copy.append(-1000)
            identified_indices.append(i)
        elif np.sign(baseline_reduction_change) == -1*np.sign(trends[i]):
            baseline_reduction_copy.append(-1000)
            identified_indices.append(i)
        else:
            baseline_reduction_copy.append(baseline_reduction[i])
    baseline_reduction_copy.append(baseline_reduction[-1])
    
    grouped_indices = [list(group) for group in mit.consecutive_groups(identified_indices)]
    final_indices_in_group = [i[-1] for i in grouped_indices]
    for each_final_index in final_indices_in_group:
        baseline_value = baseline_reduction_copy[each_final_index+1]
        for j in range(0, each_final_index+1):
            if np.sign(baseline_reduction_fast[j] - baseline_value) == np.sign(baseline_reduction_background[j+1] - baseline_reduction_background[j]):
                baseline_reduction_copy[j] = -1000
    baseline_reduction = baseline_reduction_copy
    
    baseline_reduction = baseline_reduction_copy
    if baseline_reduction[0] == -1000:
        baseline_reduction[0] = smaller_intensity[0]
    if baseline_reduction[-1] == -1000:
        baseline_reduction[-1] = smaller_intensity[-1]
    for i in range(len(baseline_reduction) - 1):
        if baseline_reduction[i] > 0:
            if baseline_reduction[i+1] == -1000:      
                start_index = i
                for j in range(i+1,len(baseline_reduction) - 1):
                    if baseline_reduction[j] == -1000:
                        if baseline_reduction[j+1] > 0:
                            end_index = j+1
                            break
                start = baseline_reduction[start_index]
                end = baseline_reduction[end_index]
                gradient = (end - start)/(end_index - start_index)
                count = 1
                for j in range(start_index+1,end_index+1):
                    if baseline_reduction[j] == -1000:
                        baseline_reduction[j] = start + count*gradient
                        count += 1
                        break  
                    
    for i in range(len(baseline_reduction)):
        if smaller_intensity[i] < baseline_reduction[i]:
            baseline_reduction[i] = smaller_intensity[i]   

    low_cutoff = 3
    low_butterworth_order = 1
    low_normalised_cutoff = low_cutoff / (sampling_frequency / 2)
    b, a = signal.butter(low_butterworth_order, low_normalised_cutoff, btype='lowpass')
    baseline_reduction = signal.filtfilt(b, a, baseline_reduction)

    reduced_intensity = smaller_intensity - baseline_reduction
    
    plt.figure()
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Raman Intensity (A.U.)')
    plt.plot(raman_wavelength[rolling_length:], intensity[rolling_length:], label = 'Unfiltered Data')
    plt.plot(raman_wavelength[rolling_length:], baseline_reduction, label = 'Baseline Subtraction')
    plt.grid()
    plt.tight_layout()
    plt.legend()
    
    plt.figure()
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Raman Intensity (A.U.)')
    plt.title('Baseline Subtracted Raman Intensity')
    plt.plot(raman_wavelength[rolling_length:], reduced_intensity)
    plt.grid()
    plt.tight_layout()
    
    return reduced_intensity


def main_peak_analysis(wls, ints, number_of_peaks):
    
    sampling_frequency = 100
    high_cutoff = 10
    high_butterworth_order = 3
    high_normalised_cutoff = high_cutoff / (sampling_frequency / 2)
    d, c = signal.butter(high_butterworth_order, high_normalised_cutoff, btype='lowpass')
    reduced_low_noise = signal.filtfilt(d, c, ints)
    
    gradients = []
    for i in range(len(reduced_low_noise)-1):
        this_grad = reduced_low_noise[i+1] - reduced_low_noise[i]
        gradients.append(this_grad)
    turning_points = []
    for i in range(len(gradients)-1):
        if gradients[i] > 0:
            if gradients[i+1] < 0:
                turning_points.append(i+1)
    # now compare with intensity input
    turning_points_corrected = []
    surroundings = 3
    for i in turning_points:
        points_to_test = np.arange(i-surroundings, i+surroundings+1, 1, dtype = int)
        intensities = [ints[j] for j in points_to_test]
        max_intensity = intensities.index(max(intensities))
        true_max_index = points_to_test[max_intensity]
        turning_points_corrected.append(true_max_index)
    turning_point_values = [[i,ints[i]] for i in turning_points_corrected]
    sorted_turns = sorted(turning_point_values, key = itemgetter(1))
    sorted_turns = sorted_turns[::-1]
    sorted_turns = sorted_turns[:number_of_peaks]
    relative_intensity_max_to_min = np.zeros(shape = (number_of_peaks))
    for i in range(len(sorted_turns)):
        intensity_ratio = sorted_turns[i][1]/sorted_turns[0][1]
        relative_intensity_max_to_min[i] = intensity_ratio
    print('Relative intensities:\n', relative_intensity_max_to_min)
    
    # Find simple width of largest peak
    peak_index = sorted_turns[0][0]
    half_max_of_peak = sorted_turns[0][1]/2
    for i in range(peak_index, 0, -1):
        if ints[i] >= half_max_of_peak:
            if ints[i-1] < half_max_of_peak:               
                x0 = wls[i-1]
                x1 = wls[i]
                y0 = ints[i-1]
                y1 = ints[i]
                y = 0.5
                upper_width = x1*((y-y0)/(y1-y0)) + x0*((y1-y)/(y1-y0))
                break
    for i in range(peak_index, len(ints)-1):
        if ints[i] >= half_max_of_peak:
            if ints[i+1] < half_max_of_peak:
                x0 = wls[i+1]
                x1 = wls[i]
                y0 = ints[i+1]
                y1 = ints[i]
                y = 0.5
                lower_width = x1*((y-y0)/(y1-y0)) + x0*((y1-y)/(y1-y0))
                break       
    peak_width = upper_width - lower_width
    peak_width_values = [lower_width, upper_width]
    print('Peak width: ', peak_width_values)
    print('Peak width: ', peak_width)
    
    return sorted_turns, peak_width

end = '?raw=true'
raman_url = 'https://github.com/JamesETyrrell/PhD_Analysis/blob/959fde4115945a03dd4eeeb2e350205afa675260/Data/Raman%20Spectra%20Example.txt' + end
raman_data = pd.read_csv(raman_url, sep = '\t')
raman_data = raman_data.drop(columns = '#Intensity')
raman_data = raman_data.rename(columns={'#Wave': 'Wavenumber', 'Unnamed: 1': 'Intensity'})

raman_wavelength = list(raman_data['Wavenumber'])
raman_intensity = list(raman_data['Intensity'])
data_length = len(raman_wavelength)
first = int(0.1*data_length)
last = int(0.9*data_length)
raman_wavelength_filtered = raman_wavelength[first:last]
d, c = signal.butter(high_butterworth_order, high_normalised_cutoff, btype='lowpass')
if analysis_type == 'Raw Data':               
    if noise_reduction == 'No Noise Reduction':
        final_raman_intensity = raman_intensity
    else:  
        raw_low_noise = signal.filtfilt(d, c, raman_intensity)
        final_raman_intensity = raw_low_noise[first:last]
elif analysis_type == 'Background Subtracted':
    low_pass_noisy = baseline_subtration(raman_wavelength, raman_intensity)
    low_pass_noisy = low_pass_noisy[first:last]
    if noise_reduction == 'No Noise Reduction':
        final_raman_intensity = low_pass_noisy
    else:
        low_pass_low_noise = signal.filtfilt(d, c, low_pass_noisy)
        final_raman_intensity = low_pass_low_noise
elif analysis_type == 'Background Subtracted and Normalised':        
    low_pass_noisy = baseline_subtration(raman_wavelength, raman_intensity)
    low_pass_noisy = low_pass_noisy[first:last]
    if noise_reduction == 'No Noise Reduction':
        low_pass_noisy_normalised = np.array(low_pass_noisy)/max(np.array(low_pass_noisy))
        peak_values_lpnn, peak_width_lpnn = main_peak_analysis(raman_wavelength_filtered, low_pass_noisy_normalised, number_of_peaks_)
        peak_wavelengths_lpnn = [raman_wavelength_filtered[i[0]] for i in peak_values_lpnn]
        peak_intensities_lpnn = [i[1] for i in peak_values_lpnn]
        print('Peak Wavelengths from highest to lowest intensity:\n', peak_wavelengths_lpnn)
        final_raman_intensity = low_pass_noisy_normalised
    else:
        low_pass_low_noise = signal.filtfilt(d, c, low_pass_noisy)
        low_pass_low_noise_normalised = np.array(low_pass_low_noise)/max(np.array(low_pass_low_noise))
        peak_values_lplnn, peak_width_lplnn = main_peak_analysis(raman_wavelength_filtered, low_pass_low_noise_normalised, number_of_peaks_)
        peak_wavelengths_lplnn = [raman_wavelength_filtered[i[0]] for i in peak_values_lplnn]
        peak_intensities_lplnn = [i[1] for i in peak_values_lplnn]
        print('Peak Wavelengths from highest to lowest intensity:\n', peak_wavelengths_lplnn)
        final_raman_intensity = low_pass_low_noise_normalised

plt.figure()
plt.xlabel('Wavenumber (cm$^{-1}$)')
plt.ylabel('Raman Intensity (A.U.)')
plt.title('Additional Filtering')
plt.grid()
plt.plot(raman_wavelength_filtered, final_raman_intensity)
plt.tight_layout()

