# -*- coding: utf-8 -*-
"""
 OECT Characterisation

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
import requests

#plt.close('all')
plt.rcParams["font.family"] = 'serif'
plt.rcParams.update({'font.size': 16})

""" --------------------------------------------------- """
# Characterisation Funcitons

def conductance(res_data):
    conductance = res_data['Conductance']
    conductance_mean = conductance.mean()
    print('Conductance: ', str(float('%.3g' % conductance_mean)), ' S (3 s.f.)')
    return conductance_mean


def output(output_data):

    gates_raw = output_data['Gate Voltage']
    gates = []
    for i in gates_raw:
        i = round(i, ndigits=1)
        if i not in gates:
            gates.append(i)
    number_of_gates = len(gates)
    output = np.zeros(shape = [int((len(output_data))/number_of_gates)+1, 2*number_of_gates])
    for i in range(number_of_gates):
        output[0,2*i] = gates[i]
        output[0,2*i+1] = gates[i]
    this_gate_index = 0
    iteration = 1
    for line in range(len(output_data)):
        data = output_data.iloc[line]
        gate_voltage = round(float(data['Gate Voltage']), ndigits=1)
        gate_index = gates.index(gate_voltage)
        if this_gate_index != gate_index: 
            this_gate_index = this_gate_index + 1 
            iteration = 1
        drain_voltage = data['Drain Voltage']
        drain_current = data['Drain Current']
        output[iteration, 2*this_gate_index] = drain_voltage
        output[iteration, 2*this_gate_index + 1] = drain_current
        iteration = iteration + 1   
        
    # find the conductance value
    conductance_search = output[2:20,0:2]
    conductance_value_output = (conductance_search[-1,1] - conductance_search[0,1])/(conductance_search[-1,0] - conductance_search[0,0])
    print('Output Conductance: ', conductance_value_output, ' S')    
        
    # 2 sweeps identified by having duplicate values in drain voltage
    for i in range(len(output)):
        if i == int(len(output)/4):
            a = output[i+1,0] - output[i,0]
        if i == 3*int(len(output)/4):
            b = output[i+1,0] - output[i,0]
    if a*b > 0:
        first_sweep_values = len(output)
    else:
        first_sweep_values = int(np.ceil(len(output))/2)
        
    # Find pinch-off     
    gate_to_find_pinch_off = 0.5
    zero_gate_index = 2*gates.index(gate_to_find_pinch_off) + 1 # misleading variable name
    zero_gate_drain_output = output[1: , zero_gate_index]
    zero_gate_drain_voltage = output[1: , zero_gate_index-1]
    moving_average_window = 10
    sma_output = pd.Series(zero_gate_drain_output).rolling(window=moving_average_window).mean().iloc[moving_average_window-1:].values 
    sma_drain_voltage = pd.Series(zero_gate_drain_voltage).rolling(window=moving_average_window).mean().iloc[moving_average_window-1:].values 
    sma_differential = np.zeros(len(sma_output)-1)
    # determine sweep switch for smoothed drain voltage
    first_sweep_values_sma = -1
    for i in range(len(sma_drain_voltage) - 2):
        if sma_drain_voltage[i] > sma_drain_voltage[i+1]:
            if sma_drain_voltage[i+2] > sma_drain_voltage[i+1]:
                first_sweep_values_sma = i
                break
        elif sma_drain_voltage[i] < sma_drain_voltage[i+1]:
            if sma_drain_voltage[i+2] < sma_drain_voltage[i+1]:
                first_sweep_values_sma = i
                break
    for i in range(len(sma_output) - 1):
        if sma_drain_voltage[i] - sma_drain_voltage[i-1] == 0:
            sma_differential[i] = sma_differential[i-1]
        else:
            sma_differential[i] = ( sma_output[i + 1] - sma_output[i] ) / (sma_drain_voltage[i] - sma_drain_voltage[i-1] )
    # Saturation region begins when gradient at Vd = 'x' V is p% of that at Vd = +0.5 V
    saturation_percentage = 15
    compare_drain_value = max(list(sma_drain_voltage))
    compare_drain_index = list(sma_drain_voltage).index(compare_drain_value) # index where Vd = VdMax
    gradient_compare_drain = sma_differential[compare_drain_index]
    pinch_current_grad = (saturation_percentage / 100)*gradient_compare_drain  # gradient of pinch current for 30% saturation level   
    pinch_off_index = first_sweep_values_sma
    for i in range(first_sweep_values_sma):
        if np.abs(sma_differential[i]) < np.abs(pinch_current_grad):
            pinch_off_index = i  
            break
    sma_differential = pd.Series(sma_differential).rolling(window=moving_average_window).mean().iloc[moving_average_window-1:].values
    pinch_off_voltage_uncorrected =  -1*sma_drain_voltage[pinch_off_index]
    pinch_off_voltage = pinch_off_voltage_uncorrected + gate_to_find_pinch_off
    print('Pinch-off voltage: ', str(float('%.3g' % pinch_off_voltage)), ' V (3 s.f.)')
    
    plt.figure(figsize=(5,4))
    # plt.title('Output Characteristics')
    plt.xlabel('Drain Voltage (V)')
    plt.ylabel('Drain Current (mA)')
    drain_voltage = output[1:first_sweep_values+1,0]
    for i in range(number_of_gates):
        drain_voltage = output[1:first_sweep_values+1,2*i]
        drain_current = 1e3*output[1:first_sweep_values+1,2*i+1]
        this_gate = output[0,2*i]
        plt.plot(drain_voltage, drain_current, linewidth=3, label = str(this_gate) + 'V')
    plt.legend(ncol = 1, fontsize = 11)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig('Output.png')
   
    return pinch_off_voltage, conductance_value_output


def transfer(transfer_data):
    
    gate_voltage = transfer_data['Gate Voltage']
    drain_current = transfer_data['Drain Current']
    gate_current = transfer_data['Gate Current']
    transfer = np.zeros(shape = [len(gate_voltage), 4])
    transfer[:,0] = np.array(gate_voltage)
    transfer[:,1] = np.array(drain_current)
    transfer[:,3] = np.array(gate_current)
    for i in range(len(transfer)):
        if i == int(len(transfer)/4):
            a = transfer[i+1,0] - transfer[i,0]
        if i == 3*int(len(transfer)/4):
            b = transfer[i+1,0] - transfer[i,0]
    if a*b > 0:
        first_sweep_number = len(transfer)
    else:
        first_sweep_number = int(np.ceil(len(transfer))/2)
    
    for i in range(len(transfer)-1):
        this_gate = transfer[i , 0]
        next_gate = transfer[i+1, 0]
        d_gate = next_gate - this_gate
        this_current = transfer[i , 1]
        next_current = transfer[i+1 , 1]
        d_current = next_current - this_current
        if d_gate == 0:
            transconductance = transfer[i-1 , 2]
        else:
            transconductance = d_current / d_gate
        transfer[i+1, 2] = transconductance
        
    # # Determine parameters from transfer curve
    transfer_indices = [i for i in range(len(gate_voltage)) if i != first_sweep_number+1]
    transfer = transfer[transfer_indices,:]
    transfer[1:,2] = gaussian_filter(transfer[1:,2], sigma=3)
    # high_cutoff = 3
    # butterworth_order = 2
    # sampling_frequency = 100
    # normalised_cutoff = high_cutoff / (sampling_frequency / 2) # Normalize the frequency
    # d, c = signal.butter(butterworth_order, normalised_cutoff, btype='lowpass')
    # transfer[0:first_sweep_number,2] = signal.filtfilt(d, c, transfer[0:first_sweep_number,2])
    # transfer_sweep = transfer[first_sweep_number:,2]
    # transfer_sweep_reverse = transfer_sweep[::-1]
    # filtered_transfer_reverse = signal.filtfilt(d, c, transfer_sweep_reverse)
    # filtered_transfer = filtered_transfer_reverse[::-1]
    # transfer[first_sweep_number:,2] = filtered_transfer
    gate_voltage = transfer[: , 0]
    current = transfer[: , 1]
    transconductance = transfer[: , 2]
    #transfer[1:,2] = gaussian_filter(transfer[1:,2], sigma=3)
      
    peak_transconductance = np.ndarray.max(transconductance[0:first_sweep_number]) 
    peak_gm_vg_index = list(transconductance[0:first_sweep_number]).index(peak_transconductance)
    peak_gm_vg = gate_voltage[peak_gm_vg_index]
    print('Maximum transconductance is ' + str(peak_transconductance) + ' S')
    print('Gate Voltage of Maximum Transcondcutance: ', str(float('%.3g' % peak_gm_vg)), ' V (3 s.f.)')
    on_current = np.min(current[0:first_sweep_number])
    # print('The on current was ', on_current, 'A')
    off_current = np.max(current[0:first_sweep_number])   
    off_current_voltage_index = list(current[0:first_sweep_number]).index(off_current)
    off_current_voltage = gate_voltage[off_current_voltage_index]
    # print('The off current was ', off_current, 'A  at a voltage of ', off_current_voltage, 'V')
    switch_ratio = on_current/off_current
    # print('The switch on/off ratio was ', switch_ratio) 
    
    fig, ax1 = plt.subplots(figsize=(5,4))
    #plt.title('LC LI Transfer')
    color = 'tab:blue'
    ax1.set_xlabel('Gate Voltage (V)')
    ax1.set_ylabel('Drain Current (mA)', color=color) 
    if a*b < 0: # assumes forward then back sweep
        gate_forward = gate_voltage[0:first_sweep_number]
        gate_back = gate_voltage[first_sweep_number:]
        current_forward = current[0:first_sweep_number]
        current_back = current[first_sweep_number:]
        ax1.plot(gate_forward[10:], 1e3*np.array(current_forward[10:]), color=color, linewidth = 2, linestyle = 'solid')
        #ax1.plot(gate_back, np.array(current_back), color=color, linestyle = 'dashdot')
        #ax1.plot([],[], color = 'k', linestyle = 'solid', label = 'Forward Sweep')
        #ax1.plot([],[], color = 'k', linestyle = 'dashdot', label = 'Backward Sweep')
    else:
        ax1.plot(gate_voltage, 1e3*np.array(current), color=color, linewidth = 2, linestyle = 'solid')
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Transconductance (mS)', color=color)  # we already handled the x-label with ax1
    if a*b < 0: 
        trans_forward = transconductance[:first_sweep_number]
        trans_back = transconductance[first_sweep_number:]
        ax2.plot(gate_forward[20:], 1e3*np.array(trans_forward[20:]), color=color, linewidth = 2, linestyle = 'solid')
        #ax2.plot(gate_back[:-10], np.array(trans_back[:-10]), color=color, linestyle = 'dashdot')
    else:
        ax2.plot(gate_voltage, 1e3*np.array(transconductance), color=color, linewidth = 2, linestyle = 'solid')
    ax2.tick_params(axis='y', labelcolor=color)
    #ax1.legend()
    plt.tight_layout()
    #plt.savefig('Transfer.png')
    
    # fig, ax1 = plt.subplots(figsize=(8,6))
    # plt.title('Transfer Characteristics, $V_D$ = -0.6V')
    # color = 'tab:blue'
    # ax1.set_xlabel('Gate Voltage (V)')
    # ax1.set_ylabel('Gate Current (A)', color=color) 
    # if a*b < 0: # assumes forward then back sweep
    #     gate_forward = gate_voltage[0:first_sweep_number]
    #     gate_back = gate_voltage[first_sweep_number:]
    #     current_forward = gate_current[0:first_sweep_number]
    #     current_back = gate_current[first_sweep_number:]
    #     ax1.plot(gate_forward, np.array(current_forward), color=color, linestyle = 'solid')
    #     ax1.plot(gate_back, np.array(current_back), color=color, linestyle = 'dashdot')
    #     ax1.plot([],[], color = 'k', linestyle = 'solid', label = 'Forward Sweep')
    #     ax1.plot([],[], color = 'k', linestyle = 'dashdot', label = 'Backward Sweep')
    # ax1.set_ylim(ax1.get_ylim()[::-1])
    # ax1.tick_params(axis='y', labelcolor=color)
    # ax1.legend()
    # plt.tight_layout()
    
    return peak_transconductance, on_current, off_current, off_current_voltage, switch_ratio, transfer, peak_gm_vg


def response_time(response_time_data):
    
    response_time_array = np.zeros(shape=[len(response_time_data['Gate Voltage']),5])
    response_time_array[:,0] = np.array(response_time_data['Gate Voltage'])
    response_time_array[:,1] = np.array(response_time_data['Gate Time (ms)'])/1000
    response_time_array[:,2] = np.array(response_time_data['Drain Voltage'])
    response_time_array[:,3] = np.array(response_time_data['Drain Current'])
    response_time_array[:,4] = np.array(response_time_data['Gate Current'])
    response_time_data = response_time_array
    
    sampling_step = response_time_array[2,1] - response_time_array[1,1]
    
    gate_apply_index = 0
    for i in range(len(response_time_data)):
        this_gate = response_time_data[i,0]
        next_gate = response_time_data[i+1,0]
        diff = next_gate-this_gate
        if diff > 0.02:
            gate_apply_index = int(i)
            break
    
    # Integrate gate current from gate apply index onwards
    total_gate_curr = spy.integrate.simps(response_time_array[gate_apply_index:,4], response_time_array[gate_apply_index:,1])
    print('Total Charge Injected: ', total_gate_curr, ' C')
    
    pre_drain_current = response_time_array[gate_apply_index-20,3]
    drain_current_changes = response_time_array[gate_apply_index-100:gate_apply_index+100,3]
    current_jump = min(drain_current_changes)
    exponential_fitting_index = list(response_time_array[0:gate_apply_index+100,3]).index(current_jump)
    if exponential_fitting_index < gate_apply_index:
        exponential_fitting_index = gate_apply_index + 1
    diff_drain_current = []
    drain_current_changes_2 = response_time_array[gate_apply_index:gate_apply_index+5,3]
    for i in range(len(drain_current_changes_2)):
        diff_drain_current.append(drain_current_changes[i+1] - drain_current_changes[i])
    fastest_change_index = list(diff_drain_current).index(max(diff_drain_current))
    exponential_fitting_index = gate_apply_index + fastest_change_index
    exponential_fitting_index = exponential_fitting_index - 3
    number_of_indices_between_gate_apply_and_current_jump = exponential_fitting_index - gate_apply_index
    exponential_fitting_time = response_time_array[exponential_fitting_index,1]
    #print(exponential_fitting_time)
    jump_ratio = current_jump/pre_drain_current
    difference_in_current = pre_drain_current - current_jump
    gate_current_maximum = max(response_time_array[gate_apply_index-100:gate_apply_index+100,4])
    f_factor = difference_in_current/gate_current_maximum
    print('Jump Ratio:', jump_ratio)

    # Response time noise
    res_noise_time = response_time_array[:gate_apply_index-20,1]
    res_noise_curr = response_time_array[:gate_apply_index-20,3]
    # plt.figure()
    # plt.plot(res_noise_time, res_noise_curr)
    sampling_frequency = 1/(res_noise_time[5] - res_noise_time[4])
    # Correction due to B2902A
    current_resolution = [len(str(res_noise_curr[i]).split('.')[1]) for i in range(len(res_noise_curr)) if '.' in str(res_noise_curr[i])]
    current_resolution_2 = [str(res_noise_curr[i]).split('.')[1] for i in range(len(res_noise_curr)) if '.' in str(res_noise_curr[i])]
    counts = []
    for i in current_resolution_2:
        count = 0
        for j in range(len(i)):
            if i[j] == '0':
                count+=1
            else:
                break
        counts.append(count)
    max_count = max(counts)
    current_resolution = [i for i in current_resolution if i < 15]
    current_resolution = max(current_resolution)
    accuracy_number = current_resolution - max_count
    minimum_current_value = (10)**(-1*current_resolution)
    # Bandpass
    low_cutoff = 10
    low_butterworth_order = 3
    noramlised_cutoff = low_cutoff / (sampling_frequency / 2) # Normalize the frequency
    b, a = signal.butter(low_butterworth_order, noramlised_cutoff, btype='highpass')
    res_noise_curr = signal.filtfilt(b, a, res_noise_curr)   
    res_noise_time = res_noise_time[3500:4500]
    res_noise_curr = res_noise_curr[3500:4500]
    res_noise = np.sqrt(np.var(res_noise_curr))/np.abs(pre_drain_current)
    # Correction due to B2902A
    # if minimum_current_value == 1e-7:
    #     if accuracy_number == 5:
    #     # b2902a_squared_error = (np.abs(pre_drain_current)*0.0002)**2
    #     # measured_square_error = np.var(res_noise_curr)
    #     # res_noise = np.sqrt(measured_square_error-b2902a_squared_error)
    #     # res_noise= res_noise/np.abs(pre_drain_current)
    #         res_noise = res_noise/4
    # plt.figure()
    # plt.plot(res_noise_time, res_noise_curr)
    print('Normalised Response time noise:', res_noise)
    print('Response time accuracy: ', current_resolution)
    
    final_current = response_time_array[-1,3]
    threshold_current = pre_drain_current + 0.9*(final_current - pre_drain_current)
    repsond_index = gate_apply_index
    for i in range(gate_apply_index, len(response_time_array[:,3])):
        current = response_time_array[i,3]
        if response_time_array[i,3] > threshold_current:
            respond_index = i
            break
    response_time_1 = (respond_index - gate_apply_index)*sampling_step
    print('Manual response time: ', response_time_1, ' s')
    
    if response_time_1 > 0.01:
        fitting_range = 500
    elif response_time_1 > 0.005:
        fitting_range = 250
    else:
        fitting_range = 100
    fitting_range = len(response_time_data)
        
    # Simple exponential fitting
    times = response_time_data[exponential_fitting_index:exponential_fitting_index+fitting_range,1]
    drain_values_to_fit = response_time_data[exponential_fitting_index:exponential_fitting_index+fitting_range,3]      
    def fit_func(time, amp, t_const, c):
        exp_current = c - amp*(1-np.exp(-1*(times-exponential_fitting_time)/t_const))
        return exp_current
    bounds_ = ((-np.inf, 0, -np.inf), (np.inf, 1, np.inf))
    p0, p0_cov = spo.curve_fit(fit_func, times, drain_values_to_fit, bounds = bounds_, maxfev=100000)
    amplitude = p0[0]
    time_constant = p0[1]
    response_time = np.log(10)*time_constant
    response_time = response_time + number_of_indices_between_gate_apply_and_current_jump*sampling_step
    print('Response time is ', str(float('%.3g' % response_time)), ' s (3 s.f.)')
    # time_constant_error = p0_cov[1,1]
    #time_shift = p0[2]
    constant = p0[2]
    exp_function_data = np.zeros(shape=[len(response_time_data)-exponential_fitting_index,2])
    for i in range(len(response_time_data)-exponential_fitting_index):
        time_ = response_time_data[i+exponential_fitting_index,1]
        exp_function_data[i,0] = time_
        exp_function_data[i,1] = constant - amplitude*(1-np.exp(-1*(time_-exponential_fitting_time)/time_constant))
    
    # fitting_range = len(response_time_data)
    # times = response_time_data[exponential_fitting_index:exponential_fitting_index+fitting_range,1]
    # drain_values_to_fit = response_time_data[exponential_fitting_index:exponential_fitting_index+fitting_range,3]      
    # # Cottrell fitting as well as exponential
    # def fit_func_2(time_values, amp, t_const, c, k, t1):
    #     #t_const = time_constant
    #     current = c - amp*(1-np.exp(-1*(time_values-t1)/t_const)) - k/np.sqrt((time_values - t1))
    #     # if -np.inf in current:
    #     #     current = c - amp*(1-np.exp(-1*(time_values-t1)/t_const))
    #     return current
    # bounds_ = ((-np.inf, 0, -np.inf, -np.inf, 0.099), (np.inf, 1, np.inf, np.inf, 0.101))
    # p0, p0_cov = spo.curve_fit(fit_func_2, times, drain_values_to_fit, maxfev=100000, bounds = bounds_)
    # amplitude = p0[0]
    # exponential_time_constant = p0[1]
    # new_response_time = np.log(10)*exponential_time_constant
    # new_response_time = new_response_time + number_of_indices_between_gate_apply_and_current_jump*sampling_step
    # print('New response time: ', new_response_time)
    # dc_current_value = p0[2]
    # gate_amplitude = p0[3]
    # time_shift = p0[4]
    # print('Gate amplitude is: ', gate_amplitude)
    # gate_function_data = np.zeros(shape=[len(response_time_data)-exponential_fitting_index,2])
    # for i in range(len(response_time_data)-exponential_fitting_index):
    #     time_ = response_time_data[i+exponential_fitting_index,1]
    #     gate_function_data[i,0] = time_
    #     gate_function_data[i,1] = fit_func_2(time_, amplitude, exponential_time_constant, dc_current_value, gate_amplitude, time_shift)

    # figsize=(5.5,4.5)
    fig, ax1 = plt.subplots(figsize=(5,4))
    color = 'tab:red'
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Drain Current (mA)', color=color)
    ax1.set_title('Response Time = ' + str(np.round(1e6*response_time, decimals = 0)) + r' $\mu$s')
    ax1.plot(1000*response_time_data[:,1], 1000*response_time_data[:,3], color=color)
    ax1.plot(1000*exp_function_data[:,0], 1000*exp_function_data[:,1], color='g', label = 'Exponential\nFit')
    ax1.set_xlim([100-5*response_time, 100+5*response_time])
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Gate Voltage (V)', color=color)  # we already handled the x-label with ax1
    ax2.plot(1000*response_time_data[:,1], response_time_data[:,0], color=color, label = 'Gate Voltage')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_xlim([100-5*1000*response_time, 100+5*1000*response_time])
    #ax1.legend(loc='upper left', fontsize = 12)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    # plt.savefig('I_d response_time_fig.png')
    
    # fig, ax1 = plt.subplots(figsize=(5.5,4.5))
    # color = 'tab:red'
    # #ax1.set_title('New Response Time = ' + str(np.round(1000*new_response_time, decimals = 3)) + ' ms')
    # ax1.set_xlabel('Time (ms)')
    # ax1.set_ylabel('Drain Current Fit with Cottrell (mA)', color=color)
    # ax1.plot(1000*response_time_data[:,1], 1000*response_time_array[:,4], color=color)
    # #ax1.plot(1000*gate_function_data[:,0], 1000*gate_function_data[:,1], color='g', label = 'Exponential\nFit')
    # #ax1.set_xlim([100-5*response_time, 100+5*response_time])
    # ax1.tick_params(axis='y', labelcolor=color)
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # color = 'tab:blue'
    # ax2.set_ylabel('Gate Voltage (V)', color=color)  # we already handled the x-label with ax1
    # ax2.plot(1000*response_time_data[:,1], response_time_data[:,0], color=color, label = 'Gate Voltage')
    # ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_xlim([100-5*1000*response_time, 100+5*1000*response_time])
    # ax1.legend(loc='upper left')
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()
    # plt.savefig('I_g response_time_fig.png')
    
    return response_time, jump_ratio, response_time_1#, gate_constant


def noise(transfer_data, noise_df):
    
    files = os.listdir()
    try:
        gate_voltage = transfer_data['Gate Voltage']
        drain_current = transfer_data['Drain Current']
        transfer = np.zeros(shape = [len(gate_voltage) , 3])
        transfer[:,0] = np.array(gate_voltage)
        transfer[:,1] = np.array(drain_current)
        for i in range(len(transfer)):
            if i == int(len(transfer)/4):
                a = transfer[i+1,0] - transfer[i,0]
            if i == 3*int(len(transfer)/4):
                b = transfer[i+1,0] - transfer[i,0]
        if a*b > 0:
            first_sweep_number = len(transfer)
        else:
            first_sweep_number = int(np.ceil(len(transfer))/2)
        
        for i in range(len(transfer)-1):
            this_gate = transfer[i , 0]
            next_gate = transfer[i+1, 0]
            d_gate = next_gate - this_gate
            this_current = transfer[i , 1]
            next_current = transfer[i+1 , 1]
            d_current = next_current - this_current
            if d_gate == 0:
                transconductance = transfer[i-1 , 2]
            else:
                transconductance = d_current / d_gate
            transfer[i+1, 2] = transconductance
            
        # Determine parameters from transfer curve
            # Determine parameters from transfer curve
        transfer_indices = [i for i in range(len(gate_voltage)) if i != first_sweep_number+1]
        transfer = transfer[transfer_indices,:]
        #transfer[1:,2] = gaussian_filter(transfer[1:,2], sigma=3)
        high_cutoff = 1
        butterworth_order = 3
        sampling_frequency = 100
        normalised_cutoff = high_cutoff / (sampling_frequency / 2) # Normalize the frequency
        d, c = signal.butter(butterworth_order, normalised_cutoff, btype='lowpass')
        transfer[0:first_sweep_number,2] = signal.filtfilt(d, c, transfer[0:first_sweep_number,2])
        transfer[first_sweep_number:,2] = signal.filtfilt(d, c, transfer[first_sweep_number:,2])
        gate_voltage = transfer[: , 0]
        transconductance = transfer[: , 2]     
        zero_gate_gm = max([transconductance[i] for i in range(len(transconductance)) if np.abs(gate_voltage[i]) <= 0.05])
        
    except Exception as e:
        print(e)
    
    noise_type = 1
    try:
        drain_current = noise_df['Drain Current']
    except:
        pass
    try:
        drain_voltage = noise_df['Drain Voltage']
    except:
        pass
    try:
        time = (noise_df['Drain Time (us)'] - noise_df['Drain Time (us)'].iloc[0])/1000000
    except:
        noise_type = 2
        time = noise_df['Gate Time']
    try:
        gate_voltage = noise_df['Gate Voltage']
    except:
        pass
    
    current_resolution = [len(str(drain_current[i]).split('.')[1]) for i in range(len(drain_current)) if '.' in str(drain_current[i])]
    current_resolution = [i for i in current_resolution if i < 15]
    current_resolution = max(current_resolution)
    minimum_current_value = (10)**(-1*current_resolution)
    sampling_frequency = 1/(time.iloc[2] - time.iloc[1])
    current_for_normalisation = drain_current.mean()
    
    # plt.figure()
    # plt.xlabel('Time')
    # plt.ylabel('Drain Current')
    # plt.plot(time, drain_current)
    # plt.tight_layout()
    
    # notch filters - optional
    notched_drain_current = drain_current
    frequency_to_remove = np.arange(50,300,100)
    for i in frequency_to_remove:
        noramlised_frequency = i/ (sampling_frequency / 2)
        Q = 50 # quality factor of notch filter
        b, a = signal.iirnotch(noramlised_frequency, Q)
        notched_drain_current = signal.filtfilt(b, a, notched_drain_current)     
    drain_current = notched_drain_current
    
    # Bandpass
    low_cutoff = 1
    low_butterworth_order = 1
    noramlised_cutoff = low_cutoff / (sampling_frequency / 2) # Normalize the frequency
    b, a = signal.butter(low_butterworth_order, noramlised_cutoff, btype='highpass')
    drain_current = signal.filtfilt(b, a, drain_current)
    drain_current = drain_current[10000:90000]
    drain_voltage = round(drain_voltage.iloc[0], ndigits = 1)
    
    freq_dom, S_Id = signal.periodogram(drain_current, sampling_frequency)
    S_Vg = S_Id / zero_gate_gm**2
    #S_Id_gauss = gaussian_filter(S_Id, sigma=2)
    S_Vg_gauss = gaussian_filter(S_Vg, sigma=2)
    norm_S_Id = S_Id / current_for_normalisation**2
    
    norm_noise_at_10_hz = max([norm_S_Id[i] for i in range(len(norm_S_Id)) if np.abs(freq_dom[i] - 10) <= 2])
    print('Noise at 10 Hz: ', str(norm_noise_at_10_hz), ' $Hz^{-1}$')
    norm_S_Vg_at_10_hz = max([S_Vg[i] for i in range(len(S_Vg)) if np.abs(freq_dom[i] - 10) <= 2])
    print('Gate equivalent noise at 10 Hz: ', str(norm_S_Vg_at_10_hz), ' $Hz^{-1}$')
    
    total_noise = np.sqrt(spy.integrate.simps(S_Id, freq_dom))
    print('Total RMS Noise: ', total_noise, ' A')
    total_normalised_noise = np.sqrt(spy.integrate.simps(norm_S_Id, freq_dom))
    print('Total Normalised Current Noise: ', total_normalised_noise)
    
    total_noise_vg = np.sqrt(spy.integrate.simps(S_Vg, freq_dom))
    print('Total V_{G,RMS} Noise: ', total_noise_vg, ' V')
    total_noise_vg_1_100 = np.sqrt(spy.integrate.simps(S_Vg[2:200], freq_dom[2:200]))
    print('Total V_{G,RMS} from 1 Hz to 100 Hz: ', total_noise_vg_1_100, ' V')
    
    # 50 Hz identifier: if amplitude at 50 Hz > amplitude at 20 Hz
    amplitude_50 = max([norm_S_Id[i] for i in range(len(norm_S_Id)) if np.abs(freq_dom[i] - 50) <= 2])
    amplitude_20 = max([norm_S_Id[i] for i in range(len(norm_S_Id)) if np.abs(freq_dom[i] - 20) <= 2])
    identifier_50 = 0
    if amplitude_50 > amplitude_20:
        identifier_50 = 1
        print('50 Hz Noise Detected')
    
    upper_frequency_1 = 100
    fit_freq_dom = [i for i in freq_dom[1:] if i <= upper_frequency_1]
    fit_norm_S_Id = [norm_S_Id[i] for i in range(1,len(norm_S_Id[1:])) if freq_dom[i] <= upper_frequency_1]
    fit_freq_dom = fit_freq_dom[2:]
    fit_norm_S_Id = fit_norm_S_Id[2:]
    def fit_func(f, k, a):
        return k/(f**a)
    def fit_func_2(f, k, a):
        return (k - a*f)
    # p0, p0_cov = spo.curve_fit(fit_func, fit_freq_dom, fit_norm_S_Id)#, maxfev=int(100e3))
    p1, p1_cov = spo.curve_fit(fit_func_2, np.log(fit_freq_dom), np.log(fit_norm_S_Id))#, maxfev=int(100e3))
    # k_value = p0[0]
    # a_value = p0[1]
    k_value_1 = np.exp(p1[0])
    a_value_1 = p1[1]
    # best_fit_line = fit_func(np.array(fit_freq_dom), k_value, a_value)
    best_fit_line_1 = fit_func(np.array(fit_freq_dom), k_value_1, a_value_1)
    # print('k value of k/(f^a) : ', k_value)
    # print('a value of k/(f^a): ', a_value)
    print('k_1 value of k/(f^a) : ', k_value_1)
    print('a_1 value of k/(f^a) : ', a_value_1)
    
    # # Noise resolution frequencies
    # frequency_rms = [[0,0]]
    # frequency_steps = 100
    # frequencies = np.arange(0, freq_dom[-1], frequency_steps)
    # frequency_indices = [list(freq_dom).index(min(freq_dom, key=lambda x:abs(x-frequencies[i]))) for i in range(len(frequencies))]
    # frequency_indices = frequency_indices[1:]
    # for i in range(len(frequency_indices) - 1):
    #     integral = np.sqrt(spy.integrate.simps(S_Id[0:frequency_indices[i]], freq_dom[0:frequency_indices[i]]))
    #     frequency_rms.append([frequency_indices[i], integral])
    # frequency_index = -1
    # for i in range(len(frequency_rms)-1):
    #     rms_1 = frequency_rms[i][1]
    #     rms_2 = frequency_rms[i+1][1]
    #     noise_change = rms_2 - rms_1
    #     if noise_change < minimum_current_value:
    #         frequency_index = i+1
    #         break
    # upper_frequency = frequency_rms[frequency_index][0]
    # print('Upper frequency: ', upper_frequency, 'Hz')
    upper_frequency = 0
    

    current_stdev = np.sqrt(np.var(drain_current))
    # print('Current Standard Deviation: ', current_stdev, ' A')
    normalised_curr_stdev = np.abs(current_stdev / current_for_normalisation)
    print('Normalised Current Standard Deviation: ', normalised_curr_stdev)
    # Output noise data to csv
    # noise_array = np.transpose(np.vstack((S_Id, S_Vg, norm_S_Id)))
    # np.savetxt("SId__SVg__SId_Id2.csv", noise_array, delimiter=",")
    
    # plt.figure()
    # plt.xlabel('Time')
    # plt.ylabel('Bandpassed Drain Current')
    # plt.plot(time[10000:90000], drain_current)
    # plt.tight_layout()

    # plt.figure()
    # plt.title('Current Power Density \n (Gaussian Filter)')
    # plt.xlabel('Frequency')
    # plt.ylabel('Gate-referred Current Power')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.plot(freq_dom[1:], S_Vg_gauss[1:])
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig('IR_Curr_Pd.png')
    
    plt.figure(figsize=(5,4))
    # plt.title('Normalised Current Power Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'$S_{I_D}/I_D^2$ ($A^2$/Hz)')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(freq_dom[1:], gaussian_filter(norm_S_Id[1:], sigma=2) , 'k')
    plt.plot(fit_freq_dom, best_fit_line_1, 'g', label = 'Best fit:\n' + str(float('%.3g' % k_value_1)) + '/$f^{'+str(float('%.3g' % a_value_1))+'}$')
    plt.grid()
    plt.tight_layout()
    #plt.legend()
    #plt.savefig('Norm SId_Id2 with fit.png')
    
    return total_noise, total_normalised_noise, current_stdev, normalised_curr_stdev, drain_voltage, current_for_normalisation, minimum_current_value, upper_frequency, k_value_1, a_value_1, identifier_50, norm_noise_at_10_hz, norm_S_Vg_at_10_hz, total_noise_vg, total_noise_vg_1_100

end = '?raw=true'

resistance_url = 'https://github.com/JamesETyrrell/PhD_Analysis/blob/5bc804262327f40dfba515ec0e6d4c8b19a639e6/Data/OECT%20Characterisation%20Example/Resistance.csv' + end
resistance_data = pd.read_csv(resistance_url,index_col=0)
conductance_value = conductance(resistance_data)

output_url = 'https://github.com/JamesETyrrell/PhD_Analysis/blob/5bc804262327f40dfba515ec0e6d4c8b19a639e6/Data/OECT%20Characterisation%20Example/Output.csv' + end
output_values = pd.read_csv(output_url,index_col=0)
pinch_off_voltage, output_conductance = output(output_values)

transfer_url = 'https://github.com/JamesETyrrell/PhD_Analysis/blob/5bc804262327f40dfba515ec0e6d4c8b19a639e6/Data/OECT%20Characterisation%20Example/Transfer.csv' + end
transfer_values = pd.read_csv(transfer_url,index_col=0)
peak_transconductance, on_current, off_current, off_current_voltage, switch_ratio, transfer_array, peak_gm_vg = transfer(transfer_values)

response_url = 'https://github.com/JamesETyrrell/PhD_Analysis/blob/13cdbcf3c8a2892b522589b4aaac8d4349214a89/Data/OECT%20Characterisation%20Example/Response_time.csv' + end
response_values = pd.read_csv(response_url,index_col=0)
response_time_90, jump_ratio, response_time_1 = response_time(response_values)

noise_url = 'https://github.com/JamesETyrrell/PhD_Analysis/blob/a3d487e63167e58d0604acc5b857a63293739a27/Data/OECT%20Characterisation%20Example/IR%20noise.csv' + end
noise_values = pd.read_csv(noise_url,index_col=0)
total_noise, total_normalised_noise, current_stdev, normalised_curr_stdev, noise_drain_voltage, current_for_normalisation, noise_current_resolution, upper_frequency, k_value_1, a_value_1, identifier_50, norm_noise_at_10_hz, norm_S_Vg_at_10_hz, total_noise_vg, total_noise_vg_1_100 = noise(transfer_values, noise_values)

