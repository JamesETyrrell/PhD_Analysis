# -*- coding: utf-8 -*-
"""
Amplifier Simulation

James Tyrrell

18/04/2020

"""

##########################################
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.optimize as spo
from scipy.ndimage import gaussian_filter
import pandas as pd
import xlrd
import numpy.fft as fft_
import csv
import time
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as spy
from scipy import signal

plt.rcParams["font.family"] = 'serif'
plt.rcParams.update({'font.size': 14})
##########################################

#model_previous_data = 'no'
model_previous_data = 'yes'

#plot_characteristics = 'no'
plot_characteristics = 'yes'

simulate_temporal_response = 'no'
#simulate_temporal_response = 'yes'

simulate_trans_freq = 'no'
#simulate_trans_freq = 'yes'

simulate_cut_off_response = 'no'
#simulate_cut_off_response = 'yes'

simulate_current_response = 'no'
#simulate_current_response = 'yes'

simulate_differential_amplifier = 'no'
#simulate_differential_amplifier = 'yes'

simluate_common_source_amplifier = 'no'
#simluate_common_source_amplifier = 'yes'

# REQUIRES simluate_common_source_amplifier
common_source_amp_char = 'no'
#common_source_amp_char = 'yes'

##########################################

if simluate_common_source_amplifier == 'yes':
    if common_source_amp_char == 'yes':
        common_source_amp_char = 'yes'
else:
    common_source_amp_char = 'no'

#-----------------------------------------------------#
# Define the current response

def master_current(drain_voltage, gate_voltage, conductance, pinch_off, alpha, beta, lamda):
    # gamma is called lambda in paper
    gate_voltage = np.array(gate_voltage) 
    amplitude = -(1/3)*conductance
    gate_shift = alpha*gate_voltage + beta*pinch_off
    func_1 = (drain_voltage - gate_shift)*(np.arctan(lamda*(drain_voltage - gate_shift)) - np.pi/2)
    func_2 = gate_shift*((np.arctan(-1*lamda*gate_shift)) - np.pi/2)
    final = amplitude*(func_1 + func_2)
    return final


def temporal_response(curr_samp_freq, gate_drain_times, temp_drain_voltage, temp_gate_voltage, response_time_constant, conductance, pinch_off, alpha, beta, gamma):
    """
    Parameters
    ----------
    curr_samp_freq : float
        Sampling frequency of drain current, 1 = 1 Hz
    gate_drain_times : array
        Array containing the time values to make the temp_gate_voltage
    temp_drain_voltage : array
        Temporal Drain Voltage, same length as Temporal Gate Voltage, assumes constant sampling
    temp_gate_voltage : array
        Temporal Gate Voltage, same length as Temporal Drain Voltage, assumes constant sampling
    response_time : float
        Response time CONSTANT TAU of the OECT
    conductance : float
        Conductance of the OECT
    pinch_off : float
        Pinch-off voltage of the OECT
    alpha : float
        Alpha value of model
    beta : float
        Beta value of model
    gamma : float
        Gamma value of model

    Returns
    List of OECT response values

    """
    
    curr_times = []
    curr_response = []
    gate_drain_sampling_freq = 1/(gate_drain_times[1] - gate_drain_times[0])
    curr_sampling_time = 1/curr_samp_freq
    sampling_ratio = int(curr_samp_freq / gate_drain_sampling_freq)
    gate_extended = list(np.repeat(temp_gate_voltage, sampling_ratio))
    drain_extended = list(np.repeat(temp_drain_voltage, sampling_ratio))
    
    first_drain_current = master_current(temp_drain_voltage[0], temp_gate_voltage[0], conductance, pinch_off, alpha, beta, gamma)
    current_gate_voltage = temp_gate_voltage[0]
    current_drain_voltage = temp_drain_voltage[0]
    current_time_value = 0
    current_curr = first_drain_current
    
    count_1 = 0
    start_current = current_curr
    final_steady_state_response = current_curr
    curr_change = 0
    t = 0
    for i in range(len(gate_extended) - 1):        
        count_1 += 1
        curr_times.append(current_time_value + i*curr_sampling_time)
        change = 0
        gate_voltage = gate_extended[count_1]
        drain_voltage = drain_extended[count_1]
        # print('Gate ', gate_voltage)
        # print('Drain ', drain_voltage)
        # print('Current gate', current_gate_voltage)
        if gate_voltage != current_gate_voltage:
            current_gate_voltage = gate_voltage
            change = 1
        if drain_voltage != current_drain_voltage:
            current_drain_voltage = drain_voltage
            change = 1
            
        if change == 1:
            t = 0
            start_current = current_curr
            final_steady_state_response = master_current(drain_voltage, gate_voltage, conductance, pinch_off, alpha, beta, gamma)
            curr_change = start_current - final_steady_state_response
            #print(start_current)
        if change == 0:
            t += curr_sampling_time
            current_curr = final_steady_state_response + curr_change*np.exp(-t/response_time_constant)
                        
        curr_response.append(current_curr)
        #print('Temporal Response: Completed ', i+1, ' of ', len(gate_extended)-1)
    
    return curr_times, curr_response

def exp_list(start, stop, multiple):
    
    return_list = []
    return_list.append(start)
    while True:
        previous_value = return_list[-1]
        new_value = multiple*previous_value
        if previous_value < stop:
            return_list.append(new_value)
        else:
            break
    
    return return_list

def output_voltage_finder(gate_ground, amplifier_transfer_list):
    
    gate_voltages = [i[0] for i in amplifier_transfer_list]
    closest_index = min(range(len(gate_voltages)), key=lambda i: abs(gate_voltages[i]-gate_ground))
    output_voltage = amplifier_transfer_list[closest_index][1]
    
    return output_voltage

def step_function(time, step_time, step_start, step_height):
    
    step_values = []
    for i in time:
        if i <= step_time:
            value = step_start
        else:
            value = step_height
        step_values.append(value)
    
    return step_values

def square_wave_function(time, amplitude_low, amplitude_high, low_time, high_time):
    # start with low then high
    total_period = low_time + high_time
    cycle_number = int(np.floor(time[-1] / total_period))
    cycle_list = []
    time_list = [0]
    time_now = 0
    for i in range(cycle_number):
        cycle_list.append([time_now, amplitude_low])
        time_now = time_now + low_time
        time_list.append(time_now)
        cycle_list.append([time_now, amplitude_high])
        time_now = time_now + high_time
        time_list.append(time_now)
    cycle_list.append([time_now, amplitude_low])
    gate_values = []
    for i in time:
        closest_time_index = min(range(len(time_list)), key=lambda j: abs(time_list[j]-i))
        closest_time = time_list[closest_time_index]
        if i == closest_time:
            amplitude = cycle_list[closest_time_index][1]
        if i < closest_time:
            closest_time_index = closest_time_index - 1
            amplitude = cycle_list[closest_time_index][1]
        if i > closest_time:
            amplitude = cycle_list[closest_time_index][1]
        gate_values.append(amplitude)

    return gate_values

def output_voltage_finder_new(gate_ground, amplifier_transfer_list):
    
    gate_voltages = [i[0] for i in amplifier_transfer_list]
    drain_voltages = [i[1] for i in amplifier_transfer_list]
    
    if gate_ground > max(gate_voltages):
        output_voltage = min(drain_voltages)
    elif gate_ground < min(gate_voltages):
        output_voltage = max(drain_voltages)
    else:  
        closest_gate_index = min(range(len(gate_voltages)), key=lambda i: abs(gate_voltages[i]-gate_ground))
        indices = [closest_gate_index - 1, closest_gate_index + 1]
        diff_list = [abs(gate_voltages[closest_gate_index-1] - gate_ground), abs(gate_voltages[closest_gate_index+1] - gate_ground)]
        other_closest_diff = diff_list.index(min(diff_list))
        other_gate_index = indices[other_closest_diff]           
        output_voltage = ( ( (gate_ground - gate_voltages[closest_gate_index]) / (gate_voltages[other_gate_index] - gate_voltages[closest_gate_index]) ) * (drain_voltages[other_gate_index] - drain_voltages[closest_gate_index]) ) + drain_voltages[closest_gate_index]
    
    return output_voltage


def amplifier_response(amp_samp_freq, gate_drain_times, temp_gate_voltage, response_time_constant, gate_ground_drain_combination_list):
    
    amp_times = []
    amp_response = []
    gate_drain_sampling_freq = 1/(gate_drain_times[1] - gate_drain_times[0])
    amp_sampling_time = 1/amp_samp_freq
    sampling_ratio = int(amp_samp_freq / gate_drain_sampling_freq)
    gate_gr_extended = list(np.repeat(temp_gate_voltage, sampling_ratio))
    
    first_amp_voltage = output_voltage_finder_new(temp_gate_voltage[0], gate_ground_drain_combination_list)
    current_gate_voltage = temp_gate_voltage[0]
    current_time_value = 0
    current_amp_voltage = first_amp_voltage
    
    count_1 = 0
    start_amp_volt = current_amp_voltage
    final_steady_state_response = current_amp_voltage
    amp_change = 0
    t = 0
    for i in range(len(gate_gr_extended) - 1):        
        count_1 += 1
        amp_times.append(current_time_value + i*amp_sampling_time)
        change = 0
        gate_voltage = gate_gr_extended[count_1]
        if gate_voltage != current_gate_voltage:
            current_gate_voltage = gate_voltage
            change = 1
        if change == 1:
            t = 0
            start_amp_volt = current_amp_voltage
            final_steady_state_response = output_voltage_finder_new(gate_voltage, gate_ground_drain_combination_list)
            amp_change = start_amp_volt - final_steady_state_response
        if change == 0:
            t += amp_sampling_time
            current_amp_voltage = final_steady_state_response + amp_change*np.exp(-t/response_time_constant)
                        
        amp_response.append(current_amp_voltage)
    
    return amp_times, amp_response

#-----------------------------------------------------#

# Definitions for getting previous data parameter values

# Conductance Measurement
def conductance_(res_data):
    conductance = res_data['Conductance']
    conductance_mean = conductance.mean()
    print('Conductance: ', str(float('%.3g' % conductance_mean)), ' S (3 s.f.)')
    return conductance_mean

# Output Characteristics
def output_(conductance_value, output_data):
    
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
    
    suitable = 'yes'
    #suitable = 'no' 
    
    return pinch_off_voltage, output, first_sweep_values, suitable

def transfer_(transfer_data):
    
    gate_voltage = transfer_data['Gate Voltage']
    drain_current = transfer_data['Drain Current']
    gate_current = transfer_data['Gate Current']
    transfer = np.zeros(shape = [len(gate_voltage) , 4])
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
        
    # Determine parameters from transfer curve
    transfer_indices = [i for i in range(len(gate_voltage)) if i != first_sweep_number+1]
    transfer = transfer[transfer_indices,:]
    #transfer[1:,2] = gaussian_filter(transfer[1:,2], sigma=3)
    high_cutoff = 5
    butterworth_order = 3
    sampling_frequency = 100
    normalised_cutoff = high_cutoff / (sampling_frequency / 2) # Normalize the frequency
    d, c = signal.butter(butterworth_order, normalised_cutoff, btype='lowpass')
    transfer[0:first_sweep_number,2] = signal.filtfilt(d, c, transfer[0:first_sweep_number,2])
    transfer_sweep = transfer[first_sweep_number:,2]
    transfer_sweep_reverse = transfer_sweep[::-1]
    filtered_transfer_reverse = signal.filtfilt(d, c, transfer_sweep_reverse)
    filtered_transfer = filtered_transfer_reverse[::-1]
    transfer[first_sweep_number:,2] = filtered_transfer
    gate_voltage = transfer[: , 0]
    current = transfer[: , 1]
    transconductance = transfer[: , 2]
    
    smooth_gate_1 = gate_voltage[0:first_sweep_number]
    smooth_current_1 = current[0:first_sweep_number]
    smooth_trans_1 = transconductance[0:first_sweep_number]
    
    return smooth_gate_1, smooth_current_1, smooth_trans_1

def response_time_(response_time_data):

    response_time_array = np.zeros(shape=[len(response_time_data['Gate Voltage']),5])
    response_time_array[:,0] = np.array(response_time_data['Gate Voltage'])
    response_time_array[:,1] = np.array(response_time_data['Gate Time'])
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
        if diff > 0.1:
            gate_apply_index = int(i)
            break
    
    pre_drain_current = response_time_array[gate_apply_index-20,3]
    current_jump = min(response_time_array[gate_apply_index-100:gate_apply_index+100,3])
    exponential_fitting_index = list(response_time_array[0:gate_apply_index+100,3]).index(current_jump)
    if exponential_fitting_index < gate_apply_index:
        exponential_fitting_index = gate_apply_index + 1
    number_of_indices_between_gate_apply_and_current_jump = exponential_fitting_index - gate_apply_index
    
    final_current = response_time_array[-1,3]
    threshold_current = pre_drain_current + 0.9*(final_current - pre_drain_current)
    repsond_index = gate_apply_index
    for i in range(gate_apply_index, len(response_time_array[:,3])):
        current = response_time_array[i,3]
        if response_time_array[i,3] > threshold_current:
            respond_index = i
            break
    response_time_1 = (respond_index - gate_apply_index)*sampling_step
    
    if response_time_1 > 0.01:
        fitting_range = 500
    elif response_time_1 > 0.005:
        fitting_range = 250
    else:
        fitting_range = 50
        
    # Simple exponential fitting
    times = response_time_data[exponential_fitting_index:exponential_fitting_index+fitting_range,1]
    drain_values_to_fit = response_time_data[exponential_fitting_index:exponential_fitting_index+fitting_range,3]      
    def fit_func(times, amp, t_const, t1, c):
        exp_current = c - amp*(1-np.exp(-1*(times-t1)/t_const))
        return exp_current
    bounds_ = ((-np.inf, -np.inf, 0.099, -np.inf), (np.inf, np.inf, 0.101, np.inf))
    p0, p0_cov = spo.curve_fit(fit_func, times, drain_values_to_fit, bounds = bounds_, maxfev=100000)
    amplitude = p0[0]
    time_constant = p0[1]
    response_time = np.log(10)*time_constant
    response_time = response_time + number_of_indices_between_gate_apply_and_current_jump*sampling_step
    
    return response_time, time_constant

#-----------------------------------------------------#

# Fit the OECT characteristics and compare output & transfer

if model_previous_data == 'yes':
    
    end = '?raw=true'

    resistance_url = 'https://github.com/JamesETyrrell/PhD_Analysis/blob/5bc804262327f40dfba515ec0e6d4c8b19a639e6/Data/OECT%20Characterisation%20Example/Resistance.csv' + end
    resistance_data = pd.read_csv(resistance_url,index_col=0)
    conductance_value = conductance_(resistance_data)
    output_url = 'https://github.com/JamesETyrrell/PhD_Analysis/blob/5bc804262327f40dfba515ec0e6d4c8b19a639e6/Data/OECT%20Characterisation%20Example/Output.csv' + end
    output_values = pd.read_csv(output_url,index_col=0)
    pinch_off_value, output_array, first_sweep, suitable = output_(conductance_value, output_values)
    transfer_url = 'https://github.com/JamesETyrrell/PhD_Analysis/blob/5bc804262327f40dfba515ec0e6d4c8b19a639e6/Data/OECT%20Characterisation%20Example/Transfer.csv' + end
    transfer_values = pd.read_csv(transfer_url,index_col=0)
    transfer_gate, transfer_current, transfer_gm = transfer_(transfer_values)
    response_url = 'https://github.com/JamesETyrrell/PhD_Analysis/blob/13cdbcf3c8a2892b522589b4aaac8d4349214a89/Data/OECT%20Characterisation%20Example/Response_time.csv' + end
    response_values = pd.read_csv(response_url,index_col=0)
    oect_response_time_value, oect_response_time_constant = response_time_(response_values) # time constant vs response time
    
    number_of_gates = int(len(output_array[0,:])/2)
    number_of_samples = int(len(output_array[1:,0]))
    number_of_samples = int(len(output_array[125:377,0]))
    new_output = np.zeros(shape = [3, number_of_gates*number_of_samples])
    for i in range(number_of_gates):
        gate_value = output_array[0,2*i]
        drain_voltage_values = output_array[1:,2*i]
        drain_current_values = output_array[1:,2*i+1]
        drain_voltage_values = output_array[125:377,2*i]
        drain_current_values = output_array[125:377,2*i+1]
        new_output[0,number_of_samples*i:number_of_samples*(i+1)] = gate_value
        new_output[1,number_of_samples*i:number_of_samples*(i+1)] = drain_voltage_values
        new_output[2,number_of_samples*i:number_of_samples*(i+1)] = drain_current_values
    
    def fit_func(drain_values, alpha, beta, gamma, G):
        pinch_off = pinch_off_value
        conductance = G
        gate_voltage = new_output[0,:]
        drain_voltage = drain_values
        
        gate_voltage = np.array(gate_voltage) 
        amplitude = -(1/3)*conductance
        gate_shift = alpha*gate_voltage + beta*pinch_off
        func_1 = (drain_voltage - gate_shift)*(np.arctan(gamma*(drain_voltage - gate_shift)) - np.pi/2)
        func_2 = gate_shift*((np.arctan(-gamma*gate_shift)) - np.pi/2)
        final = amplitude*(func_1 + func_2)
        
        return final          
    
    #bounds_ = ( (-100, -100, -100, -100), (100, 100, 100, 100) )
    p0, p0_cov = spo.curve_fit(fit_func, new_output[1,:], new_output[2,:])#, bounds = bounds_, maxfev=100000)
    alpha_fit = p0[0]
    beta_fit =  p0[1]
    gamma_fit = p0[2]
    conductance_value = p0[3]
    #pinch_off_value = p0[4]
    
    print('alpha =', alpha_fit, ', beta =', beta_fit, ', gamma =', gamma_fit)
    print('Conductance =', conductance_value, ', Pinch off =', pinch_off_value, ', OECT response time constant =', oect_response_time_constant)
    
    if plot_characteristics == 'yes':
        
        plt.figure(figsize = (5.5,4.5))
        plt.title('Fitted Output Characteristics\n$\\alpha$ = ' + str(round(alpha_fit,2)) + ', $\\beta$ = ' + str(round(beta_fit,2)) + ', $\\lambda$ = ' + str(round(gamma_fit,2)) + ' V$^{-1}$' )
        plt.xlabel('Drain-Source Voltage (V)')
        plt.ylabel('Drain Current (mA)')
        first_sweep = int(len(output_array[:,0])/2)
        number_of_gates = int(len(output_array[0,:])/2)
        for i in range(number_of_gates):
            this_gate = output_array[0,2*i]
            if np.round(this_gate, decimals=1) in [-0.2, 0, 0.2, 0.4, 0.6]:
                drain_voltage = output_array[1:first_sweep,2*i]
                drain_current = output_array[1:first_sweep,2*i+1]
                simulation_result = master_current(drain_voltage, this_gate, conductance_value, pinch_off_value, alpha_fit, beta_fit, gamma_fit)
                if np.abs(this_gate) < 0.001:
                    this_gate = 0
                plt.plot(drain_voltage, 1000*np.array(simulation_result), linewidth=2, label = str(this_gate) + 'V Fit')
                plt.plot(drain_voltage, 1000*np.array(drain_current), linewidth=2, label = str(this_gate) + 'V')
        plt.legend(title = 'Gate-Source Voltage (V)', ncol = 2, fontsize = 10, title_fontsize = 11)
        #plt.xlim([min(drain_voltage_values),0])
        #plt.ylim([1.1*1000*min(new_output[2,:]),0])
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.grid()
        
        measured_transfer_drain_voltage = -0.6
        simulation_result = master_current(measured_transfer_drain_voltage, transfer_gate, conductance_value, pinch_off_value, alpha_fit, beta_fit, gamma_fit)
        transconductance = [(simulation_result[i+1] - simulation_result[i])/(transfer_gate[i+1] - transfer_gate[i]) for i in range(len(simulation_result) - 1)]
        transconductance.append(transconductance[-1])
        fig, ax1 = plt.subplots(figsize=(5.5, 4.5))
        plt.title('Simulated Transfer\nCharacteristics, $V_{DS}$ = -0.6 V')
        color = 'tab:blue'
        #ax1.set_yscale('log')
        ax1.set_xlabel('Gate-Source Voltage (V)')
        ax1.set_ylabel('Drain Current (mA)', color=color) 
        ax1.plot(transfer_gate[20:], 1000*np.array(transfer_current[20:]), color=color, linestyle = 'solid')
        ax1.plot(transfer_gate[20:], 1000*np.array(simulation_result[20:]), color=color, linestyle = 'dashdot')
        ax1.plot([],[], color = 'k', linestyle = 'solid', label = 'Measured')
        ax1.plot([],[], color = 'k', linestyle = 'dashdot', label = 'Simulated')
        ax1.set_ylim(ax1.get_ylim()[::-1])
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('Transconductance (mS)', color=color)  # we already handled the x-label with ax1
        ax2.plot(transfer_gate[20:], 1000*np.array(transfer_gm[20:]), color=color, linestyle = 'solid')
        ax2.plot(transfer_gate[20:], 1000*np.array(transconductance[20:]), color=color, linestyle = 'dashdot')
        ax2.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc = 'lower center', fontsize = 11)
        #ax1.set_yscale('symlog')
    
#-----------------------------------------------------#

# OECT Characteristics for arbitrary parameter inputs

drain_voltage = -0.6

if model_previous_data == 'no':
    alpha = 1.04
    beta = -0.55
    gamma = -3.5
    conductance = 0.1214
    pinch_off = 0.56
    oect_response_time = 10e-3
    oect_response_time_constant = oect_response_time/np.log(10) # time constant vs response time
    # Cut-off frequency using response time = 100 us is about 3.6 kHz
    # Cut-off frequency using response time constant = 100 us is 1.6 kHz
    
    if plot_characteristics == 'yes':
        
        # Output
        plt.figure(figsize = (6.2,5))
        plt.title('Simulated Output Characteristics\n$\\alpha$ = ' + str(alpha) + ', $\\beta$ = ' + str(beta) + ', $\\lambda$ = ' + str(gamma) + ' V$^{-1}$')
        plt.xlabel('Drain-Source Voltage (V)')
        plt.ylabel('Drain Current (mA)')
        drain_voltage_output = np.arange(-1, 0.5, 0.001)
        gate_voltage_output = np.arange(-0.2, 0.65, 0.1)
        for i in range(len(gate_voltage_output)):
            this_gate = np.round(gate_voltage_output[i], decimals=2)
            simulation_result = master_current(drain_voltage_output, gate_voltage_output[i], conductance, pinch_off, alpha, beta, gamma)
            plt.plot(drain_voltage_output, 1000*np.array(simulation_result), linewidth=2, label = str(this_gate) + 'V')
        plt.legend(fontsize=11, title_fontsize=12, title = 'Gate-Source Voltage (V)', ncol=3 )
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.grid()
        
        
        # Transfer
        gate_voltage_transfer = np.arange(-1,1,0.001)
        simulation_result = master_current(drain_voltage, gate_voltage_transfer, conductance, pinch_off, alpha, beta, gamma)
        transconductance = [(simulation_result[i+1] - simulation_result[i])/(gate_voltage_transfer[i+1] - gate_voltage_transfer[i]) for i in range(len(simulation_result) - 1)]
        transconductance.append(transconductance[-1])
        fig, ax1 = plt.subplots(figsize = (5.5,4.5))
        plt.title('Simulated Transfer Characteristics')#'\n$V_{DS}$ = ' + str(drain_voltage) + ' V, $\\alpha$ = ' + str(alpha) + ', $\\beta$ = ' + str(beta) + ', $\\lambda$ = ' + str(gamma) + ' V$^{-1}$')
        color = 'tab:blue'
        ax1.set_xlabel('Gate-Source Voltage (V)')
        ax1.set_ylabel('Drain Current (mA)', color=color) 
        ax1.plot(gate_voltage_transfer, 1000*np.array(simulation_result), linewidth=2.5, color=color, linestyle = 'solid')
        ax1.set_ylim(ax1.get_ylim()[::-1])
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('Transconductance (mS)', color=color)  # we already handled the x-label with ax1
        ax2.plot(gate_voltage_transfer, 1000*np.array(transconductance), linewidth=2.5, color=color, linestyle = 'solid')
        ax2.tick_params(axis='y', labelcolor=color)
        plt.tight_layout()
        
        # Transfer - shows capability
        gate_voltage_transfer = np.arange(-1,1,0.001)
        simulation_result = master_current(drain_voltage, gate_voltage_transfer, conductance, pinch_off, alpha, beta, gamma)
        transconductance = [(simulation_result[i+1] - simulation_result[i])/(gate_voltage_transfer[i+1] - gate_voltage_transfer[i]) for i in range(len(simulation_result) - 1)]
        transconductance.append(transconductance[-1])
        fig, ax1 = plt.subplots(figsize = (5.5,4.5))
        plt.title('Simulated Transfer Characteristics')#'\n$V_{DS}$ = ' + str(drain_voltage) + ' V, $\\alpha$ = ' + str(alpha) + ', $\\beta$ = ' + str(beta) + ', $\\lambda$ = ' + str(gamma) + ' V$^{-1}$')
        color = 'tab:blue'
        ax1.set_xlabel('Gate-Source Voltage (V)')
        ax1.set_ylabel('Negative Drain Current Square Root ($\sqrt{A}$)', color=color)  # we already handled the x-label with ax1
        ax1.plot(gate_voltage_transfer, np.sqrt(np.array(-1*simulation_result)), linewidth=2.5, color=color, linestyle = 'solid')
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_yscale('log')
        ax2.set_ylabel('Negative Drain Current (mA)', color=color) 
        ax2.plot(gate_voltage_transfer, -1000*np.array(simulation_result), linewidth=2.5, color=color, linestyle = 'solid')
        ax2.tick_params(axis='y', labelcolor=color)
        plt.tight_layout()

else:
    alpha = alpha_fit
    beta = beta_fit
    gamma = gamma_fit
    conductance = conductance_value
    pinch_off = pinch_off_value
    oect_response_time_constant = oect_response_time_constant

#-----------------------------------------------------#

if simulate_current_response == 'yes':

    # Device Response
    
    t = np.arange(0,10,0.001)
    gate_voltage = 0.2*np.sin(2*np.pi*t)
    
    current_response = master_current(drain_voltage, gate_voltage, conductance, pinch_off, alpha, beta, gamma)
    
    fig, ax1 = plt.subplots(figsize = (9,7))
    plt.title('Device Response, $V_{DS}$ = ' + str(drain_voltage) + 'V')
    color = 'tab:blue'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Gate-Source Voltage Input (mV)', color=color) 
    ax1.plot(t, 1000*np.array(gate_voltage), color=color, linestyle = 'solid')
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Drain Current Output (mA)', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, 1000*np.array(current_response), color=color, linestyle = 'solid')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.gca().invert_yaxis()

#-----------------------------------------------------#

if simulate_temporal_response == 'yes':
    
    # simulate temporal response
    input_sampling_rate = 1e5
    duration = 0.2
    time_values = np.arange(0, duration, 1/input_sampling_rate)
    input_frequency = 500
    response_sampling_rate = 1e5
    
    #gate_voltage = 0.01*np.sin(input_frequency*2*np.pi*time_values) + 0.2
    #gate_voltage = step_function(time_values, 0.1, 0, 0.1)
    #gate_voltage = square_wave_function(time_values, 0, 0.6, 1e-2, 1.5e-2)
    gate_voltage = [0 for i in range(len(time_values))]
    
    drain_voltage = [drain_voltage for i in range(len(gate_voltage))]
    #drain_voltage = square_wave_function(time_values, 0, 2*drain_voltage, 0.05e-3, 0.05e-3)
    
    response_time_values, response_current = temporal_response(response_sampling_rate, time_values, drain_voltage, gate_voltage, oect_response_time_constant, conductance, pinch_off, alpha, beta, gamma)
    print('Temporal response completed')
    #print(np.mean(response_current[-1000:])*(-1000))
    # print(np.mean(response_current[-1000:])*(-1000/3.66))
    
    # plt.figure()
    # plt.title('OECT Response')
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Drain Current (mA)')
    # plt.gca().invert_yaxis()
    # plt.plot(1e3*(np.array(response_time_values[:int(-(len(response_current))/10)])), 1000*np.array(response_current[:int(-(len(response_current))/10)]), 'k-')
    # plt.tight_layout()
    
    fig, ax1 = plt.subplots(figsize=(6.2,5))
    plt.title('Response Time')
    color = 'tab:blue'
    ax1.set_xlabel('Time ($\mu$s)')
    ax1.set_ylabel('Gate-Source Voltage Input (mV)', color=color) 
    ax1.plot(1000000*(np.array(time_values)), 1000*np.array(gate_voltage), color=color, linestyle = 'solid')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Drain Current Response (mA)', color=color)  # we already handled the x-label with ax1
    ax2.plot(1000000*(np.array(response_time_values)), 1000*np.array(response_current), color=color, linestyle = 'solid')
    ax2.tick_params(axis='y', labelcolor=color)
    #ax2.set_xlim([-100, 500])
    fig.tight_layout()
    #ax2.set_ylim(ax2.get_ylim()[::-1])

#-----------------------------------------------------#

if simulate_trans_freq == 'yes':
    
    # parameters for transconductnace-frequency
    gate_amplitude = 0.01
    start = 1e-2
    stop = 1e6
    multiple = 1.5
    frequencies_to_test = exp_list(start, stop, multiple)
    print('Number of frequencies: ', len(frequencies_to_test))
    
    input_sampling_rate = [100*i for i in frequencies_to_test]
    response_sampling_rate = [10*i for i in input_sampling_rate]
    time_values_for_frequencies = [np.arange(0, 1.5/frequencies_to_test[i], 1/input_sampling_rate[i]) for i in range(len(frequencies_to_test))]
    gate_voltages = [gate_amplitude*np.sin(frequencies_to_test[i]*2*np.pi*time_values_for_frequencies[i]) for i in range(len(frequencies_to_test))]
    drain_voltage = -0.6
    drain_voltages = [[drain_voltage for j in range(len(gate_voltages[i]))] for i in range(len(gate_voltages))]
    response_values = []
    transconductance_values = []
    for i in range(len(frequencies_to_test)):
        response_time_values, response_current_values = temporal_response(response_sampling_rate[i], time_values_for_frequencies[i], drain_voltages[i], gate_voltages[i], oect_response_time_constant, conductance, pinch_off, alpha, beta, gamma)
        response_values.append([response_time_values, response_current_values])
        # plt.figure()
        # plt.title(str(frequencies_to_test[i]))
        # plt.plot(response_time_values, response_current_values)
        current_p2p = max(response_current_values) - min(response_current_values)
        transcond = current_p2p/(2*gate_amplitude)
        transconductance_values.append(transcond)
        print('Frequency completed: ', frequencies_to_test[i])
    #transconductance_fractions = [i/max(transconductance_values) for i in transconductance_values]
    plt.figure(figsize=(6.2,5))
    plt.xscale('log')
    plt.plot(frequencies_to_test, 1000*np.array(transconductance_values), 'k')
    plt.grid()
    plt.title('Simulated Transconductance-Frequency\n$V_{DS}$ = -0.6 V, $\\tau$ = 43 $\\mu$s')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Transconductance (mS)')
    cut_off_trans = max(transconductance_values) / np.sqrt(2)
    #cut_off_trans_fraction = 1 / np.sqrt(2)
    cut_off_index = 0
    for i in range(len(transconductance_values)):
        if transconductance_values[i] < cut_off_trans:
            cut_off_index = i
            break
    frequency_1 = frequencies_to_test[cut_off_index-1]
    frequency_2 = frequencies_to_test[cut_off_index]
    trans_1 = transconductance_values[cut_off_index-1]
    trans_2 = transconductance_values[cut_off_index]
    gradient = (trans_2 - trans_1) / (frequency_2 - frequency_1)
    cut_off_freq = (cut_off_trans - trans_1) / gradient + frequency_1
    print('The cut-off transconductance was ', 1000*cut_off_trans, 'mS')
    print('The cut-off frequency was ', cut_off_freq, 'Hz')
    plt.ylim([-0.5,1.05*1000*max(transconductance_values)])
    plt.plot([0,cut_off_freq],[1000*cut_off_trans, 1000*cut_off_trans], 'r', linestyle = 'dashed')
    plt.plot([cut_off_freq,cut_off_freq],[-1, 1000*cut_off_trans], 'r', linestyle = 'dashed', label = 'Cut-off frequency:\n' + str(int(cut_off_freq)) + 'Hz')
    plt.legend()
    
#-----------------------------------------------------#

if simulate_cut_off_response == 'yes':
    
    oect_response_times = exp_list(10e-6, 1e-3, 2)
    cut_off_frequencies = []
    
    for responses in oect_response_times:
        gate_amplitude = 0.01
        start = 1e-3
        stop = 1e6
        multiple = 2
        frequencies_to_test = exp_list(start, stop, multiple)
        #print('Number of frequencies: ', len(frequencies_to_test))    
        input_sampling_rate = [100*i for i in frequencies_to_test]
        response_sampling_rate = [10*i for i in input_sampling_rate]
        time_values_for_frequencies = [np.arange(0, 1.5/frequencies_to_test[i], 1/input_sampling_rate[i]) for i in range(len(frequencies_to_test))]
        gate_voltages = [gate_amplitude*np.sin(frequencies_to_test[i]*2*np.pi*time_values_for_frequencies[i]) + 0.2 for i in range(len(frequencies_to_test))]
        drain_voltage = -0.6
        drain_voltages = [[drain_voltage for j in range(len(gate_voltages[i]))] for i in range(len(gate_voltages))]
        response_values = []
        transconductance_values = []
        for i in range(len(frequencies_to_test)):
            response_time_values, response_current_values = temporal_response(response_sampling_rate[i], time_values_for_frequencies[i], drain_voltages[i], gate_voltages[i], responses, conductance, pinch_off, alpha, beta, gamma)
            response_values.append([response_time_values, response_current_values])
            # plt.figure()
            # plt.title(str(frequencies_to_test[i]))
            # plt.plot(response_time_values, response_current_values)
            current_p2p = max(response_current_values) - min(response_current_values)
            transcond = current_p2p/(2*gate_amplitude)
            transconductance_values.append(transcond)
            print('Frequency completed: ', frequencies_to_test[i])
        cut_off_trans = max(transconductance_values) / np.sqrt(2)
        cut_off_index = 0
        for i in range(len(transconductance_values)):
            if transconductance_values[i] < cut_off_trans:
                cut_off_index = i
                break
        frequency_1 = frequencies_to_test[cut_off_index-1]
        frequency_2 = frequencies_to_test[cut_off_index]
        trans_1 = transconductance_values[cut_off_index-1]
        trans_2 = transconductance_values[cut_off_index]
        gradient = (trans_2 - trans_1) / (frequency_2 - frequency_1)
        cut_off_freq = (cut_off_trans - trans_1) / gradient + frequency_1
        cut_off_frequencies.append(cut_off_freq)
        print('Response time completed:', responses)
    
    def fit_func(res_time, m, c):
        return m*res_time + c
    p0, p0_cov = spo.curve_fit(fit_func, np.log(oect_response_times), np.log(cut_off_frequencies)) 
    gradient, intercept = p0[0], p0[1]
    print('Response time/cut-off freq. relationship is log linear.\nGradient = ', gradient, ', intercept = ', intercept)

    
    plt.figure()
    plt.title('Response Time - Cut-off Frequency Relationship')
    plt.xlabel('Response Time (s)')
    plt.ylabel('Cut-off Frequency (Hz)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(b=True, which = 'major', linestyle = '-', color='darkgray')
    plt.grid(b=True, which = 'minor', linestyle = '--', color = 'lightgray')
    plt.plot(oect_response_times, cut_off_frequencies, 'k')
    
#-----------------------------------------------------#

if simulate_differential_amplifier == 'yes':

    # Differential Amplifier Simulation
    
    supply_voltage = -5.0
    res_1 = 15/conductance_value
    res_1 = 565
    res_1 = np.round(res_1, decimals=1)
    res_2 = 15/conductance_value
    res_2 = 565
    res_2 = np.round(res_2, decimals=1)
    
    
    # Determination of linear amplifier load-line output intersection 
    drain_resolution = 0.0001
    drain_sweep = np.arange(supply_voltage, 0, drain_resolution)
    gate_sweep = np.arange(-0.5, 1.5, drain_resolution/1000)
    gate_voltage_values = np.zeros(len(drain_sweep))
    # for j in range(len(drain_sweep)):
    #     equation_to_solve = np.array((supply_voltage/res_1) - (drain_sweep[j]/res_1) - master_current(drain_sweep[j], gate_sweep, conductance, pinch_off, alpha, beta, gamma))
    #     sorted_array = np.sort(equation_to_solve)
    #     if sorted_array[0] < 0:
    #         if sorted_array[-1] > 0:
    #             equation_to_solve = list(equation_to_solve)
    #             smallest_index = min(range(len(equation_to_solve)), key=lambda i: abs(equation_to_solve[i]-0))
    #             gate_voltage_values[j] = gate_sweep[smallest_index]
    #     print(j+1, '/', str(len(drain_sweep)))
    for j in range(len(drain_sweep)):
        equation_to_solve = np.array((supply_voltage/res_1) - (drain_sweep[j]/res_1) - master_current(drain_sweep[j], gate_sweep, conductance, pinch_off, alpha, beta, gamma))
        sorted_array = equation_to_solve
        if sorted_array[0]*sorted_array[-1] < 0: # if a solution exists
            # linearly interpolate
            equation_to_solve = list(equation_to_solve)
            smallest_index = min(range(len(equation_to_solve)), key=lambda i: abs(equation_to_solve[i]-0))
            gate_1 = gate_sweep[smallest_index - 1]
            gate_2 = gate_sweep[smallest_index]
            if smallest_index == len(gate_sweep) - 1:
                gate_3 = gate_2
            else:
                gate_3 = gate_sweep[smallest_index + 1]
            drain_1 = np.array((supply_voltage/res_1) - (drain_sweep[j]/res_1) - master_current(drain_sweep[j], gate_1, conductance, pinch_off, alpha, beta, gamma))
            drain_2 = np.array((supply_voltage/res_1) - (drain_sweep[j]/res_1) - master_current(drain_sweep[j], gate_2, conductance, pinch_off, alpha, beta, gamma))
            drain_3 = np.array((supply_voltage/res_1) - (drain_sweep[j]/res_1) - master_current(drain_sweep[j], gate_3, conductance, pinch_off, alpha, beta, gamma))
            if np.sign(drain_1*drain_2) == -1:
                #interpolate: (gate_1, drain_1) and (gate_2, drain_2) - find gate for drain = 0
                zero_drain_gate = gate_1 - (gate_2 - gate_1)*( drain_1 / (drain_2 - drain_1) )
            elif np.sign(drain_2*drain_3) == -1:
                #interpolate: (gate_2, drain_2) and (gate_3, drain_3) - find gate for drain = 0
                zero_drain_gate = gate_3 - (gate_2 - gate_3)*( drain_3 / (drain_2 - drain_3) )
            else:
                print('Transfer: Interpolation not found')
                zero_drain_gate = gate_2
            gate_voltage_values[j] = zero_drain_gate
        if (j+1) % 100 == 0:
            print(j+1, '/', str(len(drain_sweep)))
    # Gives all allowed initial conditions
    
    gate_drain_combinations = [[gate_voltage_values[i], drain_sweep[i]] for i in range(len(drain_sweep)) if gate_voltage_values[i] != 0]
    drain_voltage_success = [i[1] for i in gate_drain_combinations]
    gate_success = [i[0] for i in gate_drain_combinations]
    gain = [-(drain_voltage_success[i+1] - drain_voltage_success[i])/(gate_success[i+1] - gate_success[i]) for i in range(len(gate_success) - 1)]
    gain.append(gain[-1])
    hydrolysis_combinations = []
    for i in gate_drain_combinations:
        if np.abs(i[0] - i[1]) < 1.2:
            hydrolysis_combinations.append(i)
    hydrolysis_drain_success = [i[1] for i in hydrolysis_combinations]
    hydrolysis_gate_success = [i[0] for i in hydrolysis_combinations]
    hydrolysis_gain = [-(hydrolysis_drain_success[i+1] - hydrolysis_drain_success[i])/(hydrolysis_gate_success[i+1] - hydrolysis_gate_success[i]) for i in range(len(hydrolysis_combinations) - 1)] 
    hydrolysis_gain.append(hydrolysis_gain[-1])
    
    beyond_pinch_index = min(range(len(hydrolysis_combinations)), key=lambda i: abs((hydrolysis_combinations[i][0] - hydrolysis_combinations[i][1]) - pinch_off))
    highest_hydrolysis_gate_index = hydrolysis_gate_success.index(max(hydrolysis_gate_success))
    gate_source_range = max(hydrolysis_gate_success) - hydrolysis_gate_success[beyond_pinch_index]
    average_hydrolysis_gain = np.mean(hydrolysis_gain[highest_hydrolysis_gate_index:beyond_pinch_index])
    print('Gate source range from ', str(np.round(hydrolysis_gate_success[beyond_pinch_index],decimals=4)), 'V to ', 
                                         str(np.round(max(hydrolysis_gate_success), decimals = 4)), 'V with a range of ',
                                             str(np.round(gate_source_range, decimals = 4)), 'V , where the average gain was',
                                             str(np.round(average_hydrolysis_gain, decimals = 4)) )
    
    gate_drain_combinations_to_plot = [i for i in gate_drain_combinations if i[1] not in hydrolysis_drain_success]
    drain_voltage_success_to_plot = [i[1] for i in gate_drain_combinations_to_plot]
    gate_voltage_success_to_plot = [i[0] for i in gate_drain_combinations_to_plot]
    gain_to_plot = [-(drain_voltage_success_to_plot[i+1] - drain_voltage_success_to_plot[i])/(gate_voltage_success_to_plot[i+1] - gate_voltage_success_to_plot[i]) for i in range(len(gate_voltage_success_to_plot) - 1)]
    gain_to_plot.append(gain_to_plot[-1])
    
    fig, ax1 = plt.subplots(figsize = (9,7))
    plt.title('Gate-Drain Solutions to Linear Amplifier, $V_{out}$ =' + str(supply_voltage) + 'V , $R_{load}$ =' + str(res_1) + '$\Omega$')
    color = 'tab:blue'
    ax1.set_xlabel('Gate-Source Voltage (Input) (V)')
    ax1.set_ylabel('Drain-Source Voltage (Output) (V)', color=color) 
    ax1.plot(np.array(gate_voltage_success_to_plot), np.array(drain_voltage_success_to_plot), color=color, linestyle = 'solid')
    ax1.plot(np.array(hydrolysis_gate_success), np.array(hydrolysis_drain_success), color=color, linestyle = 'dotted')
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.plot([],[], color = 'k', linestyle = 'solid', label = 'Hydrolysis not satisfied')
    ax1.plot([],[], color = 'k', linestyle = 'dotted', label = 'Hydrolysis satisfied')
    #plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Voltage Gain', color=color)  # we already handled the x-label with ax1
    ax2.plot(np.array(gate_voltage_success_to_plot), np.array(gain_to_plot), color=color, linestyle = 'solid')
    ax2.plot(np.array(hydrolysis_gate_success), np.array(hydrolysis_gain), color=color, linestyle = 'dotted')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.gca().invert_xaxis()
    ax1.legend()
    
    # plot the output characteristics with load line
    
    if plot_characteristics == 'yes':
        
        plt.figure(figsize = (9,7))
        plt.title('Output Characteristics')#', k = ' + str(k) + ', a = '+ str(a) + ', b = '+ str(b))
        plt.xlabel('Drain-Source Voltage (V)')
        plt.ylabel('Drain Current (mA)')
        load_line = [((supply_voltage/res_1) - (i/res_1)) for i in drain_voltage_success]
        plt.plot(drain_voltage_success, 1000*np.array(load_line), label = 'Load-line')
        drain_voltage = np.arange(drain_sweep[0],0.6,0.001)
        gate_voltage = np.arange(-1,1.5,0.05)
        for i in range(len(gate_voltage)):
            this_gate = np.round(gate_voltage[i], decimals = 2)
            simulation_result = master_current(drain_voltage, this_gate, conductance_value, pinch_off_value, alpha_fit, beta_fit, gamma_fit)
            plt.plot(drain_voltage, 1000*np.array(simulation_result), linewidth=1, label = str(this_gate) + 'V Simulation')
        #plt.legend(loc = 'upper left', title = 'Gate-Source \n Voltage (V)', ncol = 2)
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.grid()
    
    # Differential Amplifier Response = Difference between 2 linear amplifier responses
    # Assumes the transistors have equal characteristics
    
    t = np.arange(0,10,0.01)
    gate_1 = 0.02*np.cos(2*np.pi*t) + 0
    gate_2 = 0.02*np.cos(2*np.pi*(t-0.5)) + 0
    response_1 = []
    response_2 = []
    for j in range(len(t)):
        gate_1_value = gate_1[j]
        nearest_gate_index = min(range(len(gate_success)), key=lambda i: abs(gate_success[i]-gate_1_value))
        drain_source_value = drain_voltage_success[nearest_gate_index]
        if np.abs(gate_1_value - drain_source_value) < 1.2:
            response_1.append([drain_source_value, 1, j])
        else:
            response_1.append([drain_source_value, 0, j])
        gate_2_value = gate_2[j]
        nearest_gate_index = min(range(len(gate_success)), key=lambda i: abs(gate_success[i]-gate_2_value))
        drain_source_value = drain_voltage_success[nearest_gate_index]
        if np.abs(gate_2_value - drain_source_value) < 1.2:
            response_2.append([drain_source_value, 1, j])
        else:
            response_2.append([drain_source_value, 0, j])
    
    total_response_1 = np.array([i[0] for i in response_1])
    total_response_2 = np.array([i[0] for i in response_2])
    v_out_diff = total_response_1 - total_response_2
    
    hyd_allowed_response = []
    hyd_not_allowed_response = []
    for i in range(len(response_1)):
        response_1_allow = response_1[i][1]
        response_2_allow = response_2[i][1]
        if response_1_allow*response_2_allow == 1:
            hyd_allowed_response.append([v_out_diff[i], i])
        else:
            hyd_not_allowed_response.append([v_out_diff[i], i])
            
    plt.figure(figsize = (9,7))
    plt.title('Differential Amplifier Response, $V_{out}$ =' + str(supply_voltage) + 'V , $R_{1}$ =' + str(res_1) + '$\Omega$ , $R_{2}$ =' + str(res_2) + '$\Omega$')
    plt.xlabel('Iteration number')
    plt.ylabel('Voltage Response (V)')
    plt.plot(gate_1, c = 'r', linestyle= 'dashdot', label = 'OECT 1 Input')
    plt.plot(gate_2, c = 'b', linestyle= 'dashdot', label = 'OECT 2 Input')
    plt.plot([i[2] for i in response_1], [i[0] for i in response_1], c='r', linestyle = 'solid', label = 'OECT 1 Response')
    plt.plot([i[2] for i in response_2], [i[0] for i in response_2], c='b', linestyle = 'solid', label = 'OECT 2 Response')
    plt.plot(v_out_diff, c='g', label = 'Differential Response')
    plt.grid()
    plt.legend()
    
    plt.figure(figsize = (9,7))
    plt.title('Differential Amplifier Response Hydrolysis, $V_{out}$ =' + str(supply_voltage) + 'V , $R_{1}$ =' + str(res_1) + '$\Omega$ , $R_{2}$ =' + str(res_2) + '$\Omega$')
    plt.xlabel('Iteration number')
    plt.ylabel('Voltage Response (V)')
    plt.scatter([i[2] for i in response_1 if i[1] == 1], [i[0] for i in response_1 if i[1] == 1], marker = '.', c='r', label = 'OECT 1 Response Hydrolysis satisfied')
    plt.scatter([i[2] for i in response_1 if i[1] == 0], [i[0] for i in response_1 if i[1] == 0], marker = 'x', c='r', label = 'OECT 1 Response Hydrolysis not satisfied')
    plt.scatter([i[2] for i in response_2 if i[1] == 1], [i[0] for i in response_2 if i[1] == 1], marker = '.', c='b', label = 'OECT 2 Response Hydrolysis satisfied')
    plt.scatter([i[2] for i in response_2 if i[1] == 0], [i[0] for i in response_2 if i[1] == 0], marker = 'x', c='b', label = 'OECT 2 Response Hydrolysis not satisfied')
    plt.scatter([i[1] for i in hyd_allowed_response], [i[0] for i in hyd_allowed_response], marker = '.', c='g', label = 'Differential Response Hydrolysis satisfied')
    plt.scatter([i[1] for i in hyd_not_allowed_response], [i[0] for i in hyd_not_allowed_response], marker = 'x', c='g', label = 'Differential Response Hydrolysis not satisfied')
    plt.grid()
    plt.legend()


#-----------------------------------------------------#

# Common-source linear amplifier simulation

if simluate_common_source_amplifier == 'yes':
    
    # Circuit parameter values

    supply_voltage = -2.2
    source_res = 80
    drain_res = 740
    pot_div_res_1 = 1000
    pot_div_res_2 = 40
    
    # pot_div_res_2 determined later
    
    # Determine load-line on output characteristics
    # Finds gate-source values - use to find gate ground
    # Use gate-ground to find input range and value of pot_div_res_2
    
    voltage_resolution = 0.001
    drain_sweep = np.arange(supply_voltage, 0, voltage_resolution)
    gate_sweep = np.arange(-0.5, 0.5, voltage_resolution/10)
    gate_voltage_values = np.zeros(len(drain_sweep))
    for j in range(len(drain_sweep)):
        equation_to_solve = np.array((supply_voltage/(source_res + drain_res)) - (drain_sweep[j]/(source_res + drain_res)) - master_current(drain_sweep[j], gate_sweep, conductance, pinch_off, alpha, beta, gamma))
        sorted_array = equation_to_solve
        if sorted_array[0]*sorted_array[-1] < 0: # if a solution exists
            # linearly interpolate
            equation_to_solve = list(equation_to_solve)
            smallest_index = min(range(len(equation_to_solve)), key=lambda i: abs(equation_to_solve[i]-0))
            gate_1 = gate_sweep[smallest_index - 1]
            gate_2 = gate_sweep[smallest_index]
            if smallest_index == len(gate_sweep) - 1:
                gate_3 = gate_2
            else:
                gate_3 = gate_sweep[smallest_index + 1]
            drain_1 = (supply_voltage/(source_res + drain_res)) - (drain_sweep[j]/(source_res + drain_res)) - master_current(drain_sweep[j], gate_1, conductance, pinch_off, alpha, beta, gamma)
            drain_2 = (supply_voltage/(source_res + drain_res)) - (drain_sweep[j]/(source_res + drain_res)) - master_current(drain_sweep[j], gate_2, conductance, pinch_off, alpha, beta, gamma)
            drain_3 = (supply_voltage/(source_res + drain_res)) - (drain_sweep[j]/(source_res + drain_res)) - master_current(drain_sweep[j], gate_3, conductance, pinch_off, alpha, beta, gamma)
            if np.sign(drain_1*drain_2) == -1:
                #interpolate: (gate_1, drain_1) and (gate_2, drain_2) - find gate for drain = 0
                zero_drain_gate = gate_1 - (gate_2 - gate_1)*( drain_1 / (drain_2 - drain_1) )
            elif np.sign(drain_2*drain_3) == -1:
                #interpolate: (gate_2, drain_2) and (gate_3, drain_3) - find gate for drain = 0
                zero_drain_gate = gate_3 - (gate_2 - gate_3)*( drain_3 / (drain_2 - drain_3) )
            else:
                print('Transfer: Interpolation not found')
                zero_drain_gate = gate_2
            gate_voltage_values[j] = zero_drain_gate
        if (j+1) % 100 == 0:
            print(j+1, '/', str(len(drain_sweep)))
    gate_drain_combinations = [[gate_voltage_values[i], drain_sweep[i]] for i in range(len(drain_sweep)) if gate_voltage_values[i] != 0]
    drain_current_values = [((supply_voltage/(source_res + drain_res)) - (i[1]/(source_res + drain_res))) for i in gate_drain_combinations]
    gate_ground_values = []
    for i in range(len(drain_current_values)):
        gate_source = gate_drain_combinations[i][0]
        offset = drain_current_values[i]*source_res
        gate_ground = gate_source + offset
        gate_ground_values.append(gate_ground)
    
    # Plot gate-ground voltage relationship
    gate_source_voltage_list = [gate_drain_combinations[i][0] for i in range(len(gate_drain_combinations))]
    plt.figure(figsize = (6.2,5))
    plt.title('Gate Voltage Comparison')
    plt.xlabel('Gate-Ground Votlage (V)')
    plt.ylabel('Gate-Source Voltage (V)')
    plt.grid(b=True, which = 'major', linestyle = '-', color='darkgray')
    plt.grid(b=True, which = 'minor', linestyle = '--', color = 'lightgray')
    plt.plot(gate_ground_values, gate_source_voltage_list,'k')
    plt.tight_layout()
    
    gate_ground_drain_combinations = [[gate_ground_values[i],drain_sweep[i]] for i in range(len(drain_current_values))]
    hydrolysis_gate_source_drain = []
    hydrolysis_gate_ground_drain = []
    for i in range(len(drain_current_values)):
        gate_source = gate_drain_combinations[i][0]
        drain = gate_drain_combinations[i][1]
        if np.abs(gate_source - drain) < 1.2:
            hydrolysis_gate_source_drain.append(gate_drain_combinations[i])
            hydrolysis_gate_ground_drain.append(gate_ground_drain_combinations[i])
    beyond_pinch_index = min(range(len(hydrolysis_gate_source_drain)), key=lambda i: abs((hydrolysis_gate_source_drain[i][0] - hydrolysis_gate_source_drain[i][1]) - pinch_off))
    hydrolysis_gate_ground = [i[0] for i in hydrolysis_gate_ground_drain]
    highest_hydrolysis_gate_ground = max(hydrolysis_gate_ground)
    lowest_hydrolysis_gate_ground = hydrolysis_gate_ground[beyond_pinch_index]
    gate_ground_midpoint = (highest_hydrolysis_gate_ground + lowest_hydrolysis_gate_ground)/2
    # determine the second resistor value in potential divider
    gate_ground_midpoint = (pot_div_res_2 / (pot_div_res_2 + pot_div_res_1))*supply_voltage
    # pot_div_res_2 = pot_div_res_1 * gate_ground_midpoint / (supply_voltage - gate_ground_midpoint)
    
    gain_gate_source = [np.abs((gate_drain_combinations[i+1][1] - gate_drain_combinations[i][1])/(gate_drain_combinations[i+1][0] - gate_drain_combinations[i][0]) )
                            for i in range(len(gate_drain_combinations) - 1)]
    gain_gate_source.append(gain_gate_source[-1])
    gain_gate_ground = [np.abs((gate_ground_drain_combinations[i+1][1] - gate_ground_drain_combinations[i][1])/(gate_ground_drain_combinations[i+1][0] - gate_ground_drain_combinations[i][0]) )
                            for i in range(len(gate_drain_combinations) - 1)]
    gain_gate_ground.append(gain_gate_ground[-1])
    
    gate_ground_hyd_sat = [i[0] for i in hydrolysis_gate_ground_drain] 
    drain_hyd_sat = [i[1] for i in hydrolysis_gate_ground_drain]
    gain_gate_ground_hyd_sat = [[gate_ground_hyd_sat[i], np.abs(((drain_hyd_sat[i] - drain_hyd_sat[i-1])/(gate_ground_hyd_sat[i] - gate_ground_hyd_sat[i-1])))] for i in range(1,len(gate_ground_hyd_sat))]
    gate_source_hyd_sat = [i[0] for i in hydrolysis_gate_source_drain] 
    drain_hyd_sat = [i[1] for i in hydrolysis_gate_source_drain]
    gain_gate_source_hyd_sat = [[gate_source_hyd_sat[i], np.abs(((drain_hyd_sat[i] - drain_hyd_sat[i-1])/(gate_source_hyd_sat[i] - gate_source_hyd_sat[i-1])))] for i in range(1,len(gate_source_hyd_sat))]
    
    gate_ground_not_hyd_sat = [i[0] for i in gate_ground_drain_combinations if i not in hydrolysis_gate_ground_drain] 
    drain_not_hyd_sat = [i[1] for i in gate_ground_drain_combinations if i not in hydrolysis_gate_ground_drain]
    gain_gate_ground_not_hyd_sat = [[gate_ground_not_hyd_sat[i], np.abs(((drain_not_hyd_sat[i] - drain_not_hyd_sat[i-1])/(gate_ground_not_hyd_sat[i] - gate_ground_not_hyd_sat[i-1])))] for i in range(1,len(gate_ground_not_hyd_sat))]
    gate_source_not_hyd_sat = [i[0] for i in gate_drain_combinations if i not in hydrolysis_gate_source_drain] 
    drain_not_hyd_sat = [i[1] for i in gate_drain_combinations if i not in hydrolysis_gate_source_drain]
    gain_gate_source_not_hyd_sat = [[gate_source_not_hyd_sat[i], np.abs(((drain_not_hyd_sat[i] - drain_not_hyd_sat[i-1])/(gate_source_not_hyd_sat[i] - gate_source_not_hyd_sat[i-1])))] for i in range(1,len(gate_ground_not_hyd_sat))]
    
    
    # fig, ax1 = plt.subplots(figsize = (9,7))
    # plt.title('Common-source Amplifier Transfer Curve')
    # color = 'tab:blue'
    # ax1.set_xlabel('Gate-Source Voltage (V)', color=color)
    # ax1.set_ylabel('Drain-Source Voltage (V)') 
    # lns1 = ax1.plot( [i[0] for i in gate_drain_combinations if i not in hydrolysis_gate_source_drain] , [i[1] for i in gate_drain_combinations if i not in hydrolysis_gate_source_drain] , color=color, linestyle = 'dotted', label = 'Gate-source Voltage')
    # lns2 = ax1.plot( [i[0] for i in hydrolysis_gate_source_drain] , [i[1] for i in hydrolysis_gate_source_drain] , color=color, linestyle = 'solid', label = 'Gate-source Voltage')
    # lns7 = ax1.plot( [] , [] , color='k', linestyle = 'dotted', label = 'Hydrolysis Not Satisfied')
    # lns8 = ax1.plot( [] , [] , color='k', linestyle = 'solid', label = 'Hydrolysis Satisfied')
    # ax1.tick_params(axis='x', labelcolor=color)
    # ax2 = ax1.twiny()  # instantiate a second axes that shares the same y-axis
    # color = 'tab:red'
    # ax2.set_xlabel('Gate-Ground Voltage (V)', color=color)  # we already handled the x-label with ax1
    # lns3 = ax2.plot( [i[0] for i in gate_ground_drain_combinations if i not in hydrolysis_gate_ground_drain] , [i[1] for i in gate_ground_drain_combinations if i not in hydrolysis_gate_ground_drain] , color=color, linestyle = 'dotted', label = 'Gate-ground Voltage')
    # lns4 = ax2.plot( [i[0] for i in hydrolysis_gate_ground_drain] , [i[1] for i in hydrolysis_gate_ground_drain] , color=color, linestyle = 'solid', label = 'Gate-ground Voltage')
    # ax2.tick_params(axis='x', labelcolor=color)
    # ax3 = ax1.twinx()
    # ax4 = ax2.twinx()
    # ax4.axes.yaxis.set_visible(False)
    # ax4 = ax3.twiny() 
    # ax4.axes.xaxis.set_visible(False)
    # lns5 = ax3.plot([i[0] for i in gain_gate_source_hyd_sat], [i[1] for i in gain_gate_source_hyd_sat], color = 'blue', linestyle = 'solid', label = 'Gate-source Gain')
    # lns6 = ax3.plot([i[0] for i in gain_gate_source_not_hyd_sat], [i[1] for i in gain_gate_source_not_hyd_sat], color = 'blue', linestyle = 'dotted', label = 'Gate-source Gain')
    # lns9 = ax4.plot([i[0] for i in gain_gate_ground_hyd_sat], [i[1] for i in gain_gate_ground_hyd_sat], color = 'red', linestyle = 'solid', label = 'Gate-ground Gain')
    # lns10 = ax4.plot([i[0] for i in gain_gate_ground_not_hyd_sat], [i[1] for i in gain_gate_ground_not_hyd_sat], color = 'red', linestyle = 'dotted', label = 'Gate-ground Gain')
    # ax3.set_ylabel('Gain')
    # lns = lns2 + lns5 + lns4 + lns9 + lns8 + lns7
    # labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, fontsize = 11)
    
    fig, ax1 = plt.subplots(figsize = (6.2,5))
    plt.title('Common-source Amplifier Transfer Curve')
    color = 'tab:blue'
    ax1.set_xlabel('Gate-Ground Voltage (V)')
    ax1.set_ylabel('Drain-Source (Output) Voltage (V)', color=color) 
    ax1.set_xlim([1.1*min([i[0] for i in hydrolysis_gate_ground_drain]), max([i[0] for i in gate_ground_drain_combinations if i not in hydrolysis_gate_ground_drain])])
    lns7 = ax1.plot( [] , [] , color='k', linestyle = 'dotted', label = 'Outside\nHydrolysis')
    lns8 = ax1.plot( [] , [] , color='k', linestyle = 'solid', label = 'Within\nHydrolysis')
    lns3 = ax1.plot( [i[0] for i in gate_ground_drain_combinations if i not in hydrolysis_gate_ground_drain] , [i[1] for i in gate_ground_drain_combinations if i not in hydrolysis_gate_ground_drain] , color=color, linestyle = 'dotted')
    lns4 = ax1.plot( [i[0] for i in hydrolysis_gate_ground_drain] , [i[1] for i in hydrolysis_gate_ground_drain] , color=color, linestyle = 'solid')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    #ax2.set_xlabel('Amplifier Gain')
    ax2.tick_params(axis='y', labelcolor=color)
    lns9 = ax2.plot([i[0] for i in gain_gate_ground_hyd_sat], [i[1] for i in gain_gate_ground_hyd_sat], color = color, linestyle = 'solid')
    lns10 = ax2.plot([i[0] for i in gain_gate_ground_not_hyd_sat], [i[1] for i in gain_gate_ground_not_hyd_sat], color = color, linestyle = 'dotted')
    ax2.set_ylabel('|Gain|', color=color)
    lns = lns8 + lns7
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, fontsize = 11)
    
    # # Simulate device response (NOT TEMPORAL)
    # # Find gate-ground response and define drain-source voltage as output
    # t = np.arange(0,10,0.01)
    # V_in = 0.05*np.cos(2*np.pi*t)
    # V_gate_ground = V_in + gate_ground_midpoint
    # response = []
    # for j in range(len(t)):
    #     gate_value = V_gate_ground[j]
    #     nearest_gate_index = min(range(len(hydrolysis_gate_ground)), key=lambda i: abs(hydrolysis_gate_ground[i]-gate_value))
    #     drain_source_value = hydrolysis_gate_ground_drain[nearest_gate_index][1]
    #     gate_source_value = hydrolysis_gate_source_drain[nearest_gate_index][0]
    #     if np.abs(gate_source_value - drain_source_value) < 1.2:
    #         response.append([drain_source_value, 1, j])
    #     else:
    #         response.append([drain_source_value, 0, j])
    # average_gain = np.abs((max([i[0] for i in response]) - min([i[0] for i in response]))/(max(V_gate_ground) - min(V_gate_ground)))
    # hyd_allowed_response = [ [i[0],i[2]] for i in response if i[1] == 1]
    # hyd_not_allowed_response = [ [i[0],i[2]] for i in response if i[1] == 0]
    
    # print('Average Gain: ' , str(average_gain))
    # plt.figure(figsize = (11,7))
    # plt.title('Common-source Amplifier Response Hydrolysis, $V_{out}$ =' + str(supply_voltage) + 'V , $R_{S}$ =' + str(source_res) + '$\Omega$ , $R_{D}$ =' + str(drain_res) + '$\Omega$')
    # plt.xlabel('Iteration number')
    # plt.ylabel('Voltage Response (V)')
    # plt.plot(V_gate_ground, c = 'r', linestyle= 'dashdot', label = 'OECT 1 Input')
    # plt.scatter([i[1] for i in hyd_allowed_response], [i[0] for i in hyd_allowed_response], marker = '.', c='g', label = 'Amplifier Response Hydrolysis satisfied')
    # plt.scatter([i[1] for i in hyd_not_allowed_response], [i[0] for i in hyd_not_allowed_response], marker = 'x', c='g', label = 'Amplifier Response Hydrolysis not satisfied')
    # plt.grid()
    # plt.legend()

    
    # Simulate device response (TEMPORAL)
    
    input_sampling_rate = 1e3
    time_values = np.arange(0,5,1/input_sampling_rate)
    input_frequency = 1
    V_in = 0.02*np.sin(2*input_frequency*np.pi*time_values)
    V_gate_ground = V_in + gate_ground_midpoint
    amplifier_sampling_frequency = 5e3
    
    amplifier_response_time_constant = oect_response_time_constant # actual time, not constant
    
    if input_sampling_rate <= 0.5*amplifier_sampling_frequency:
        amplifier_times, amplifier_voltages = amplifier_response(amplifier_sampling_frequency, time_values, V_gate_ground, amplifier_response_time_constant, gate_ground_drain_combinations)
        amplifier_gain = ( max(amplifier_voltages) - min(amplifier_voltages) ) / ( max(V_gate_ground) - min(V_gate_ground) )
        print('Gain: ', amplifier_gain)
        #plt.figure()
        plt.title('Temporal Amplifier Simulation')
        plt.plot(amplifier_times, amplifier_voltages)
        plt.xlabel('Time (s)')
        plt.ylabel('Output Voltage (V)')
    else:
        print('Input sampling frequency needs to be at least 1/2 the amplifier sampling frequency.')
        
    # Simulate res 2 variations
    
    input_sampling_rate = 1e3
    time_values = np.arange(0,5,1/input_sampling_rate)
    input_frequency = 1
    V_in = 0.02*np.sin(2*input_frequency*np.pi*time_values)
    resistor_2_values = [250, 200, 150, 100, 50]
    gain_res_2_values = []
    plt.figure()
    plt.title('Amplifier Output Simulation')
    plt.xlabel('Time (s)')
    plt.ylabel('Output Voltage (V)')
    for res_value in resistor_2_values:
        gate_ground_midpoint = (res_value/(pot_div_res_1 + res_value))*supply_voltage
        V_gate_ground = V_in + gate_ground_midpoint
        amplifier_sampling_frequency = 5e3
        amplifier_response_time_constant = oect_response_time_constant # actual time, not constant
        amplifier_times, amplifier_voltages = amplifier_response(amplifier_sampling_frequency, time_values, V_gate_ground, amplifier_response_time_constant, gate_ground_drain_combinations)
        amplifier_gain = ( max(amplifier_voltages) - min(amplifier_voltages) ) / ( max(V_gate_ground) - min(V_gate_ground) )
        plt.plot(amplifier_times, amplifier_voltages, label = str(res_value) + ' $\Omega$\n(' + str(round(amplifier_gain,2)) + ')')
        gain_res_2_values.append(amplifier_gain)
        print('Simulated resistance ', res_value)
    plt.grid(b=True, which = 'major', linestyle = '-', color='darkgray')
    plt.grid(b=True, which = 'minor', linestyle = '--', color = 'lightgray')
    plt.xlim([-0.2, 7.2])
    plt.legend(title = '$R_2$ Value\n(|Gain|)', loc = 'upper right')
    
    if common_source_amp_char == 'yes':
        thd_imd_amplitudes = [10e-6, 30e-6, 80e-6, 100e-6, 250e-6, 500e-6,
                          750e-6, 1e-3, 2e-3, 4e-3, 5e-3, 10e-3] # p2p = 2 x values
        thd_imd_amplitudes = [100e-6, 250e-6, 500e-6, 
                              1e-3, 3e-3, 5e-3, 
                              10e-3, 25e-3, 50e-3,
                              10e-2,20e-2] # p2p = 2 x values
        thd_frequency = 5
        number_of_harmonics = 7
        f1 = 4.9 # IMD frequency 1
        f2 = 5.1 # IMD frequency 2
        number_of_amps = len(thd_imd_amplitudes)
        frequencies_to_analyse = [ f1 , f2 , 3*f1 , 3*f2 , (2*f1 + f2) , (2*f2 + f1) , (2*f1 - f2) , (2*f2 - f1) ]
        imd_values_output = np.zeros(shape = [number_of_amps, len(frequencies_to_analyse)])
        imd_values_input = np.zeros(shape = [number_of_amps, len(frequencies_to_analyse)])

        # THD 
        final_thd = []        
        for i in range(number_of_amps):
            input_sampling_rate = 1e3
            time_values = np.arange(0,10,1/input_sampling_rate)
            input_frequency = thd_frequency
            V_in = thd_imd_amplitudes[i]*np.cos(2*input_frequency*np.pi*time_values)
            V_gate_ground = V_in + gate_ground_midpoint
            amplifier_sampling_frequency = 5e3
            amplifier_times, amplifier_voltages = amplifier_response(amplifier_sampling_frequency, time_values, V_gate_ground, amplifier_response_time_constant, gate_ground_drain_combinations)
            fft_ds_volt = list(fft_.fft(amplifier_voltages))
            time_step = amplifier_times[2] - amplifier_times[1]
            freq_dom = list(fft_.fftfreq(len(fft_ds_volt) , time_step)) 
            thd_values = np.zeros(shape = [number_of_harmonics])
            harmonic_frequencies = list(thd_frequency*np.linspace(1, number_of_harmonics, number_of_harmonics))
            for j in range(len(harmonic_frequencies)): 
                nearest_index = min(range(len(freq_dom)), key=lambda k: abs(freq_dom[k]-harmonic_frequencies[j]))
                if nearest_index > len(freq_dom)/2 - 5:
                    # frequencies domain not large enough for harmonic
                    pass
                else:
                    amplitude = fft_ds_volt[nearest_index]
                    thd_values[j] = np.abs(amplitude)
            THD = (np.sqrt(sum(np.square(thd_values[1:]))) / thd_values[0]) * 100
            final_thd.append(THD)
            print('Found THD for amplitude ', i+1)
        # Find the gate voltage value for THD = 1%
        gate_amplitudes = np.abs(np.array(thd_imd_amplitudes))
        surround_indices = []
        cut_off_thd = 1
        for i in range(len(final_thd) - 1):
            if final_thd[i] > cut_off_thd:
                if final_thd[i+1] < cut_off_thd:
                    surround_indices.append([i,i+1])
            if final_thd[i] < cut_off_thd:
                if final_thd[i+1] > cut_off_thd:
                    surround_indices.append([i,i+1])
        inter_gates = []
        for s_indices in surround_indices:
            surround_thd = [final_thd[s_indices[0]], final_thd[s_indices[1]]]
            surround_gate = [gate_amplitudes[s_indices[0]], gate_amplitudes[s_indices[1]]]
            y1 = surround_thd[0]
            y2 = surround_thd[1]
            x1 = surround_gate[0]
            x2 = surround_gate[1]
            inter_gate = ( (x2 - x1)*(cut_off_thd - y1) / (y2 - y1) ) + x1
            inter_gates.append(inter_gate)
        if len(inter_gates) > 1:
            input_dynamic_range = inter_gates[-1] - inter_gates[0]
            print('Input Dynamic Range: ', input_dynamic_range)
        inter_gates_plot = 1e6 * np.array(inter_gates)
        gate_amplitudes_plot = 1e6 * gate_amplitudes
        plt.figure(figsize = (6.2,5))
        plt.xlabel('Gate Input Peak Amplitude ($\mu$V)')
        plt.title('Simulated Total Harmonic Distortion (THD)')
        plt.ylabel('7-harmonic THD (%)')
        plt.grid(b=True, which = 'major', linestyle = '-', color='darkgray')
        plt.grid(b=True, which = 'minor', linestyle = '--', color = 'lightgray')
        plt.plot(gate_amplitudes_plot, final_thd, 'kx-', linewidth = 1.5, label = 'THD')
        plt.xlim([0.5*min(gate_amplitudes_plot),1.5*max(gate_amplitudes_plot)])
        for inter_gate in inter_gates_plot:
            plt.plot([0,inter_gate],[cut_off_thd,cut_off_thd],'r--',label='1% THD at ' + str(int(inter_gate)) + ' $\mu$V')
            plt.plot([inter_gate,inter_gate],[0,cut_off_thd],'r--')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        #print('1% THD at ' + str(int(1000000*inter_gate)) + ' $\mu$V')
    
        # BODE
        gate_amplitude = 0.01
        start = 1e-3
        stop = 1e6
        multiple = 1.5
        frequencies_to_test = exp_list(start, stop, multiple)
        print('Number of frequencies: ', len(frequencies_to_test))
        input_sampling_rate = [100*i for i in frequencies_to_test]
        response_sampling_rate = [10*i for i in input_sampling_rate]
        time_values_for_frequencies = [np.arange(0, 1.5/frequencies_to_test[i], 1/input_sampling_rate[i]) for i in range(len(frequencies_to_test))]
        gate_voltages = [gate_amplitude*np.sin(frequencies_to_test[i]*2*np.pi*time_values_for_frequencies[i]) + gate_ground_midpoint for i in range(len(frequencies_to_test))]
        response_values = []
        gain_values = []
        for i in range(len(frequencies_to_test)):
            amplifier_times, amplifier_voltages = amplifier_response(response_sampling_rate[i], time_values_for_frequencies[i], gate_voltages[i], amplifier_response_time_constant, gate_ground_drain_combinations)
            response_values.append([amplifier_times, amplifier_voltages])
            current_p2p = max(amplifier_voltages) - min(amplifier_voltages)
            transcond = current_p2p/(2*gate_amplitude)
            gain_values.append(transcond)
            print('Bode: Frequency completed: ', frequencies_to_test[i])
        plt.figure(figsize=(6.2,5))
        plt.xscale('log')
        plt.plot(frequencies_to_test, gain_values, 'k')
        plt.grid()
        plt.title('Simulated Bode Plot')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        cut_off_trans = max(gain_values) / np.sqrt(2)
        cut_off_trans_fraction = 1 / np.sqrt(2)
        cut_off_index = 0
        for i in range(len(gain_values)):
            if gain_values[i] < cut_off_trans:
                cut_off_index = i
                break
        frequency_1 = frequencies_to_test[cut_off_index-1]
        frequency_2 = frequencies_to_test[cut_off_index]
        trans_1 = gain_values[cut_off_index-1]
        trans_2 = gain_values[cut_off_index]
        gradient = (trans_2 - trans_1) / (frequency_2 - frequency_1)
        cut_off_freq = (cut_off_trans - trans_1) / gradient + frequency_1
        print('The cut-off gain was ', cut_off_trans)
        print('The cut-off frequency was ', cut_off_freq, 'Hz')
        plt.plot([0,cut_off_freq],[cut_off_trans, cut_off_trans], 'r', linestyle = 'dashed')
        plt.plot([cut_off_freq,cut_off_freq],[-1, cut_off_trans], 'r', linestyle = 'dashed', label = 'Cut-off frequency:\n' + str(int(cut_off_freq)) + 'Hz')
        plt.ylim([-0.1, 1.1*max(gain_values)])
        plt.legend()
        
        # IMD
        for i in range(number_of_amps):
            input_sampling_rate = 1e3
            time_values = np.arange(0,10,1/input_sampling_rate)
            input_frequency = thd_frequency
            V_in_1 = thd_imd_amplitudes[i]*np.cos(2*f1*np.pi*time_values)
            V_in_2 = thd_imd_amplitudes[i]*np.cos(2*f2*np.pi*time_values)
            V_in = V_in_1 + V_in_2
            V_gate_ground = V_in + gate_ground_midpoint
            amplifier_sampling_frequency = 5e3
            amplifier_times, amplifier_voltages = amplifier_response(amplifier_sampling_frequency, time_values, V_gate_ground, amplifier_response_time_constant, gate_ground_drain_combinations)
            time_step = amplifier_times[2] - amplifier_times[1]
            freq_dom, fft_ds_voltage = signal.periodogram(amplifier_voltages, fs=1/time_step)
            freq_dom_gate, fft_gate = signal.periodogram(V_gate_ground, fs=input_sampling_rate)
            fft_dB = 10*np.log10(fft_ds_voltage) # units of decibel Volts (dBV)
            fft_gate_dB = 10*np.log10(fft_gate)
            for j in range(len(frequencies_to_analyse)):
                frequency = frequencies_to_analyse[j]
                nearest_fft_index = min(range(len(freq_dom)), key=lambda k: abs(freq_dom[k]-frequency))
                fft_value_dB = fft_dB[nearest_fft_index]
                imd_values_output[i,j] = fft_value_dB
                nearest_fft_index = min(range(len(freq_dom_gate)), key=lambda k: abs(freq_dom_gate[k]-frequency))
                fft_gate_value_dB = fft_gate_dB[nearest_fft_index]
                imd_values_input[i,j] = fft_gate_value_dB
            print('IMD: Analysed amplitude ', i+1, ' of ', number_of_amps)
        fundamental_input = imd_values_input[:,0]
        fundamental_output = imd_values_output[:,0]
        imd3_input = imd_values_input[5:,0]
        imd3_output = imd_values_output[5:,4]
        def fit_func(input_power, m, c):
            return m*input_power + c
        p0_fundamental, p0_fundamental_cov = spo.curve_fit(fit_func, fundamental_input, fundamental_output, maxfev=10000000) 
        p0_imd3, p0_imd3_cov = spo.curve_fit(fit_func, imd3_input, imd3_output, maxfev=10000000) 
        fundamental_grad, fundamental_intercept = p0_fundamental[0], p0_fundamental[1]
        imd3_grad, imd3_intercept = p0_imd3[0], p0_imd3[1]
        input_intercept = (imd3_intercept - fundamental_intercept) / (fundamental_grad - imd3_grad)
        output_intercept = fundamental_grad*input_intercept + fundamental_intercept
        imd3_point = [input_intercept, output_intercept]
        print('IMD3 point occurs at ', imd3_point)
        dist_from_nearest_point_fund = [(input_intercept-imd_values_input[-1,0]),(output_intercept-imd_values_output[-1,0])]
        print('From the highest fundamental value, the distance to the closest point is ', np.mean(dist_from_nearest_point_fund))
        # Plot results
        plt.figure(figsize = (6.2,5))
        plt.xlabel('Input Power (dBV)')
        plt.ylabel('Output Power (dBV)')
        plt.title('Simulated Intermodulation Distortion')
        plt.grid(b=True, which = 'major', linestyle = '-', color='darkgray')
        plt.grid(b=True, which = 'minor', linestyle = '--', color = 'lightgray')
        plt.plot(imd_values_input[:,0], imd_values_output[:,0], color = 'r', marker = 'x', label = 'Fund., $f_1$')
        plt.plot(imd_values_input[:,1], imd_values_output[:,1], marker = 'x', label = 'Fund., $f_2$')
        #plt.plot(imd_values_input[:,0], imd_values_output[:,6], color = 'b' , marker = 'o', label = 'IMD$_3$, $3f_1$')
        #plt.plot(imd_values_input[:,0], imd_values_output[:,7], marker = 'o', label = 'IMD$_3$, $3f_2$')
        #plt.plot(imd_values_input[:,0], imd_values_output[:,2], marker = 'o', label = 'IMD$_3$, $2f_1 + f_2$')
        #plt.plot(imd_values_input[:,0], imd_values_output[:,3], marker = 'o', label = 'IMD$_3$, $2f_2 + f_1$')
        plt.plot(imd_values_input[:,0], imd_values_output[:,4], marker = 'o', label = 'IMD$_3$, $2f_1 - f_2$')
        plt.plot(imd_values_input[:,0], imd_values_output[:,5], marker = 'o', label = 'IMD$_3$, $2f_2 - f_1$')
        plt.plot(np.arange(imd_values_input[-3,0], imd_values_input[-3,0] + 60, 10), fundamental_grad*np.arange(imd_values_input[-3,0], imd_values_input[-3,0] + 60, 10)+fundamental_intercept, 'k--', label = 'Extrapolation')
        plt.plot(np.arange(imd_values_input[-3,0], imd_values_input[-3,0] + 60, 10), imd3_grad*np.arange(imd_values_input[-3,0], imd_values_input[-3,0] + 60, 10)+imd3_intercept, 'k--')
        plt.legend(ncol=1,fontsize=12)
        plt.tight_layout()
        
        print('Conductance: ', conductance_value)
        print('Pinch off: ', pinch_off_value)
        print('Response Time Constant: ', oect_response_time_constant)
        print('alpha =', alpha_fit, ', beta =', beta_fit, ', gamma =', gamma_fit)
        print('Supply Voltage: ', supply_voltage)
        print('Source Resistor: ', source_res)
        print('Drain Resistor: ', drain_res)
        print('Potential divider resistor 2: ', pot_div_res_2)
        print('Gain: ', amplifier_gain)
        print('1% THD at ' + str(int(1000000*inter_gate)) + ' $\mu$V')
        try:
            print('Input Dynamic Range: ', input_dynamic_range)
        except:
            print('Input Dynamic Range not found')
            pass
        print('The cut-off gain was ', cut_off_trans)
        print('The cut-off frequency was ', cut_off_freq, 'Hz')
        print('IMD3 point occurs at ', imd3_point)
