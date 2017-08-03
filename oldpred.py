'''
oldpred.py
This file contains the get_old_pred and analyze_old_pred_data functions.


Main Function:
         get_old_pred(bg_df, start_index, end_index, num_pred_minutes):

Input:
        bg_df                           The bg dataframe.
        start_index                     The start index of the data.
        end_index                       The end index of the data.
        num_pred_minutes                The number of minutes in the future the prediction is for.

Output:
        eventual_pred_data              The namedtuple with eventualBG data composed of ['result_actual_bg_array', 'result_actual_bg_time_array', 'result_pred_array', 'result_pred_time_array'].
        iob_pred_data                   The namedtuple with iob data composed of ['result_actual_bg_array', 'result_actual_bg_time_array', 'result_pred_array', 'result_pred_time_array'].
        cob_pred_data                   The namedtuple with cob data composed of ['result_actual_bg_array', 'result_actual_bg_time_array', 'result_pred_array', 'result_pred_time_array'].
        acob_pred_data                  The namedtuple with acob data composed of ['result_actual_bg_array', 'result_actual_bg_time_array', 'result_pred_array', 'result_pred_time_array'].

USAGE:
    eventual_pred_data, iob_pred_data, cob_pred_data, acob_pred_data = get_old_pred(bg_df, start_test_index, end_test_index, pred_minutes)


Main Function:
        analyze_old_pred_data(old_pred_data, show_pred_plot, save_pred_plot, show_clarke_plot, save_clarke_plot, id_str, algorithm_str, minutes_str)

Input:
        old_pred_data                   The namedtuple with ['result_actual_bg_array', 'result_actual_bg_time_array', 'result_pred_array', 'result_pred_time_array']
        show_pred_plot                  Boolean to show the plots of prediction and actual bg values.
        save_pred_plot                  Boolean to save the plot of prediction and actual bg values
        show_clarke_plot                Boolean to show the Clarke Error Grid plot.
        save_clarke_plot                Boolean to save the Clarke Error Grid plot.
        id_str                          String of the ID number.
        algorithm_str                   String of the algorithm name.
        minutes_str                     String with the prediction minutes.

USAGE:
        analyze_old_pred_data(old_pred_data, True, False, True, False, "00897741", "Linear Regression", "Pred30")

Trevor Tsue
2017-7-26
'''

import numpy as np
from collections import namedtuple
import math
from sklearn.metrics import mean_squared_error
import ClarkeErrorGrid
import matplotlib.pyplot as plt


#The number of minutes that each time point is spaced out in the data (e.g. data is taken every 5 minutes)
DATA_SPACING = 5
#Defines the range such that any actual BG within this range will be compared to the predBG.
#(e.g. if predBG is at 0 min and there is no actual BG at 30 min, this ACTUAL_BG_RANGE will accept an actualBG
#the time 30 - ACTUAL_BG_RANGE < x < 30 + ACTUAL_BG_RANGE, or in this case, 25 < x < 35)
ACTUAL_BG_RANGE = 5



#Function to create the new prediction array, prediction time array, curr, and number of missed
def _new_pred_array(start_index, end_index, total_len):
    pred_array = np.zeros(total_len)
    time_array = np.zeros(total_len)
    curr = miss = 0

    return pred_array, time_array, curr, miss


#Function to get the eventualBG and actual BG. The predicted array index is 0 for the actual current BG
#and NUM_PRED_SECTIONS for the eventualBG
def _get_other_bg(bg_df, pred_array, pred_time_array, curr, miss, start_index, data_index, bg_str, pred_array_index):
    pred_time_array[curr] = (bg_df.iloc[data_index]['created_at'] - bg_df.iloc[start_index]['created_at']) / np.timedelta64(1, 'm')

    try:
        pred_array[curr] = bg_df.iloc[data_index]['openaps']['enacted'][bg_str]
        curr += 1

    except:
        try:
            pred_array[curr] = bg_df.iloc[data_index]['openaps']['suggested'][bg_str]
            curr += 1

        except:
            miss += 1

    return pred_array, pred_time_array, curr, miss


#Function to get the predicted bg for the IOB, COB, and aCOB predictions
def _get_named_pred(bg_df, pred_array, pred_time_array, curr, miss, start_index, data_index, pred_str, pred_array_index):
    pred_time_array[curr] = (bg_df.iloc[data_index]['created_at'] - bg_df.iloc[start_index]['created_at']) / np.timedelta64(1, 'm')

    try:
        pred_array[curr] = bg_df.iloc[data_index]['openaps']['enacted']['predBGs'][pred_str][pred_array_index]
        curr += 1

    except:
        try:
            pred_array[curr] = bg_df.iloc[data_index]['openaps']['suggested']['predBGs'][pred_str][pred_array_index]
            curr += 1

        except:
            miss += 1

    return pred_array, pred_time_array, curr, miss


#Function to get the raw actual bg array and the old prediction arrays (eventualBG, IOB, COB, aCOB)
#directly from the dataframe
def _get_raw_pred_array(bg_df, start_index, end_index, pred_array_index):
    total_len = start_index - end_index + 1

    actual_bg_array, actual_bg_time_array, actual_curr, actual_miss = _new_pred_array(start_index, end_index, total_len)
    eventual_pred_array, eventual_pred_time_array, eventual_curr, eventual_miss = _new_pred_array(start_index, end_index, total_len)
    iob_pred_array, iob_pred_time_array, iob_curr, iob_miss = _new_pred_array(start_index, end_index, total_len)
    cob_pred_array, cob_pred_time_array, cob_curr, cob_miss = _new_pred_array(start_index, end_index, total_len)
    acob_pred_array, acob_pred_time_array, acob_curr, acob_miss = _new_pred_array(start_index, end_index, total_len)

    for data_index in range(start_index, end_index - 1, -1):
        actual_bg_array, actual_bg_time_array, actual_curr, actual_miss= _get_other_bg(bg_df, actual_bg_array, actual_bg_time_array, actual_curr, actual_miss, start_index, data_index, 'bg', 0)

        eventual_pred_array, eventual_pred_time_array, eventual_curr, eventual_miss = _get_other_bg(bg_df, eventual_pred_array, eventual_pred_time_array,
                                                                                                        eventual_curr, eventual_miss, start_index, data_index, 'eventualBG', pred_array_index)
        iob_pred_array, iob_pred_time_array, iob_curr, iob_miss = _get_named_pred(bg_df, iob_pred_array, iob_pred_time_array,
                                                                                    iob_curr, iob_miss, start_index, data_index, 'IOB', pred_array_index)
        cob_pred_array, cob_pred_time_array, cob_curr, cob_miss = _get_named_pred(bg_df, cob_pred_array, cob_pred_time_array,
                                                                                    cob_curr, cob_miss, start_index, data_index, 'COB', pred_array_index)
        acob_pred_array, acob_pred_time_array, acob_curr, acob_miss = _get_named_pred(bg_df, acob_pred_array, acob_pred_time_array,
                                                                                    acob_curr, acob_miss, start_index, data_index, 'aCOB', pred_array_index)

    #Resize arrays to remove missed data points
    actual_bg_array = np.resize(actual_bg_array, total_len - actual_miss)
    actual_bg_time_array = np.resize(actual_bg_time_array, total_len - actual_miss)

    eventual_pred_array = np.resize(eventual_pred_array, total_len - eventual_miss)
    eventual_pred_time_array = np.resize(eventual_pred_time_array, total_len - eventual_miss)

    iob_pred_array = np.resize(iob_pred_array, total_len - iob_miss)
    iob_pred_time_array = np.resize(iob_pred_time_array, total_len - iob_miss)

    cob_pred_array = np.resize(cob_pred_array, total_len - cob_miss)
    cob_pred_time_array = np.resize(cob_pred_time_array, total_len - cob_miss)

    acob_pred_array = np.resize(acob_pred_array, total_len - acob_miss)
    acob_pred_time_array = np.resize(acob_pred_time_array, total_len - acob_miss)

    return actual_bg_array, actual_bg_time_array, eventual_pred_array, eventual_pred_time_array, iob_pred_array, iob_pred_time_array, cob_pred_array, cob_pred_time_array, acob_pred_array, acob_pred_time_array


#Finds the nearest value in time to the given values in the given array. Otherwise, it returns -1
def find_nearest_index(array, value):
    nearest_index = (np.abs(array-value)).argmin()
    if (int(np.abs(array[nearest_index] - value)) < ACTUAL_BG_RANGE):
        return nearest_index
    else:
        return -1


#Returns the arrays such that the predBG corresponds to the actualBG in NUM_PRED_MINUTES in the future
def _find_compare_array(actual_bg_array, actual_bg_time_array, pred_array, pred_time_array, num_pred_minutes):
    array_len = len(pred_array)

    result_actual_bg_array = np.zeros(array_len)
    result_actual_bg_time_array = np.zeros(array_len)
    result_pred_array = np.zeros(array_len)
    result_pred_time_array = np.zeros(array_len)
    curr = miss = 0

    for array_index in range(array_len):
        #The time that the prediction is predicting for
        future_time = int(pred_time_array[array_index]) + num_pred_minutes
        nearest_index = find_nearest_index(actual_bg_time_array, future_time)

        if nearest_index == -1:
            miss += 1
        else:
            result_actual_bg_array[curr] = actual_bg_array[nearest_index]
            result_actual_bg_time_array[curr] = actual_bg_time_array[nearest_index]
            result_pred_array[curr] = pred_array[array_index]
            result_pred_time_array[curr] = future_time
            curr += 1

    result_actual_bg_array = np.resize(result_actual_bg_array, array_len - miss)
    result_actual_bg_time_array = np.resize(result_actual_bg_time_array, array_len - miss)
    result_pred_array = np.resize(result_pred_array, array_len - miss)
    result_pred_time_array = np.resize(result_pred_time_array, array_len - miss)

    #Created namedtuple to hold the data
    OldPredData = namedtuple('OldPredData', ['result_actual_bg_array', 'result_actual_bg_time_array', 'result_pred_array', 'result_pred_time_array'])

    return OldPredData(result_actual_bg_array, result_actual_bg_time_array, result_pred_array, result_pred_time_array)


#This function takes in the bg dataframe, the start and end indices, and the numebr of minutes of the prediction
#The number of minutes in the future that you want to make a prediction for (e.g. make a prediction for 30 minutes in the future).

#It returns the namedtuple with the following attributes ['result_actual_bg_array', 'result_actual_bg_time_array', 'result_pred_array', 'result_pred_time_array']
def get_old_pred(bg_df, start_index, end_index, num_pred_minutes):
    #The number of 5 minute sections until the prediction (e.g. 30 minutes = 6 sections)
    pred_array_index = num_pred_minutes / DATA_SPACING

    actual_bg_array, actual_bg_time_array, eventual_pred_array, eventual_pred_time_array, iob_pred_array, iob_pred_time_array, cob_pred_array, cob_pred_time_array, acob_pred_array, acob_pred_time_array = _get_raw_pred_array(bg_df, start_index, end_index, pred_array_index)

    eventual_pred_data = _find_compare_array(actual_bg_array, actual_bg_time_array, eventual_pred_array, eventual_pred_time_array, num_pred_minutes)
    iob_pred_data = _find_compare_array(actual_bg_array, actual_bg_time_array, iob_pred_array, iob_pred_time_array, num_pred_minutes)
    cob_pred_data= _find_compare_array(actual_bg_array, actual_bg_time_array, cob_pred_array, cob_pred_time_array, num_pred_minutes)
    acob_pred_data = _find_compare_array(actual_bg_array, actual_bg_time_array, acob_pred_array, acob_pred_time_array, num_pred_minutes)

    return eventual_pred_data, iob_pred_data, cob_pred_data, acob_pred_data



#Plots old pred data
def analyze_old_pred_data(old_pred_data, show_pred_plot, save_pred_plot, show_clarke_plot, save_clarke_plot, id_str, algorithm_str, minutes_str):
    actual_bg_array = old_pred_data.result_actual_bg_array
    actual_bg_time_array = old_pred_data.result_actual_bg_time_array
    pred_array = old_pred_data.result_pred_array
    pred_time_array = old_pred_data.result_pred_time_array

    #Root mean squared error
    rms = math.sqrt(mean_squared_error(actual_bg_array, pred_array))
    print "                Root Mean Squared Error: " + str(rms)

    plot, zone = ClarkeErrorGrid.clarke_error_grid(actual_bg_array, pred_array, id_str + " " + algorithm_str)
    print "                Zones are A:{}, B:{}, C:{}, D:{}, E:{}\n".format(zone[0],zone[1],zone[2],zone[3],zone[4])
    if save_clarke_plot: plt.savefig(id_str + algorithm_str.replace(" ", "") + minutes_str + "clarke.png")
    if show_clarke_plot: plot.show()

    plt.clf()
    plt.plot(pred_time_array, pred_array, label="BG Prediction")
    plt.plot(actual_bg_time_array, actual_bg_array, label="Actual BG")
    plt.title(id_str + " " + algorithm_str + " BG Analysis")
    plt.ylabel("Blood Glucose Level (mg/dl)")
    plt.xlabel("Time (minutes)")
    plt.legend(loc='upper left')

    # SHOW/SAVE PLOT DEPENDING ON THE BOOLEAN PARAMETER
    if save_pred_plot: plt.savefig(id_str + algorithm_str.replace(" ","") + minutes_str + "plot.png")
    if show_pred_plot: plt.show()
