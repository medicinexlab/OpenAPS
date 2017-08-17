'''
oldpred.py
This file contains the functions to analyze the old OpenAPS prediction algorithms from the devicestatus.json files.
The data must be in the data folder in another folder with the ID only as the title.
The data must be named devicestatus.json

Main Function:
         analyze_old_pred_data(bg_df, old_pred_algorithm_array, start_test_index, end_test_index, pred_minutes, show_pred_plot, save_pred_plot, show_clarke_plot, save_clarke_plot, id_str)

MedicineX OpenAPS
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


#Function to get the eventualBG and actual BG. Looks at the enacted directory before the suggested directory.
#If there is no data, then it increases the miss count by 1.
def _get_other_bg(bg_df, pred_array, pred_time_array, curr, miss, start_index, data_index, bg_str):
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
#Looks at enacted directory first before looking at suggested directory.
#If there is no data, then it increases the miss count by 1.
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
#directly from the dataframe given the dataframe, the start_index, end_index, and the index of the pred array (eg 30 min prediction has 6 as the index of pred array)
def _get_raw_pred_array(bg_df, start_index, end_index, pred_array_index):
    total_len = start_index - end_index + 1

    actual_bg_array, actual_bg_time_array, actual_curr, actual_miss = _new_pred_array(start_index, end_index, total_len)
    eventual_pred_array, eventual_pred_time_array, eventual_curr, eventual_miss = _new_pred_array(start_index, end_index, total_len)
    iob_pred_array, iob_pred_time_array, iob_curr, iob_miss = _new_pred_array(start_index, end_index, total_len)
    cob_pred_array, cob_pred_time_array, cob_curr, cob_miss = _new_pred_array(start_index, end_index, total_len)
    acob_pred_array, acob_pred_time_array, acob_curr, acob_miss = _new_pred_array(start_index, end_index, total_len)

    for data_index in range(start_index, end_index - 1, -1):
        actual_bg_array, actual_bg_time_array, actual_curr, actual_miss= _get_other_bg(bg_df, actual_bg_array, actual_bg_time_array, actual_curr, actual_miss, start_index, data_index, 'bg')

        eventual_pred_array, eventual_pred_time_array, eventual_curr, eventual_miss = _get_other_bg(bg_df, eventual_pred_array, eventual_pred_time_array,
                                                                                                        eventual_curr, eventual_miss, start_index, data_index, 'eventualBG')
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


#Finds the nearest value in time to the given values in the given array.
#The nearest value in time must be in within the ACTUAL_BG_RANGE (for example, if ACTUAL_BG_RANGE = 5
#and the time is 30, then the nearest index must lie from 25 < x < 35 or else -1 is returned)
#If there is no nearest index at all, 01 is returned
def _find_nearest_index(array, value):
    nearest_index = (np.abs(array-value)).argmin() #finds the index of the time value closest to the input value
    if (int(np.abs(array[nearest_index] - value)) < ACTUAL_BG_RANGE):
        #If inside the ACTUAL_BG_RANGE, then return the nearest index
        return nearest_index
    else:
        return -1


#Given the actual_bg_array, actual_bg_time_array, pred_array, pred_time_array, and num_pred_minutes,
#this function finds the nearest actual bg value to compare to the prediction value.
#If there is one, then it adds all the values to the result arrays, which are returned as a namedtuple.
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
        nearest_index = _find_nearest_index(actual_bg_time_array, future_time)

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


#This function takes in the bg dataframe, the start and end indices, and the number of minutes in the future
#that you want to make a prediction for AKA prediction horizon (e.g. make a prediction for 30 minutes in the future).
#It returns the namedtuple with the following attributes ['result_actual_bg_array', 'result_actual_bg_time_array', 'result_pred_array', 'result_pred_time_array']
def _get_old_pred(bg_df, start_index, end_index, num_pred_minutes):
    #The number of 5 minute sections until the prediction (e.g. 30 minutes = 6 sections)
    pred_array_index = num_pred_minutes / DATA_SPACING

    actual_bg_array, actual_bg_time_array, eventual_pred_array, eventual_pred_time_array, iob_pred_array, iob_pred_time_array, cob_pred_array, cob_pred_time_array, acob_pred_array, acob_pred_time_array = _get_raw_pred_array(bg_df, start_index, end_index, pred_array_index)

    eventual_pred_data = _find_compare_array(actual_bg_array, actual_bg_time_array, eventual_pred_array, eventual_pred_time_array, 30)
    iob_pred_data = _find_compare_array(actual_bg_array, actual_bg_time_array, iob_pred_array, iob_pred_time_array, num_pred_minutes)
    cob_pred_data= _find_compare_array(actual_bg_array, actual_bg_time_array, cob_pred_array, cob_pred_time_array, num_pred_minutes)
    acob_pred_data = _find_compare_array(actual_bg_array, actual_bg_time_array, acob_pred_array, acob_pred_time_array, num_pred_minutes)

    return eventual_pred_data, iob_pred_data, cob_pred_data, acob_pred_data


#Plots old pred data given namedtuple of old data (eventualBG, acob, cob, or iob).
#Can show or save prediction plot based on show_pred_plot or save_pred_plot, respectively.
#Same goes for the Clarke Error grid with show_clarke_plot or save_clarke_plot, respectively.
#id_str, algorithm_str, minutes_str are strings of the ID, the prediction algorithm and the number of prediction minutes used for the title.
def _plot_old_pred_data(old_pred_data, show_pred_plot, save_pred_plot, show_clarke_plot, save_clarke_plot, id_str, algorithm_str, minutes_str):
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
    plt.plot(actual_bg_time_array, actual_bg_array, label="Actual BG")
    plt.plot(pred_time_array, pred_array, label="BG Prediction")
    plt.title(id_str + " " + algorithm_str + " BG Analysis")
    plt.ylabel("Blood Glucose Level (mg/dl)")
    plt.xlabel("Time (minutes)")
    plt.legend(loc='upper left')

    # SHOW/SAVE PLOT DEPENDING ON THE BOOLEAN PARAMETER
    if save_pred_plot: plt.savefig(id_str + algorithm_str.replace(" ","") + minutes_str + "plot.png")
    if show_pred_plot: plt.show()


#Function to analyze the old OpenAPS data
def analyze_old_pred_data(bg_df, old_pred_algorithm_array, start_test_index, end_test_index, pred_minutes, show_pred_plot, save_pred_plot, show_clarke_plot, save_clarke_plot, id_str):
    """
    Function that analyzes the old OpenAPS prediction models (eventualBG, aCOB, COB, and IOB)
    based on what is put in the old_pred_algorithm_array. If it is empty, nothing will be plotted.
    Since all the algorithms are calculated every 5 minutes, pred_minutes must be a multiple of 5.
    eventualBG is only calculated by 30 minutes, so it will always be 30 minutes.
    It will save the prediction plot if save_pred_plot is True and the clarke plot if save_clarke_plot is True.
    It will show the prediction plot if show_pred_plot is true and the clarke plot if show_clarke_plot is True.

    Input:      bg_df                           Pandas dataframe of all of the data from ./data/[id_str]/devicestatus.json
                old_pred_algorithm_array        The array of the original OpenAPS prediction algorithms that you want to receive
                                                data from. It can contain any/none of the following: "eventualBG", "acob", "cob", "iob"
                start_test_index                The starting index of the testing data
                end_test_index                  The ending index of the testing data
                pred_minutes                    The number of minutes in the future the prediction is for (predicion horizon). Must be a multiple of 5
                show_pred_plot                  Boolean to show the prediction plot
                save_pred_plot                  Boolean to save the prediction plot
                show_clarke_plot                Boolean to show the Clarke Error Grid Plot
                save_clarke_plot                Boolean to save the Clarke Error Grid Plot
                id_str                          The ID of the person as a string. Used for the title.

    Output:     None
    Usage:      analyze_old_pred_data(bg_df, ['iob', 'cob'], 1500, 0, 30, True, False, True, False, "00000001")
    """

    if pred_minutes % 5 != 0: raise Exception("The prediction minutes is not a multiple of 5.")
    eventual_pred_data, iob_pred_data, cob_pred_data, acob_pred_data = _get_old_pred(bg_df, start_test_index, end_test_index, pred_minutes)
    if 'eventualBG' in old_pred_algorithm_array:
        print("        eventualBG")
        _plot_old_pred_data(eventual_pred_data, show_pred_plot, save_pred_plot, show_clarke_plot, save_clarke_plot, id_str, "eventualBG", "Pred" + str(30))
    if 'iob' in old_pred_algorithm_array:
        print("        iob")
        _plot_old_pred_data(iob_pred_data, show_pred_plot, save_pred_plot, show_clarke_plot, save_clarke_plot, id_str, "IOB", "Pred" + str(pred_minutes))
    if 'cob' in old_pred_algorithm_array:
        print("        cob")
        _plot_old_pred_data(cob_pred_data, show_pred_plot, save_pred_plot, show_clarke_plot, save_clarke_plot, id_str, "COB", "Pred" + str(pred_minutes))
    if 'acob' in old_pred_algorithm_array:
        print("        acob")
        _plot_old_pred_data(acob_pred_data, show_pred_plot, save_pred_plot, show_clarke_plot, save_clarke_plot, id_str, "aCOB", "Pred" + str(pred_minutes))
