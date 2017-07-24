'''
data_matrix.py
Creates the data matrix for the OpenAPS prediction algorithms

Main Function:
        make_data_matrix(bg_df, lomb_data, start_index, end_index, num_data_minutes, num_pred_minutes)

Input:
        bg_df                           The pandas dataframe containing all of the data.
        lomb_data                       The namedtuple containg all of the lomb-scargle data.
        start_index                     The start index of the data in the dataframe.
        end_index                       The end index of the data in the dataframe.
        num_data_minutes                The number of minutes of data given for each prediction.
        num_pred_minutes                The number of minutes in the future the prediction is for.

Output:
        data_matrix                     This is the numpy matrix with each row containing the data elements for each prediction.
        actual_bg_array                 This is the array of actual bg values. Can be used for both training and testing.

USAGE:
    train_data_matrix, actual_bg_train_array = make_data_matrix(bg_df, train_lomb_data, start_train_index, end_train_index, data_minutes, pred_minutes)
    test_data_matrix, actual_bg_test_array = make_data_matrix(bg_df, test_lomb_data, start_test_index, end_test_index, data_minutes, pred_minutes)
Trevor Tsue
2017
'''

import pandas as pd
import numpy as np
from collections import namedtuple


#The number of elements for each prediction time section (time, BG, IOB, COB)
NUM_DATA_ELEMENTS = 4



#This functions finds all of the actual valid BG levels. It also keeps track of the minutes they occur in time_bg_array.
#Makes sure that all actual BGs are past the prediction_start_time, which takes into account the data and pred gap
def _make_actual_bg_array(bg_df, start_index, end_index, prediction_start_time):
    total_len = start_index - end_index + 1
    time_bg_array = np.zeros(total_len)
    actual_bg_array = np.zeros(total_len)
    array_index = miss = 0

    for df_index in range(start_index, end_index - 1, -1):
        #Keep track of the time starting at 0 at the start_index
        time = (bg_df.iloc[df_index]['created_at'] - bg_df.iloc[start_index]['created_at']) / np.timedelta64(1, 'm')

        if time > prediction_start_time:
            time_bg_array[array_index] = time
            try:
                actual_bg_array[array_index] = bg_df.iloc[df_index]['openaps']['suggested']['bg']
                array_index += 1
            except:
                try:
                    actual_bg_array[array_index] = bg_df.iloc[df_index]['openaps']['enacted']['bg']
                    array_index += 1
                except:
                    try:
                        actual_bg_array[array_index] = bg_df.iloc[df_index]['loop']['predicted']['values'][0]
                        array_index += 1
                    except:
                        #If a miss, don't move to the next index and instead add one to the number missed
                        miss += 1
        else:
            miss += 1

    #Remove the number of missed data
    time_bg_array = np.resize(time_bg_array, total_len - miss)
    actual_bg_array = np.resize(actual_bg_array, total_len - miss)

    return time_bg_array, actual_bg_array


#This function takes in the data arrays for BG time, actual BG, and lomb_data (timeValue, BG, IOB, and COB) and fills in and returns the data training matrix.
#It creates vectors with NUM_DATA_MINUTES of time, BG, IOB, and COB on each row, starting with the current time on the top.
#Each row is a different current time, with the earliest current time on the bottom.
#You can use either training or testing arrays.
def _fill_matrix(time_bg_array, actual_bg_array, lomb_data, num_data_minutes, num_pred_minutes):
    #The total number of r in the data_matrix. It will be the length of the actual_bg_array. We will find
    #the times and values of the actual BGs and compare our predictions to these actual values.
    num_data_time_rows = len(actual_bg_array)
    num_data_cols = NUM_DATA_ELEMENTS * num_data_minutes

    #data_matrix[row,col]
    data_matrix = np.zeros((num_data_time_rows, num_data_cols))

    for row_index in range(num_data_time_rows):
        #The index of the arrays. Need to find the time of the actual_bg_array and then subtract by num_pred_minutes to find the first entry of the data that we will use to make the prediction
        #Needs to be a WHOLE minute, not any decimals, so convert to an integer
        overall_data_index = int(time_bg_array[row_index]) - num_pred_minutes

        #row_index: Iterate over the data rows, skipping by the NUM_DATA_ELEMENTS rows (e.g. skipping by 4 rows)
        #data_index: Iterate over the entries in the data based on the col_index (eg start at 9 and get ten minutes of data until time is 0)
        for col_index, data_index in zip(range(0, num_data_cols, NUM_DATA_ELEMENTS), range(overall_data_index, overall_data_index - num_data_minutes, -1)):
            data_matrix[row_index, col_index] = lomb_data.time_value_array[data_index]
            data_matrix[row_index, col_index + 1] = lomb_data.bg_lomb[data_index]
            data_matrix[row_index, col_index + 2] = lomb_data.iob_lomb[data_index]
            data_matrix[row_index, col_index + 3] = lomb_data.cob_lomb[data_index]

    return data_matrix


#Function that returns the actual_bg arrays and the train_matrix for both the training and testing sets
def make_data_matrix(bg_df, lomb_data, start_index, end_index, num_data_minutes, num_pred_minutes):
    #This is the first possible prediction time due to the data and prediction gap. Anything less is not used
    prediction_start_time = num_data_minutes + num_pred_minutes - 1

    time_bg_array, actual_bg_array = _make_actual_bg_array(bg_df, start_index, end_index, prediction_start_time)
    data_matrix = _fill_matrix(time_bg_array, actual_bg_array, lomb_data, num_data_minutes, num_pred_minutes)

    return data_matrix, actual_bg_array
