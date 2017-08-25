"""
data_matrix.py
Creates the data matrix for the OpenAPS prediction algorithms

Main Functions:      make_data_matrix(bg_df, lomb_data, start_index, end_index, num_data_minutes, num_pred_minutes)

MedicineX OpenAPS
2017-7-24
"""

import numpy as np


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
                actual_bg_array[array_index] = bg_df.iloc[df_index]['openaps']['enacted']['bg']
                array_index += 1
                last_time = time
            except:
                try:
                    actual_bg_array[array_index] = bg_df.iloc[df_index]['openaps']['suggested']['bg']
                    array_index += 1
                    last_time = time
                except:
                    #If a miss, don't move to the next index and instead add one to the number missed
                    miss += 1
        else:
            miss += 1


    #Remove the number of missed data
    time_bg_array = np.resize(time_bg_array, total_len - miss)
    actual_bg_array = np.resize(actual_bg_array, total_len - miss)

    return time_bg_array, actual_bg_array


#Returns true if the data lies in a data gap, so it will not be used
def _in_data_gap(data_gap_start_time, data_gap_end_time, data_curr_time, data_start_time):
    for gap_index in range(len(data_gap_start_time)):
        if (data_curr_time > data_gap_start_time[gap_index] and data_curr_time < data_gap_end_time[gap_index]) or (data_start_time > data_gap_start_time[gap_index] and data_start_time < data_gap_end_time[gap_index]):
            return True

    return False


#This function takes in the data arrays for BG time and lomb_data namedtuple (timeValue, BG, IOB, and COB) and fills in and returns the data training matrix.
#It creates vectors with num_data_minutes of time, BG, IOB, and COB on each row, starting with the current time on the top.
#Each row is a different current time, with the earliest current time on the bottom.
#You can use either training or testing arrays.
def _fill_matrix(time_bg_array, actual_bg_array, lomb_data, data_gap_start_time, data_gap_end_time, num_data_minutes, num_pred_minutes):
    #The total number of r in the data_matrix. It will be the length of the time_bg_array. We will find
    #the times and values of the actual BGs and compare our predictions to these actual values.
    num_data_time_rows = len(time_bg_array)
    num_data_cols = NUM_DATA_ELEMENTS * num_data_minutes
    curr_data_row = 0

    #data_matrix[row,col]
    data_matrix = np.zeros((num_data_time_rows, num_data_cols))
    bg_output = np.zeros(num_data_time_rows)

    for row_index in range(num_data_time_rows):
        #The index of the arrays. Need to find the time of the time_bg_array and then subtract by num_pred_minutes to find the first entry of the data that we will use to make the prediction
        #Needs to be a WHOLE minute, not any decimals, so convert to an integer
        data_curr_time = int(time_bg_array[row_index]) - num_pred_minutes
        data_start_time = data_curr_time - num_data_minutes + 1 #Start time of the data horizon

        if not _in_data_gap(data_gap_start_time, data_gap_end_time, data_curr_time, data_start_time):
            #row_index: Iterate over the data rows, skipping by the NUM_DATA_ELEMENTS rows (e.g. skipping by 4 rows)
            #data_index: Iterate over the entries in the data based on the col_index (eg start at 9 and get ten minutes of data until time is 0)
            #hours from midnight, bg, iob, and cob are the order of the input vectors. This is repeated for a single array depending on how many minutes of input you want
            for col_index, data_index in zip(range(0, num_data_cols, NUM_DATA_ELEMENTS), range(data_curr_time, data_curr_time - num_data_minutes, -1)):
                data_matrix[curr_data_row, col_index] = lomb_data.time_value_array[data_index]
                data_matrix[curr_data_row, col_index + 1] = lomb_data.bg_lomb[data_index]
                data_matrix[curr_data_row, col_index + 2] = lomb_data.iob_lomb[data_index]
                data_matrix[curr_data_row, col_index + 3] = lomb_data.cob_lomb[data_index]

            bg_output[curr_data_row] = actual_bg_array[row_index]

            curr_data_row += 1

    #Resize the matrices and arrays
    data_matrix = np.resize(data_matrix, (curr_data_row, num_data_cols))
    bg_output = np.resize(bg_output, curr_data_row)

    return data_matrix, bg_output


#Function that returns the actual_bg arrays and the train_matrix for both the training and testing sets
def make_data_matrix(bg_df, lomb_data, start_index, end_index, num_data_minutes, num_pred_minutes):
    """
    Function to make the input data matrix for the machine learning algorithms. It
    takes in the dataframe, the lomb-scargle data, the data gap start and stop time arrays,
    the start and end indices, and the number of data and prediction minutes (data and prediction horizons).

    Input:      bg_df                           Pandas dataframe of all of the data from ./data/[id_str]/devicestatus.json
                lomb_data                       The namedtuple holding the lomb-scargle data with these arrays:
                                                    ['period', 'bg_lomb', 'iob_lomb', 'cob_lomb', 'time_value_array', data_gap_start_time, data_gap_end_time]
                start_index                     The start index of the set. Should be higher in value than end_index, as
                                                    the earlier times have higher indices
                end_index                       The end index of the set. Should be lower in value than start_index, as
                                                    the later times have lower indices. Inclusive, so this index is included in the data
                num_data_minutes                The number of data minutes (data horizon). Includes the current data point,
                                                    so 1 will only have the current data value
                num_pred_minutes                The number of minutes in the future the prediction is for (prediction horizon).
                                                    It does not include the the current data point, so 1 means that it is a prediction for 1 minute in the future
.
    Output:     data_matrix                     The input data matrix for the machine learning algorithm
                bg_output                       The output array of bg values that corresponds to the data_matrix
    Usage:      train_data_matrix, actual_bg_train_array = make_data_matrix(bg_df, train_lomb_data, start_train_index, end_train_index, 5, 30)
    """

    #This is the first possible prediction time due to the data and prediction gap. Anything less is not used
    prediction_start_time = num_data_minutes + num_pred_minutes - 1

    time_bg_array, actual_bg_array = _make_actual_bg_array(bg_df, start_index, end_index, prediction_start_time)
    data_matrix, bg_output = _fill_matrix(time_bg_array, actual_bg_array, lomb_data, lomb_data.data_gap_start_time, lomb_data.data_gap_end_time, num_data_minutes, num_pred_minutes)

    return data_matrix, bg_output
