'''
bgdataframe.py
This file contains the functions to get the OpenAPS data from the devicestatus.json files.

Main Function:
        get_bg_data(id_str, start_train_str, end_train_str, start_valid_str, end_valid_str, start_test_str, end_test_str)

Input:
        id_str                          ID number as a string
        start_train_str                 Start train date (%Y-%m-%d) as a string or datetime
        end_train_str                   End train date (%Y-%m-%d) as a string or datetime
        start_valid_str                 Start validation date (%Y-%m-%d) as a string or datetime
        end_valid_str                   End validation date (%Y-%m-%d) as a string or datetime
        start_test_str                  Start test date (%Y-%m-%d) as a string or datetime
        end_test_str                    End test date (%Y-%m-%d) as a string or datetime

Output:
        bg_df                           Pandas dataframe of all of the data from /data/[id_str]/devicestatus.json
        start_train_index               Index of the start of training
        end_train_index                 Index of the end of training (inclusive)
        start_valid_index               Index of the start of validation
        end_valid_index                 Index of the end of validation (inclusive)
        start_test_index                Index of the start of testing
        end_test_index                  Index of the end of testing (inclusive)



USAGE:
        bg_df, start_train_index, end_train_index, start_valid_index, end_valid_index, start_test_index, end_test_index = get_bg_data(id_str, start_train_str, end_train_str,
                                                                                                                                        start_valid_str, end_valid_str, start_test_str, end_test_str)

Trevor Tsue
2017-7-24
'''

import pandas as pd
import numpy as np


#Function to convert the json file to a dataFrame
def _get_file(id_str):
    try:
        file_location = "data/" + id_str + "/devicestatus.json"
        bg_df = pd.read_json(file_location) #Opens the data file and reads in the data into a dataFrame
    except:
        raise IOError(file_location + " is not a valid file.")
    return bg_df


#Function to read in the start and end date according to year-month-day format
def _get_date(start_str, end_str):
    start_date = pd.to_datetime(start_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_str, format='%Y-%m-%d')

    return start_date, end_date


#Function to find the start and stop indices for both the train and test dates
#The start_index is a higher number than the end_index because the higher indices are the earlier dates while the lower indices are the later dates
def _find_index(bg_df, start_date, end_date, make_col):
    if (make_col): bg_df['date'] = bg_df['created_at'].apply(lambda x: x.date()) #create column with just the date only once

    start_index = bg_df[bg_df['date'] == start_date.date()].index.max()
    end_index = bg_df[bg_df['date'] == end_date.date()].index.min()

    #Raises exception if invalid dates
    if np.isnan(start_index): raise Exception("Invalid start date: " + str(start_date.date()))
    if np.isnan(end_index): raise Exception("Invalid end date: " + str(end_date.date()))

    return start_index, end_index


#Function to get the bg data
def get_bg_data(id_str, start_train_str, end_train_str, start_valid_str, end_valid_str, start_test_str, end_test_str):
    bg_df = _get_file(id_str)

    start_train_date, end_train_date = _get_date(start_train_str, end_train_str)
    start_valid_date, end_valid_date = _get_date(start_valid_str, end_valid_str)
    start_test_date, end_test_date = _get_date(start_test_str, end_test_str)

    start_train_index, end_train_index = _find_index(bg_df, start_train_date, end_train_date, True)
    start_valid_index, end_valid_index = _find_index(bg_df, start_valid_date, end_valid_date, False)
    start_test_index, end_test_index = _find_index(bg_df, start_test_date, end_test_date, False)

    print
    print("{} total entries.".format(len(bg_df)))
    print("Training: {} total entries from {} to {}".format(start_train_index - end_train_index + 1, start_train_date, end_train_date))
    print("Validation: {} total entries from {} to {}".format(start_valid_index - end_valid_index + 1, start_valid_date, end_valid_date))
    print("Testing: {} total entries from {} to {}".format(start_test_index - end_test_index + 1, start_test_date, end_test_date))
    print
    print("Training Start Index = {} and Training End Index = {}".format(start_train_index, end_train_index))
    print("Validation Start Index = {} and Validation End Index = {}".format(start_valid_index, end_valid_index))
    print("Testing Start Index = {} and Testing End Index = {}".format(start_test_index, end_test_index))
    print

    return bg_df, start_train_index, end_train_index, start_valid_index, end_valid_index, start_test_index, end_test_index
