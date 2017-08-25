"""
bgdata.py
This file contains the functions to get the OpenAPS data from the devicestatus.json files.
The data must be in the data folder in another folder with the ID only as the title.
The data must be named devicestatus.json.

Main Functions:     get_bg_dataframe(id_str)
                    get_bg_index(bg_df, start_str, end_str, set_str, make_col_bool)


MedicineX OpenAPS
2017-7-24
"""

import pandas as pd
import numpy as np

#The number of minutes of spacing between each actual data entry in the actual_bg_array.
#Keeping every entry greater than this value prevents overfitting (e.g. entries every 5 minutes work
#well, but entries every minute cause overfitting)
MIN_ENTRY_SPACING_MINUTE = 5



#Function to read in the start and end date according to year-month-day format and convert them to datetimes
def _get_date(start_str, end_str):
    start_date = pd.to_datetime(start_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_str, format='%Y-%m-%d')

    return start_date, end_date


#Function to find the start and stop indices for both the train and test dates given the dataframe, the start and end datetimes, and the boolean make_col_bool
#The start_index is a higher number than the end_index because the higher indices are the earlier dates while the lower indices are the later dates
def _find_index(bg_df, start_date, end_date, make_col_bool):
    if (make_col_bool): bg_df['date'] = bg_df['created_at'].apply(lambda x: x.date()) #create column with just the date if make_col_bool is True

    #Find the first date with the start date (first entry) and the last date with the end date (last entry)
    #Since the older dates have higher indices, we use max() for start and min() for the end dates
    start_index = bg_df[bg_df['date'] == start_date.date()].index.max()
    end_index = bg_df[bg_df['date'] == end_date.date()].index.min()

    #Raises exception if invalid dates (which are labeled as NaN)
    if np.isnan(start_index): raise Exception("Invalid start date: " + str(start_date.date()))
    if np.isnan(end_index): raise Exception("Invalid end date: " + str(end_date.date()))

    return bg_df, start_index, end_index


#Function to get the bg data
def get_bg_dataframe(id_str):
    """
    Function to convert the json file to a pandas dataframe.
    It takes in the string of the id and looks for the devicestatus.json file.
    All data should be stored such that in the directory where main.py lies,
    there is a directory called "data". Inside this directory,
    there is another directory with just the ID Number. Inside this data folder lies the
    devicestatus.json file, which contains the data. If the file is not in the path given,
    it raises an IOError. The path should look like the following example:

    ./data/12345678/devicestatus.json

    Input:      id_str                          ID number as a string
    Output:     bg_df                           Pandas dataframe of all of the data from ./data/[id_str]/devicestatus.json
    Usage:      bg_df = get_bg_dataframe("12345678")
    """

    try:
        file_location = "./data/" + id_str + "/devicestatus.json"
        bg_df = pd.read_json(file_location) #Opens the data file and reads in the data into a dataFrame
    except:
        raise IOError(file_location + " is not a valid file.")

    print
    print("{} total entries.".format(len(bg_df)))

    return bg_df


#Function to find the indices for the given start and end date strings
def get_bg_index(bg_df, start_str, end_str, set_str, make_col_bool):
    """
    Function to find the indices of the start and end dates in the dataframe
    given the string version of the start and end dates. It then prints the total entries
    and indices for the given set (training, validation, or testing set). It then returns
    the dataframe and the start and end indices.

    Input:      bg_df                           Pandas dataframe of all of the data from ./data/[id_str]/devicestatus.json
                start_str                       String of the start date in year-month-day form (e.g. "2017-12-25")
                end_str                         String of the end date in year-month-day form (e.g. "2017-12-25")
                set_str                         String of the set name (i.e. "Training", "Validation", "Testing")
                make_col_bool                   Boolean that tells allows the function to add the 'date' column,
                                                    which simply holds the date of the entry. This should be True the very first
                                                    time this function is called and False otherwise
    Output:     bg_df                           Pandas dataframe of all of the data from ./data/[id_str]/devicestatus.json
    Usage:      bg_df, start_train_index, end_train_index = get_bg_index(bg_df, "2017-05-05", "2017-05-14", "Training", True)
    """

    start_date, end_date = _get_date(start_str, end_str)
    bg_df, start_index, end_index = _find_index(bg_df, start_date, end_date, make_col_bool)

    #Print the number of entries, the start index, and the end index
    print("{}: {} total entries from {} to {}".format(set_str, start_index - end_index + 1, start_date, end_date))
    print("{} Start Index = {} and {} End Index = {}".format(set_str, start_index, set_str, end_index))
    print

    return bg_df, start_index, end_index


#Makes new dataframe spaced out by 5 minute entries
def get_new_df_entries_every_5_minutes(bg_df, start_index, end_index, set_str):
    new_bg_df = pd.DataFrame()
    last_time = - MIN_ENTRY_SPACING_MINUTE
    starting_df = True

    for df_index in range(end_index, start_index):
        add_entry = False
        try:
            time = (bg_df.iloc[df_index]['created_at'] - bg_df.iloc[start_index]['created_at']) / np.timedelta64(1, 'm')
            test_if_has_enacted = bg_df.iloc[df_index]['openaps']['enacted']['bg'] #Test to see if df entry has suggested and enacted functioning
            test_if_has_suggested = bg_df.iloc[df_index]['openaps']['suggested']['bg']

            if last_time - time >= MIN_ENTRY_SPACING_MINUTE or starting_df: #check if spaced out by 5 minute or just starting the df
                starting_df = False
                last_time = time

                #If it has both enacted and suggested and is spaced out by MIN_ENTRY_SPACING_MINUTE from last entry, then set boolean to be true
                add_entry = True
        except:
            add_entry = False

        if add_entry:
            new_bg_df = new_bg_df.append(bg_df.iloc[df_index], ignore_index=True) #if boolean is true, add it to the new dataframe

    start_index = len(new_bg_df) - 1
    end_index = 0

    #Print the number of entries, the start index, and the end index
    print("{}: {} number entries".format(set_str, start_index - end_index + 1))
    print("{} Start Index = {} and End Index = {}".format(set_str, start_index, end_index))
    print

    return new_bg_df, start_index, end_index
