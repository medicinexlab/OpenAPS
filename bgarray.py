'''
bgarray.py
This file contains the functions to take the start and end indices of training and testing and returns the
lomb-scargle model of the data.

Main Function:
        get_bg_array(bg_df, start_train_index, end_train_index, start_test_index, end_test_index, plot_lomb_array)

Input:
        bg_df                           Pandas dataframe of all of the data from /data/[id_string]/devicestatus.json
        start_train_index               Index of the start of training
        end_train_index                 Index of the end of testing (inclusive)
        start_test_index                Index of the start of testing
        end_test_index                  Index of the end of testing (inclusive)
        plot_lomb_array                 This is the array/list that will plot the lomb scargle data
                                        if they have 0, 1, or 2 (0:BG, 1:IOB, and 2:COB)
                                        Keep this list empty if you don't want to plot anything

Output:
        train_lomb_data                 This namedtuple struct contains the period, bg_lomb, iob_lomb, cob_lomb,
                                        and time_value_array for the training data
        test_lomb_data                  This namedtuple struct contains the period, bg_lomb, iob_lomb, cob_lomb,
                                        and time_value_array for the testing data

USAGE:
    train_lomb_data, test_lomb_data = get_bg_array(bg_df, start_train_index, end_train_index, start_test_index, end_test_index, PLOT_LOMB_ARRAY)

Trevor Tsue
2017-7-24
'''

import numpy as np
import gatspy
import matplotlib.pyplot as plt
from collections import namedtuple


#Lomb-Scargle Periodogram Global Variables:

#This is the maximum distance between data times that is allowed. If the gap in data is greater than this threshold, then it adds data in between to make the lomb-scargle more accurate
MAX_TIME_GAP = 5
#Extra space to add to arrays to make sure all entries are entered
EXTRA_SPACE = 10000
#These two define the period range for the lomb-scargle. It will search for the period from max - PERIOD_RANGE_MAX to max - PERIOD_RANGE_MIN
PERIOD_RANGE_MIN = 5
PERIOD_RANGE_MAX = 100
#The first pass coverage number for the lomb scargle
FIRST_PASS_COVERAGE_NUM = 25
#The number of fourier series used on the BG, IOB, and COB Lomb-Scargles
BG_NUM_FOURIER_SERIES = 150
IOB_NUM_FOURIER_SERIES = 125
COB_NUM_FOURIER_SERIES = 100



#This function takes in a Timestamp and applies the time function to transform the timestamp from the 24-hour clock to
#the time in minutes in a day as defined from 0 to 1,439
def _timestamp_function(time):
    minutes_in_hour = 60
    return (time.hour * minutes_in_hour) + time.minute


#Returns the the time_value_array (the array of the exact minute of the day for each entry that will be used for the vectors)
def _make_time_value_array(bg_df, start_index, end_index):
    min_in_day = 1440
    array_len = int((bg_df.iloc[end_index]['created_at'] - bg_df.iloc[start_index]['created_at']) / np.timedelta64(1, 'm')) + 1
    time_value_array = np.zeros(array_len)

    #First entry is the minute of the start_index
    time_value_array[0] = _timestamp_function(bg_df.iloc[start_index]['created_at'])
    for index in range(array_len - 1):
        #Since everything is spaced out by minutes, everything entry is a minute later than the previous
        time_value_array[index + 1] = (time_value_array[index] + 1) % min_in_day

    return time_value_array


#Function that adds data to fill in the gaps of the original data before the lomb-scargle is applied. IT helps make lomb-scargle more accurate
#Essentially, if there is a gap bigger than the size of the MAX_TIME_GAP, then this recursive function will add a data point in between the two time points, creating two more gaps.
#It will recursively call on both of these gaps until the gap size is less than or equal to the MAX_TIME_GAP
#To add data, this function takes the mean of the old and new time, and it sets the value at this middle time to be the mean of the values between the old and new time.
#It will update the array accordingly to make sure the time points are still in order and the indices are correct
def _fill_data_gaps(old_time, new_time, old_value, new_value, time_array, value_array, index):
    if new_time - old_time <= MAX_TIME_GAP:
        time_array[index] = new_time
        value_array[index] = new_value
        return index + 1

    mid_time = (new_time + old_time) / 2
    mid_value = (new_value + old_value) / 2

    index = _fill_data_gaps(old_time, mid_time, old_value, mid_value, time_array, value_array, index)
    index = _fill_data_gaps(mid_time, new_time, mid_value, new_value, time_array, value_array, index)

    return index


#Helper for the _make_data_array. It allows you to choose which Column One and Column Two Names you want
def _make_data_array_helper(bg_df, time_array, value_array, start_index, index, curr, last, num_extra_added, col_one_name, col_two_name, item_string):
    new_time = (bg_df.iloc[index]['created_at'] - bg_df.iloc[start_index]['created_at']) / np.timedelta64(1, 'm')

    new_value = bg_df.iloc[index][col_one_name][col_two_name][item_string]
    old_time = time_array[last]
    old_value = value_array[last]

    #keep track of the curr value before passing into _fill_data_gaps
    start_curr = curr
    curr = _fill_data_gaps(old_time, new_time, old_value, new_value, time_array, value_array, curr)

    #Find the number of extra entries num_extra_added
    num_extra_added += curr - start_curr - 1
    last = curr - 1

    return time_array, value_array, curr, last, num_extra_added


#Function to make the data array for lomb-scargle given the bg_df dataframe, the start_index, the end_index, and the item_string, which is the column that you want to get
#Can put any start and end index as a parameter
def _make_data_array(bg_df, start_index, end_index, item_string):
        total_len = start_index - end_index + EXTRA_SPACE + 1
        time_array = np.zeros(total_len)
        value_array = np.zeros(total_len)
        curr = last = num_extra_added = miss = 0

        for index in range(start_index, end_index - 1, -1):
            try:
                time_array, value_array, curr, last, num_extra_added = _make_data_array_helper(bg_df, time_array, value_array, start_index, index, curr, last, num_extra_added, 'openaps', 'suggested', item_string)
            except:

                try:
                    time_array, value_array, curr, last, num_extra_added = _make_data_array_helper(bg_df, time_array, value_array, start_index, index, curr, last, num_extra_added, 'openaps', 'enacted', item_string)
                except:
                    #count the number of misses
                    miss += 1

        #Check to see if the number added exceeds EXTRA_SPACE. If it does, raise exception and change it in the global variables above
        if num_extra_added > EXTRA_SPACE:
            raise Exception("The number of data added to fill in data gaps exceeds array size. Add more to EXTRA_SPACE.")

        #Resize the array to fit the data
        time_array = np.resize(time_array, total_len - miss + num_extra_added - EXTRA_SPACE)
        value_array = np.resize(value_array, total_len - miss + num_extra_added - EXTRA_SPACE)

        return time_array, value_array


#This function runs the lomb scargle given the time_array, the value_array, the period, and the number of fourier terms
def _run_lomb_scargle(time_array, value_array, period, num_fourier_terms):
    lomb = gatspy.periodic.LombScargle(Nterms=num_fourier_terms, fit_period=True, optimizer_kwds={'quiet':True})
    lomb.optimizer.set(period_range=(int(time_array.max()) - PERIOD_RANGE_MAX, int(time_array.max()) - PERIOD_RANGE_MIN), first_pass_coverage=FIRST_PASS_COVERAGE_NUM)
    lomb.fit(time_array, value_array)

    return lomb.predict(period)


#Plot lomb scargle function
def _plot_lomb(period, lomb, time_array, value_array, name_string):
    plt.plot(period, lomb, label='Lomb '+name_string)
    plt.plot(time_array, value_array, label='Actual '+name_string)


#This function gets the data from the lomb scargle. It takes in the start and end indices and returns a lomb scargle model for BG, IOB, and COB as well as the period
def _get_lomb_scargle(bg_df, start_index, end_index, plot_lomb_array):
        bg_time_array, bg_value_array = _make_data_array(bg_df, start_index, end_index, 'bg')
        iob_time_array, iob_value_array = _make_data_array(bg_df, start_index, end_index, 'IOB')
        cob_time_array, cob_value_array = _make_data_array(bg_df, start_index, end_index, 'COB')

        period = np.linspace(0, int(bg_time_array.max()), int(bg_time_array.max()) + 1) #set period to be as large as possible

        bg_lomb = _run_lomb_scargle(bg_time_array, bg_value_array, period, BG_NUM_FOURIER_SERIES)
        iob_lomb = _run_lomb_scargle(iob_time_array, iob_value_array, period, IOB_NUM_FOURIER_SERIES)
        cob_lomb = _run_lomb_scargle(cob_time_array, cob_value_array, period, COB_NUM_FOURIER_SERIES)

        #Set all values below zero equal to zero (For some reason, IOB is negative in the actual data, so I did not change those values to zero)
        bg_lomb[bg_lomb < 0] = 0
        cob_lomb[cob_lomb < 0] = 0

        if len(plot_lomb_array) > 0:
            plt.clf()
            if "bg" in plot_lomb_array: _plot_lomb(period, bg_lomb, bg_time_array, bg_value_array, "BG")
            if "iob" in plot_lomb_array: _plot_lomb(period, iob_lomb, iob_time_array, iob_value_array, "IOB")
            if "cob" in plot_lomb_array: _plot_lomb(period, cob_lomb, cob_time_array, cob_value_array, "COB")
            plt.legend(loc='upper left')
            plt.show()

        return period, bg_lomb, iob_lomb, cob_lomb


#The main function to be called to get the bg data arrays
#It applies the lomb scargle periodogram to make a model of the data, and returns this model as an array
def get_bg_array(bg_df, start_train_index, end_train_index, start_test_index, end_test_index, plot_lomb_array):
    time_value_train_array = _make_time_value_array(bg_df, start_train_index, end_train_index)
    time_value_test_array = _make_time_value_array(bg_df, start_test_index, end_test_index)

    train_period, bg_train_lomb, iob_train_lomb, cob_train_lomb = _get_lomb_scargle(bg_df, start_train_index, end_train_index, plot_lomb_array)
    test_period, bg_test_lomb, iob_test_lomb, cob_test_lomb = _get_lomb_scargle(bg_df, start_test_index, end_test_index, plot_lomb_array)

    LombData = namedtuple('LombData', ['period', 'bg_lomb', 'iob_lomb', 'cob_lomb', 'time_value_array'])
    train_lomb_data = LombData(train_period, bg_train_lomb, iob_train_lomb, cob_train_lomb, time_value_train_array)
    test_lomb_data = LombData(test_period, bg_test_lomb, iob_test_lomb, cob_test_lomb, time_value_test_array)

    return train_lomb_data, test_lomb_data
