"""
bglomb.py
This file contains the functions to take the start and end indices of training and testing and returns the
lomb-scargle model of the data.

Main Functions:     get_lomb_data(bg_df, start_index, end_index, plot_lomb_array)

MedicineX OpenAPS
2017-7-24
"""

import numpy as np
import gatspy
import matplotlib.pyplot as plt


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
#This is the maximum data gap. Anything else is removed from the data
MAX_DATA_GAP_MINUTES = 120



#This function takes in the current hour (ranging from 0-23) and returns the number
#of hours from midnight (e.g. 1:00 and 23:00 are both 1 hour from midnight 0:00)
def _get_hours_from_midnight(curr_hour):
    return abs(2*(curr_hour % 12) - curr_hour)


#Returns the the time_value_array (the array of the exact minute of the day for each entry that will be used for the vectors)
def _make_time_value_array(bg_df, start_index, end_index):
    min_in_hour = 60
    hour_in_day = 24
    array_len = int((bg_df.iloc[end_index]['created_at'] - bg_df.iloc[start_index]['created_at']) / np.timedelta64(1, 'm')) + 1
    time_value_array = np.zeros(array_len)

    curr_minute = bg_df.iloc[start_index]['created_at'].minute
    curr_hour = bg_df.iloc[start_index]['created_at'].hour

    for array_index in range(array_len):
        time_value_array[array_index] = _get_hours_from_midnight(curr_hour)
        curr_minute += 1

        if curr_minute >= min_in_hour:
            curr_minute = curr_minute % min_in_hour
            curr_hour = (curr_hour + 1) % hour_in_day

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

    mid_time = int((new_time + old_time) / 2)
    mid_value = (new_value + old_value) / 2

    index = _fill_data_gaps(old_time, mid_time, old_value, mid_value, time_array, value_array, index)
    index = _fill_data_gaps(mid_time, new_time, mid_value, new_value, time_array, value_array, index)

    return index


#Helper for the _make_data_array. It allows you to choose which Column One and Column Two Names you want
def _make_data_array_helper(bg_df, time_array, value_array, data_gap_start_time, data_gap_end_time, start_index, index, curr, last, num_extra_added, col_one_name, col_two_name, item_str):
    new_time = int((bg_df.iloc[index]['created_at'] - bg_df.iloc[start_index]['created_at']) / np.timedelta64(1, 'm'))
    new_value = bg_df.iloc[index][col_one_name][col_two_name][item_str]
    old_time = time_array[last]
    old_value = value_array[last]

    #If it is a data gap, store the start and stop time for later removal
    if new_time - old_time > MAX_DATA_GAP_MINUTES:
        data_gap_start_time.append(old_time)
        data_gap_end_time.append(new_time)

    #keep track of the curr value before passing into _fill_data_gaps
    start_curr = curr
    curr = _fill_data_gaps(old_time, new_time, old_value, new_value, time_array, value_array, curr)

    #Find the number of extra entries num_extra_added
    num_extra_added += curr - start_curr - 1
    last = curr - 1

    return time_array, value_array, data_gap_start_time, data_gap_end_time, curr, last, num_extra_added


#Function to make the data array for lomb-scargle given the bg_df dataframe, the start_index, the end_index, and the item_str, which is the column that you want to get
#Can put any start and end index as a parameter
def _make_data_array(bg_df, start_index, end_index, item_str):
        total_len = start_index - end_index + EXTRA_SPACE + 1
        time_array = np.zeros(total_len)
        value_array = np.zeros(total_len)
        data_gap_start_time = []
        data_gap_end_time = []
        curr, last, num_extra_added, miss = (0 for i in range(4))

        for index in range(start_index, end_index - 1, -1):
            try: #look thorugh enacted
                time_array, value_array, data_gap_start_time, data_gap_end_time, curr, last, num_extra_added = _make_data_array_helper(bg_df, time_array, value_array, data_gap_start_time, data_gap_end_time,
                                                                                                                                        start_index, index, curr, last, num_extra_added, 'openaps', 'enacted', item_str)
            except:
                #look through succested
                try:
                    time_array, value_array, data_gap_start_time, data_gap_end_time, curr, last, num_extra_added = _make_data_array_helper(bg_df, time_array, value_array, data_gap_start_time, data_gap_end_time,
                                                                                                                                            start_index, index, curr, last, num_extra_added, 'openaps', 'suggested', item_str)
                except:

                    if item_str == 'IOB':
                        #if IOB, search for iob column
                        try:
                            time_array, value_array, data_gap_start_time, data_gap_end_time, curr, last, num_extra_added = _make_data_array_helper(bg_df, time_array, value_array, data_gap_start_time, data_gap_end_time,
                                                                                                                                                    start_index, index, curr, last, num_extra_added, 'openaps', 'iob', 'iob')
                        except:
                            miss += 1
                    else:
                        #count the number of misses
                        miss += 1

        #Check to see if the number added exceeds EXTRA_SPACE. If it does, raise exception and change it in the global variables above
        if num_extra_added > EXTRA_SPACE:
            raise Exception("The number of data added to fill in data gaps exceeds array size. Add more to EXTRA_SPACE.")

        #Resize the array to fit the data
        time_array = np.resize(time_array, total_len - miss + num_extra_added - EXTRA_SPACE)
        value_array = np.resize(value_array, total_len - miss + num_extra_added - EXTRA_SPACE)

        return time_array, value_array, data_gap_start_time, data_gap_end_time


#Depending on the size of the array, this function returns the number of fourier series to be used on the lomb scargle
#with 320 as the maximum
def _get_num_fourier_series(size):
    if size <= 1000:
        return 40
    if size <= 2000:
        return 80
    elif size <= 4000:
        return 160
    else:
        return 320


#This function runs the lomb scargle given the time_array, the value_array, the period, and the number of fourier terms
def _run_lomb_scargle(time_array, value_array, period):
    lomb = gatspy.periodic.LombScargle(Nterms=_get_num_fourier_series(len(time_array)), fit_period=True, optimizer_kwds={'quiet':True})
    lomb.optimizer.set(period_range=(int(time_array.max()) - PERIOD_RANGE_MAX, int(time_array.max()) - PERIOD_RANGE_MIN), first_pass_coverage=FIRST_PASS_COVERAGE_NUM)
    lomb.fit(time_array, value_array)

    return lomb.predict(period)


#Plot lomb scargle function
def _plot_lomb(period, lomb, time_array, value_array, name_str):
    plt.plot(time_array, value_array, label='Actual ' + name_str)
    plt.plot(period, lomb, label='Lomb ' + name_str)


#This function gets the data from the lomb scargle. It takes in the start and end indices and returns a lomb scargle model for BG, IOB, and COB as well as the period
def _get_lomb_scargle(bg_df, start_index, end_index, plot_lomb_array):
        bg_time_array, bg_value_array, bg_gap_start_time, bg_gap_end_time = _make_data_array(bg_df, start_index, end_index, 'bg')
        iob_time_array, iob_value_array, iob_gap_start_time, iob_gap_end_time = _make_data_array(bg_df, start_index, end_index, 'IOB')
        cob_time_array, cob_value_array, cob_gap_start_time, cob_gap_end_time = _make_data_array(bg_df, start_index, end_index, 'COB')

        #Keep track of the data start and end times in the array
        data_gap_start_time = bg_gap_start_time + iob_gap_start_time + cob_gap_start_time
        data_gap_end_time = bg_gap_end_time + iob_gap_end_time + cob_gap_end_time

        period = np.linspace(0, int(bg_time_array.max()), int(bg_time_array.max()) + 1) #set period to be as large as possible

        bg_lomb = _run_lomb_scargle(bg_time_array, bg_value_array, period)
        iob_lomb = _run_lomb_scargle(iob_time_array, iob_value_array, period)
        cob_lomb = _run_lomb_scargle(cob_time_array, cob_value_array, period)

        #Set all bg/cob values below zero equal to zero (iob can be negative if it is below baseline levels)
        bg_lomb[bg_lomb < 0] = 0
        cob_lomb[cob_lomb < 0] = 0

        #Plot lomb-scargle if values in the plot_lomb_array
        if len(plot_lomb_array) > 0:
            plt.clf()
            if "bg" in plot_lomb_array: _plot_lomb(period, bg_lomb, bg_time_array, bg_value_array, "BG")
            if "iob" in plot_lomb_array: _plot_lomb(period, iob_lomb, iob_time_array, iob_value_array, "IOB")
            if "cob" in plot_lomb_array: _plot_lomb(period, cob_lomb, cob_time_array, cob_value_array, "COB")
            plt.legend(loc='upper left')
            plt.show()

        return period, bg_lomb, iob_lomb, cob_lomb, data_gap_start_time, data_gap_end_time


#Class to store the lomb data
class LombData(object):
     def __init__(self, period, bg_lomb, iob_lomb, cob_lomb, time_value_array, data_gap_start_time, data_gap_end_time):
         self.period = period
         self.bg_lomb = bg_lomb
         self.iob_lomb = iob_lomb
         self.cob_lomb = cob_lomb
         self.time_value_array = time_value_array
         self.data_gap_start_time = data_gap_start_time
         self.data_gap_end_time = data_gap_end_time


#The main function to be called to get the bg data arrays
#It applies the lomb scargle periodogram to make a model of the data, and returns this model as an array
def get_lomb_data(bg_df, start_index, end_index, plot_lomb_array):
    """
    Function to make a lomb-scargle periodogram from the OpenAPS data in order to get
    data for every minute, not just every 5 minutes. It takes in the dataframe
    and the start and stop indices along with the plot_lomb_array, which is an array
    with the values that allow you to plot the lomb-scargle values with matplotlib.

    Input:      bg_df                           Pandas dataframe of all of the data from ./data/[id_str]/devicestatus.json
                start_index                     The start index of the set. Should be higher in value than end_index, as
                                                    the earlier times have higher indices
                end_index                       The end index of the set. Should be lower in value than start_index, as
                                                    the later times have lower indices. Inclusive, so this index is included in the data
                plot_lomb_array                 Array with the types to be plotted as strings. e.g. ['bg','iob','cob']
.
    Output:     lomb_data                       The namedtuple holding the lomb-scargle data with these arrays:
                                                    ['period', 'bg_lomb', 'iob_lomb', 'cob_lomb', 'time_value_array', data_gap_start_time, data_gap_end_time]
                                                    data_gap_start_time: Array with the start times of the data gaps that will be skipped
                                                    The indices of this and data_gap_end_time correspond to the same data gap
                                                    data_gap_end_time: Array with the end times of the data gaps that will be skipped
                                                    The indices of this and data_gap_start_time correspond to the same data gap
    Usage:      train_lomb_data = get_lomb_data(bg_df, start_train_index, end_train_index, ['bg', 'iob'])
    """

    #Make the time_value_array, which is the array of hours from midnight
    time_value_array = _make_time_value_array(bg_df, start_index, end_index)
    period, bg_lomb, iob_lomb, cob_lomb, data_gap_start_time, data_gap_end_time = _get_lomb_scargle(bg_df, start_index, end_index, plot_lomb_array)

    lomb_data = LombData(period, bg_lomb, iob_lomb, cob_lomb, time_value_array, data_gap_start_time, data_gap_end_time)

    return lomb_data
