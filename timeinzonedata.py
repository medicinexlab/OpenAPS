"""
timeinzonedata.py
These are the methods used nby maintimeinzone.py to find the number of entries in the zone.

MedicineX OpenAPS
2017-8-30
"""
import numpy as np

NUMBER_TIME_ZONES = 8

#Hour of day that night starts
NIGHT_START_HOUR = 0
#Hour of day that night ends
NIGHT_END_HOUR = 7

#Returns array of bg_array and bg_time_array
def get_bg_array(bg_df, start_index, end_index):
    array_length = start_index - end_index + 1
    bg_array = np.zeros(array_length)
    bg_time_array = []
    curr = 0
    miss = 0

    for df_index in range(start_index, end_index - 1, -1):
        try:
            time = bg_df.iloc[df_index]['created_at']
            bg_array[curr] = bg_df.iloc[df_index]['openaps']['enacted']['bg']
            bg_time_array.append(time)
            curr += 1

        except:
            try:
                bg_array[curr] = bg_df.iloc[df_index]['openaps']['suggested']['bg']
                bg_time_array.append(time)
                curr += 1

            except:
                miss += 1

    bg_array = np.resize(bg_array, curr)
    return bg_array, bg_time_array


#Function to fill in the time zones array
def _fill_time_zones(bg_value, time_zones):
    if bg_value <= 50:
        time_zones[0] += 1
    if bg_value <= 60:
        time_zones[1] += 1
    if bg_value <= 70:
        time_zones[2] += 1
    if bg_value > 70 and bg_value <= 180:
        time_zones[3] += 1 #The target range
    if bg_value > 180:
        time_zones[4] += 1
    if bg_value > 250:
        time_zones[5] += 1
    if bg_value > 300:
        time_zones[6] += 1
    if bg_value > 350:
        time_zones[7] += 1

    return time_zones


#Gets the number of entries in different zones
def get_time_in_zone(bg_array, bg_time_array):
    night_bg_array = []

    time_zones = np.zeros(NUMBER_TIME_ZONES)
    night_time_zones = np.zeros(NUMBER_TIME_ZONES)
    night_total = 0

    for array_index in range(len(bg_array)):
        time_zones = _fill_time_zones(bg_array[array_index], time_zones)

        if bg_time_array[array_index].hour >= NIGHT_START_HOUR and bg_time_array[array_index].hour <= NIGHT_END_HOUR:
            night_time_zones = _fill_time_zones(bg_array[array_index], night_time_zones)
            night_bg_array.append(bg_array[array_index])
            night_total += 1

    #Print out the data
    print "<= 50 mg/dL: " + str(time_zones[0])
    print "<= 60 mg/dL: " + str(time_zones[1])
    print "<= 70 mg/dL: " + str(time_zones[2])
    print ">70 and <=180 mg/dL: " + str(time_zones[3])
    print "> 180 mg/dL: " + str(time_zones[4])
    print "> 250 mg/dL: " + str(time_zones[5])
    print "> 300 mg/dL: " + str(time_zones[6])
    print "> 350 mg/dL: " + str(time_zones[7])
    print "Total: " + str(len(bg_array))
    print "Average BG: {}".format(np.average(bg_array))
    print

    print "Night <= 50 mg/dL: " + str(night_time_zones[0])
    print "Night <= 60 mg/dL: " + str(night_time_zones[1])
    print "Night <= 70 mg/dL: " + str(night_time_zones[2])
    print "Night >70 and <=180 mg/dL: " + str(night_time_zones[3])
    print "Night > 180 mg/dL: " + str(night_time_zones[4])
    print "Night > 250 mg/dL: " + str(night_time_zones[5])
    print "Night > 300 mg/dL: " + str(night_time_zones[6])
    print "Night > 350 mg/dL: " + str(night_time_zones[7])
    print "Night Total: " + str(night_total)
    print "Average Night BG: {}".format(np.average(night_bg_array))
    print

    print "Percent <= 50 mg/dL: " + str(float(time_zones[0])/len(bg_array))
    print "Percent <= 60 mg/dL: " + str(float(time_zones[1])/len(bg_array))
    print "Percent <= 70 mg/dL: " + str(float(time_zones[2])/len(bg_array))
    print "Percent >70 and <=180 mg/dL: " + str(float(time_zones[3])/len(bg_array))
    print "Percent > 180 mg/dL: " + str(float(time_zones[4])/len(bg_array))
    print "Percent > 250 mg/dL: " + str(float(time_zones[5])/len(bg_array))
    print "Percent > 300 mg/dL: " + str(float(time_zones[6])/len(bg_array))
    print "Percent > 350 mg/dL: " + str(float(time_zones[7])/len(bg_array))
    print

    print "Percent Night <= 50 mg/dL: " + str(float(night_time_zones[0])/night_total)
    print "Percent Night <= 60 mg/dL: " + str(float(night_time_zones[1])/night_total)
    print "Percent Night <= 70 mg/dL: " + str(float(night_time_zones[2])/night_total)
    print "Percent Night >70 and <=180 mg/dL: " + str(float(night_time_zones[3])/night_total)
    print "Percent Night > 180 mg/dL: " + str(float(night_time_zones[4])/night_total)
    print "Percent Night > 250 mg/dL: " + str(float(night_time_zones[5])/night_total)
    print "Percent Night > 300 mg/dL: " + str(float(night_time_zones[6])/night_total)
    print "Percent Night > 350 mg/dL: " + str(float(night_time_zones[7])/night_total)
    print

    return time_zones, night_time_zones
