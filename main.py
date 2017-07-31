'''
mainpred.py
This file is the main code for running the prediction algorithms for OpenAPS.

The OpenAPS data must be in a subdirectory called "data" with a subdirectory of
the ID number that contains the devicestatus.json file. For example:
        ./data/00897741/devicestatus.json
        ./data/01352464/devicestatus.json
        ./data/01884126/devicestatus.json
        ./data/14092221/devicestatus.json
        ./data/15634563/devicestatus.json
        ./data/17161370/devicestatus.json
        ./data/24587372/devicestatus.json
        ./data/40997757/devicestatus.json
        ./data/41663654/devicestatus.json
        ./data/45025419/devicestatus.json
        ./data/46966807/devicestatus.json
        ./data/68267781/devicestatus.json
        ./data/84984656/devicestatus.json
where . represents the current directory with the code.

The code requires the following files:
        bgdataframe.py
        bgarray.py
        datamatrix.py
        mlalgorithm.py
        ClarkeErrorGrid.py

This code also requires the following libraries:
        pandas
        numpy
        gatspy
        sklearn

Trevor Tsue
2017-7-24
'''

from collections import namedtuple
from bgdataframe import get_bg_data
from bgarray import get_bg_array
from datamatrix import make_data_matrix
from oldpred import get_old_pred
from oldpred import analyze_old_pred_data
from mlalgorithm import *
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn import svm
from sklearn import neural_network



#DATA CONSTANTS MODIFY THESE TO RUN DATA

#Array of the ID to use
ID_ARRAY = np.array(["40997757", "41663654", "45025419", "46966807", "68267781", "84984656"])
#Array of the data minutes that will be tested. e.g. [1,15,30,45,60,75,90,105,120]
DATA_MINUTES_ARRAY = np.array([120])
#Array of the minutes in the future that the predictions will be made for.
PRED_MINUTES_ARRAY = np.array([30])
#Choose whether to run 'eventualBG', 'iob', 'cob', 'acob'
#e.g. ['iob', 'acob']
#Leave empty to run none
OLD_PRED_ALGORITHM_ARRAY = np.array([])
#Array of the algorithms that will be tested.
ALGORITHM_ARRAY = np.array(["Ridge Regression"])




#PLOTTING CONSTANTS

#List of lomb-scargle plots to print
#Leave empty to print none. Otherwise, use 'bg','iob','cob'. e.g. ['bg','cob']
PLOT_LOMB_ARRAY = np.array([])

#Boolean to show the prediction plot versus the actual bg
SHOW_PRED_PLOT = False
#Boolean to save the prediction plot
SAVE_PRED_PLOT = False
#Boolean to show the Clarke Error Grid plot
SHOW_CLARKE_PLOT = False
#Boolean to save the Clarke Error Grid plot
SAVE_CLARKE_PLOT = False



#ALGORITHM CONTANTS
VARIABLE_VALUE_ARRAY = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24])


def linear_regression_model(variable_index_keeper):
    return linear_model.LinearRegression(normalize=True)

def ridge_regression_model(variable_index_keeper):
    variable_str = str(variable_index_keeper)
    alpha_value = VARIABLE_VALUE_ARRAY[int(variable_str[len(variable_str) - 1])] #alpha value index is first digit
    return linear_model.Ridge(normalize=True, alpha=alpha_value), alpha_value

def svm_linear_regression(variable_index_keeper):
    variable_str = str(variable_index_keeper)
    c_value = VARIABLE_VALUE_ARRAY[int(variable_str[len(variable_str) - 1])] #c value index is first digit
    epsilon_value = VARIABLE_VALUE_ARRAY[int(variable_str[len(variable_str) - 2])] #epsilon value index is second digit
    return svm.SVR(kernel='linear', C=c_value, epsilon=epsilon_value)


#Dictionary with the name of the algorithm as the key and the function as the value
ALGORITHM_DICT = {"Linear Regression":linear_regression_model,
                    "Ridge Regression":ridge_regression_model,
                    "SVM Linear Regression":svm_linear_regression,
                    "Neural Network":neural_network.MLPRegressor(activation='identity')
                    }

ALGORITHM_NUM_VARIABLES = {"Linear Regression":0,
                        "Ridge Regression":1,
                        "SVM Linear Regression":2
                        }

#Dictionary with the name of the algorithm as the key and boolean to apply the StandardScaler transformation as the value
ALGORITHM_TRANSFORM = {"Linear Regression":False,
                        "Ridge Regression":False,
                        "SVM Linear Regression":True,
                        "Neural Network":True
                        }


#ID CONSTANTS
ID_LIST = ["01352464", "14092221", "15634563", "40997757", "46966807", "68267781"]
#They work, but they don't have enough days to make accurate data charts
WORK_BUT_SMALL_ID_LIST = ["00897741", "17161370", "41663654", "45025419", "84984656"]
#THESE DON'T WORK!!!!! (e.g. they have loop algorithm only)
BAD_ID_LIST = ["01884126", "24587372"]

#Start and end training set dates
START_TRAIN_DATE_DICT = {"00897741": "2017-02-03", "01352464": "2016-10-09", "01884126": "2017-02-03",
                            "14092221": "2017-01-06", "15634563": "2017-01-02", "17161370": "2017-02-09",
                            "24587372": "2017-01-10", "40997757": "2017-01-03", "41663654": "2017-01-25",
                            "45025419": "2017-02-10", "46966807": "2017-01-02", "68267781": "2016-12-07",
                            "84984656": "2017-01-08"}

END_TRAIN_DATE_DICT = {"00897741": "2017-02-09", "01352464": "2016-10-18", "01884126": "2017-02-07",
                            "14092221": "2017-01-19", "15634563": "2017-01-15", "17161370": "2017-02-13",
                            "24587372": "2017-01-21", "40997757": "2017-01-16", "41663654": "2017-01-25",
                            "45025419": "2017-02-15", "46966807": "2017-01-15", "68267781": "2016-12-20",
                            "84984656": "2017-01-24"}

#Start and end validation set dates
START_VALID_DATE_DICT = {"00897741": "2017-02-03", "01352464": "2016-10-19", "01884126": "2017-02-03",
                            "14092221": "2017-01-06", "15634563": "2017-01-16", "17161370": "2017-02-09",
                            "24587372": "2017-01-10", "40997757": "2017-01-17", "41663654": "2017-01-25",
                            "45025419": "2017-02-10", "46966807": "2017-01-16", "68267781": "2016-12-21",
                            "84984656": "2017-01-08"}

END_VALID_DATE_DICT = {"00897741": "2017-02-09", "01352464": "2016-10-22", "01884126": "2017-02-07",
                            "14092221": "2017-01-19", "15634563": "2017-01-19", "17161370": "2017-02-13",
                            "24587372": "2017-01-21", "40997757": "2017-01-20", "41663654": "2017-01-25",
                            "45025419": "2017-02-15", "46966807": "2017-01-19", "68267781": "2016-12-24",
                            "84984656": "2017-01-24"}

#Start and end test set dates
START_TEST_DATE_DICT = {"00897741": "2017-02-10", "01352464": "2016-10-23", "01884126": "2017-02-08",
                            "14092221": "2017-01-20", "15634563": "2017-01-20", "17161370": "2017-02-14",
                            "24587372": "2017-01-22", "40997757": "2017-01-21", "41663654": "2017-01-26",
                            "45025419": "2017-02-16", "46966807": "2017-01-20", "68267781": "2016-12-25",
                            "84984656": "2017-01-25"}

END_TEST_DATE_DICT = {"00897741": "2017-02-12", "01352464": "2016-10-29", "01884126": "2017-02-09",
                            "14092221": "2017-01-26", "15634563": "2017-01-26", "17161370": "2017-02-15",
                            "24587372": "2017-01-11", "40997757": "2017-01-27", "41663654": "2017-01-26",
                            "45025419": "2017-02-17", "46966807": "2017-01-26", "68267781": "2016-12-31",
                            "84984656": "2017-01-26"}


#Function to return the training, validaiton, and testing set dates given the string of the ID number
def _get_id_dates(id_str):
    return START_TRAIN_DATE_DICT[id_str], END_TRAIN_DATE_DICT[id_str], START_VALID_DATE_DICT[id_str], END_VALID_DATE_DICT[id_str], START_TEST_DATE_DICT[id_str], END_TEST_DATE_DICT[id_str]


#Main function
def main():
    for id_str in ID_ARRAY:
        print "\nID Number: " + id_str
        start_train_str, end_train_str, start_valid_str, end_valid_str, start_test_str, end_test_str = _get_id_dates(id_str)
        bg_df, start_train_index, end_train_index, start_valid_index, end_valid_index, start_test_index, end_test_index = get_bg_data(id_str, start_train_str, end_train_str,
                                                                                                                                        start_valid_str, end_valid_str, start_test_str, end_test_str)
        train_lomb_data, valid_lomb_data, test_lomb_data = get_bg_array(bg_df, start_train_index, end_train_index, start_valid_index, end_valid_index, start_test_index, end_test_index, PLOT_LOMB_ARRAY)

        for pred_minutes in PRED_MINUTES_ARRAY:
            print "    Prediction Minutes: " + str(pred_minutes)

            #Analyze old pred methods
            if len(OLD_PRED_ALGORITHM_ARRAY) != 0:
                if pred_minutes % 5 != 0: raise Exception("The prediction minutes is not a multiple of 5.")
                eventual_pred_data, iob_pred_data, cob_pred_data, acob_pred_data = get_old_pred(bg_df, start_test_index, end_test_index, pred_minutes)
                if 'eventualBG' in OLD_PRED_ALGORITHM_ARRAY: analyze_old_pred_data(eventual_pred_data, SHOW_PRED_PLOT, SAVE_PRED_PLOT, SHOW_CLARKE_PLOT, SAVE_CLARKE_PLOT, id_str, "eventualBG", "Pred" + str(pred_minutes))
                if 'iob' in OLD_PRED_ALGORITHM_ARRAY: analyze_old_pred_data(iob_pred_data, SHOW_PRED_PLOT, SAVE_PRED_PLOT, SHOW_CLARKE_PLOT, SAVE_CLARKE_PLOT, id_str, "IOB", "Pred" + str(pred_minutes))
                if 'cob' in OLD_PRED_ALGORITHM_ARRAY: analyze_old_pred_data(cob_pred_data, SHOW_PRED_PLOT, SAVE_PRED_PLOT, SHOW_CLARKE_PLOT, SAVE_CLARKE_PLOT, id_str, "COB", "Pred" + str(pred_minutes))
                if 'acob' in OLD_PRED_ALGORITHM_ARRAY: analyze_old_pred_data(acob_pred_data, SHOW_PRED_PLOT, SAVE_PRED_PLOT, SHOW_CLARKE_PLOT, SAVE_CLARKE_PLOT, id_str, "aCOB", "Pred" + str(pred_minutes))

            for data_minutes in DATA_MINUTES_ARRAY:
                print "        Data Minutes: " + str(data_minutes)

                train_data_matrix, actual_bg_train_array = make_data_matrix(bg_df, train_lomb_data, start_train_index, end_train_index, data_minutes, pred_minutes)
                valid_data_matrix, actual_bg_valid_array = make_data_matrix(bg_df, valid_lomb_data, start_valid_index, end_valid_index, data_minutes, pred_minutes)
                test_data_matrix, actual_bg_test_array = make_data_matrix(bg_df, test_lomb_data, start_test_index, end_test_index, data_minutes, pred_minutes)


                #Analyze ml algorithms
                for algorithm_str in ALGORITHM_ARRAY:
                    print "            Algorithm: " + algorithm_str

                    #Since we have 10 variable values for each variable, when we validate, we have a power of 10 models to run
                    #Each digit represents a different variable value. If there is one variable, we have 0-9
                    #If there are two variables, the first variable is controlled by X0 - X9, and the second is 0X - 9X
                    #With three variables, the third is controlled by 0XY - 9XY, and this trend is continued
                    for variable_index_keeper in range(10**ALGORITHM_NUM_VARIABLES[algorithm_str]):
                        if ALGORITHM_TRANSFORM[algorithm_str]:
                            train_data_matrix, valid_data_matrix, test_data_matrix = preprocess_data(train_data_matrix, valid_data_matrix, test_data_matrix)

                        reg_model = ALGORITHM_DICT[algorithm_str](variable_index_keeper)
                        reg_model = reg_model.fit(train_data_matrix, actual_bg_train_array)


                        valid_prediction = reg_model.predict(valid_data_matrix)
                        error_value = mean_squared_error(actual_bg_valid_array, valid_prediction)

                        if variable_index_keeper == 0 or error_value < best_error_value:
                            best_error_value = error_value
                            best_reg_model = reg_model
                            best_variable_index_keeper = variable_index_keeper


                print "                Best Validation RMSE: " + str(math.sqrt(best_error_value))
                print "                Best Validation Variable Value: " + str(best_variable_index_keeper)
                print "                Best Validation Parameters: " + str(best_reg_model.get_params())
                test_prediction = best_reg_model.predict(test_data_matrix)
                analyze_ml_data(actual_bg_test_array, test_prediction, SHOW_PRED_PLOT, SAVE_PRED_PLOT, SHOW_CLARKE_PLOT, SAVE_CLARKE_PLOT, id_str, algorithm_str,
                                "Pred" + str(pred_minutes) + "Data" + str(data_minutes))


#Run the main function
if __name__ ==  "__main__":
    main()
